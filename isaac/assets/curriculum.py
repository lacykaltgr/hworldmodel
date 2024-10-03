import torch

from omni.isaac.lab.envs import ManagerBasedRLEnv
from omni.isaac.lab.managers import TerminationTermCfg as DoneTerm
from omni.isaac.lab.managers import EventTermCfg as EventTerm
from omni.isaac.lab.managers import SceneEntityCfg
import omni.isaac.lab_tasks.manager_based.locomotion.velocity.mdp as mdp
from collections.abc import Sequence
import wandb 
import open3d as o3d
import numpy as np
import math

num_envs = 4
reset_flags = [False] * num_envs
done_count = 0
  
def enable_base_collision(env: ManagerBasedRLEnv) -> torch.Tensor:
    """
    Re-enables the base contact termination after the robot has learned to stand.
    """
    term_name = "base_contact"
    base_contact = DoneTerm(func=mdp.illegal_contact, params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names="base"), "threshold": 700.0},)
    env.termination_manager.set_term_cfg("base_contact", base_contact)
    print("itt voltam")


def task_order(env: ManagerBasedRLEnv, env_ids, num_steps = 500) -> torch.Tensor:
    """
    Curriculum logic to switch between standing and walking tasks by altering termination conditions.
    """
    # Check the current curriculum stage
    wandb.log({"common_step_counter": int(env.common_step_counter)})
    if int(env.common_step_counter) >= num_steps and int(env.common_step_counter) <= num_steps + 100:
        enable_base_collision(env)
        return 1   # Enable base contact termination for walking
    return 0


def sample_from_nodes(nodes_pcd_path="/home/czimber_mark/hworldmodel/isaac/assets/pointclouds/nodes_57.pcd"):
    pcd = o3d.io.read_point_cloud(nodes_pcd_path)

    # Extract the points as a numpy array
    points = np.asarray(pcd.points, dtype=np.float32)

    # Remove the z-coordinate, keeping only x and y
    points_2d = points[:, :2]

    # Sample a random point using the local random generator
    sample_point = points_2d[np.random.randint(len(points_2d))]

    return sample_point

def node_based_termiantions(env: ManagerBasedRLEnv, env_ids: Sequence[int], threshold = 10) -> torch.Tensor: 
    # Load the PCD file
    global done_count
    global reset_flags

    start_goal_pairs = 100

    # Check if any environment has reset in this step
    if env.termination_manager.dones.any():
        # Iterate over each environment
        for i in range(len(reset_flags)):
            if env.termination_manager.dones[i]:
                # Mark this environment as having reset
                reset_flags[i] = True

        # If all environments have reset (i.e., all elements of reset_flags are True)
        if all(reset_flags):
            # Increment done_count since all environments have reset
            done_count += 1
            mean_distance = 0
            pose_command = env.command_manager.get_term("pose_command")
            agent_pos = pose_command.robot.data.root_pos_w 
            goal_pos = pose_command.pos_command_w
            for i in range(len(reset_flags)):
                mean_distance += math.hypot(agent_pos[i, 0] - goal_pos[i, 0], agent_pos[i, 1] - goal_pos[i, 1])
            wandb.log({"mean_distance_from_goal": mean_distance / len(reset_flags)})

            # Reset the reset_flags for the next round of tracking
            reset_flags = [False] * len(reset_flags)

    if(done_count % start_goal_pairs == 0 and done_count != 0):

        # Sample a random element from points_2d
        reset_point = sample_from_nodes()
        #print(f"Robot's Current position: {reset_point}")

        reset_base = EventTerm(
            func=mdp.reset_root_state_uniform,
            mode="reset",
            params={
                "pose_range": {"x": (reset_point[0], reset_point[0]), "y": (reset_point[1], reset_point[1]), "yaw": (-3.14, 3.14)},
                "velocity_range": {
                    "x": (-0.0, -0.0),
                    "y": (-0.0, 0.0),
                    "z": (-0.0, 0.0),
                    "roll": (-0.0, 0.0),
                    "pitch": (-0.0, 0.0),
                    "yaw": (-0.0, 0.0),
                    },
                },
            )

        env.event_manager.set_term_cfg("reset_base", reset_base)
        
        # Keep resampling the second point until it meets the distance threshold
        while True:
            # Sample a second random point
            command_point = sample_from_nodes()

            # Calculate the Euclidean distance between the two points
            distance = math.hypot(reset_point[0] - command_point[0], reset_point[1] - command_point[1])

            # Check if the distance is within the threshold
            if distance < threshold and distance > 4:
                break  # Stop resampling if the condition is met

        pose_command = env.command_manager.get_term("pose_command")
        # print(pose_command)
        pose_command.points = command_point

    return 0
    