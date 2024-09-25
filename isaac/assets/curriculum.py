import torch

from omni.isaac.lab.envs import ManagerBasedRLEnv
from omni.isaac.lab.managers import TerminationTermCfg as DoneTerm
from omni.isaac.lab.managers import SceneEntityCfg
import omni.isaac.lab_tasks.manager_based.locomotion.velocity.mdp as mdp
import wandb 

  
def enable_base_collision(env: ManagerBasedRLEnv) -> torch.Tensor:
    """
    Re-enables the base contact termination after the robot has learned to stand.
    """
    term_name = "base_contact"
    base_contact = DoneTerm(func=mdp.illegal_contact, params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names="base"), "threshold": 1.0},)
    env.termination_manager.set_term_cfg("base_contact", base_contact)
    print("itt voltam")


def task_order(env: ManagerBasedRLEnv, env_ids, num_steps = 4200) -> torch.Tensor:
    """
    Curriculum logic to switch between standing and walking tasks by altering termination conditions.
    """
    # Check the current curriculum stage
    wandb.log({"common_step_counter": int(env.common_step_counter)})
    if int(env.common_step_counter) >= num_steps and int(env.common_step_counter) <= num_steps + 420:
        enable_base_collision(env)
        return 1   # Enable base contact termination for walking
    return 0