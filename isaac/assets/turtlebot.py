import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.actuators import ActuatorNetMLPCfg, IdealPDActuatorCfg, ImplicitActuatorCfg, ActuatorBaseCfg
from omni.isaac.lab.assets.articulation import ArticulationCfg
from omni.isaac.lab.utils.assets import ISAACLAB_NUCLEUS_DIR
from omni.isaac.lab.sim.schemas import JointDrivePropertiesCfg


##
# Configuration
##

TURTLEBOT4_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"/hworldmodel/isaac/turtlebot4/turtlebot4/turtlebot4.usd",
        joint_drive_props=JointDrivePropertiesCfg(
            drive_type="acceleration",
        )
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(-2.0, 2.0, 0.0),
        joint_pos={".*": 0.0},
        joint_vel={".*": 0.0},
    ),
    actuators={
        "left_wheel": IdealPDActuatorCfg(
            joint_names_expr=["left_wheel_joint*"],
            stiffness=0.0,
            damping=5000,
            effort_limit=8.0,
            velocity_limit=8.0,
            friction=10.0,
        ),
        "right_wheel": IdealPDActuatorCfg(
            joint_names_expr=["right_wheel_joint*"],
            stiffness=0.0,
            damping=5000,
            effort_limit=8.0,
            velocity_limit=8.0,
            friction=10.0,
        ),
    },
)
"""
Configuration of Turtlebot4 using DC motor.
"""

