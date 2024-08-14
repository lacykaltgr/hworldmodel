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
        ),
        activate_contact_sensors=True
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.0),
        joint_pos={".*": 0.0},
        joint_vel={".*": 0.0},
    ),
    actuators={
        "wheels": IdealPDActuatorCfg(
            joint_names_expr=[".*_wheel_joint"],
            stiffness=0.0,
            damping=5000,
            effort_limit=8.0,
            velocity_limit=8.0,
            friction=15.0,
        ),
    },
)
"""
Configuration of Turtlebot4 using DC motor.
"""


TURTLEBOT3_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"http://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/4.0/Isaac/Robots/Turtlebot/turtlebot3_burger.usd",
        joint_drive_props=JointDrivePropertiesCfg(
            drive_type="acceleration",
        ),
        activate_contact_sensors=True
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.0),
        joint_pos={".*": 0.0},
        joint_vel={".*": 0.0},
    ),
    actuators={
        "wheels": IdealPDActuatorCfg(
            joint_names_expr=["wheel_.*_joint"],
            stiffness=0.0,
            damping=5000,
            effort_limit=8.0,
            velocity_limit=8.0,
            friction=15.0,
        ),
    },
)

