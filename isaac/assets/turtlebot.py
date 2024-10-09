import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.actuators import ActuatorNetMLPCfg, IdealPDActuatorCfg, ImplicitActuatorCfg, ActuatorBaseCfg, ActuatorNetLSTMCfg
from omni.isaac.lab.assets.articulation import ArticulationCfg
from omni.isaac.lab.utils.assets import ISAACLAB_NUCLEUS_DIR
from omni.isaac.lab.sim.schemas import JointDrivePropertiesCfg
from omni.isaac.lab.utils.assets import ISAACLAB_NUCLEUS_DIR
from omni.isaac.lab.sensors import RayCasterCfg
from omni.isaac.lab.utils.assets import ISAACLAB_NUCLEUS_DIR
from omni.isaac.lab_assets.velodyne import VELODYNE_VLP_16_RAYCASTER_CFG


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
            stiffness=1000.0,
            damping=500,
            effort_limit=8.0,
            velocity_limit=8.0,
            friction=1.0,
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

NOVABOT_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"http://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/4.0/Isaac/Robots/Carter/nova_carter.usd",
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
            joint_names_expr=["joint_wheel_.*"],
            stiffness=0.0,
            damping=3.0,
            effort_limit=15.0,
            velocity_limit=15.0,
            friction=0.1,
        ),
    },
)

"""Configuration for ANYdrive 3.0 (used on ANYmal-C) with LSTM actuator model."""
ANYDRIVE_3_LSTM_ACTUATOR_CFG = ActuatorNetLSTMCfg(
    joint_names_expr=[".*HAA", ".*HFE", ".*KFE"],
    network_file=f"{ISAACLAB_NUCLEUS_DIR}/ActuatorNets/ANYbotics/anydrive_3_lstm_jit.pt",
    saturation_effort=120.0,
    effort_limit=80.0,
    velocity_limit=7.5,
)

"""Configuration of ANYmal-C robot using actuator-net."""
SATIDOG_CFG = ArticulationCfg(
   spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAACLAB_NUCLEUS_DIR}/Robots/ANYbotics/ANYmal-C/anymal_c.usd",
        # usd_path=f"{ISAAC_NUCLEUS_DIR}/Robots/ANYbotics/anymal_instanceable.usd",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True, solver_position_iteration_count=4, solver_velocity_iteration_count=0
        ),
        # collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.02, rest_offset=0.0),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(1.0, 0.0, 0.6),
        joint_pos={
            ".*HAA": 0.0,  # all HAA
            ".*F_HFE": 0.4,  # both front HFE
            ".*H_HFE": -0.4,  # both hind HFE
            ".*F_KFE": -0.8,  # both front KFE
            ".*H_KFE": 0.8,  # both hind KFE
        },
    ),
    actuators={"legs": ANYDRIVE_3_LSTM_ACTUATOR_CFG},
    soft_joint_pos_limit_factor=0.95,
)


ANYMAL_LIDAR_CFG = VELODYNE_VLP_16_RAYCASTER_CFG.replace(
    offset=RayCasterCfg.OffsetCfg(pos=(-0.310, 0.000, 0.159), rot=(0.0, 0.0, 0.0, 1.0))
)
"""Configuration for the Velodyne VLP-16 sensor mounted on the ANYmal robot's base."""
