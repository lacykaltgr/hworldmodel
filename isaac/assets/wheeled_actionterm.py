import torch
import carb

from typing import List

from omni.isaac.lab.assets.articulation import Articulation
from omni.isaac.lab.managers.action_manager import ActionTerm, ActionTermCfg
from omni.isaac.lab.utils import configclass
from omni.isaac.lab.envs import ManagerBasedEnv
from collections.abc import Sequence
from .differential_controller import DifferentialController

##
# Custom action term
##




class WheeledRobotActionTerm(ActionTerm):

    cfg: ActionTermCfg
    """The configuration of the action term."""

    _asset: Articulation
    """The articulation asset on which the action term is applied."""

    _wheel_radius: float | torch.Tensor
    _wheel_base: float | torch.Tensor
    _max_linear_speed: float | torch.Tensor
    _max_angular_speed: float | torch.Tensor
    _max_wheel_speed: float | torch.Tensor

    _controller: DifferentialController

    def __init__(self, cfg, env: ManagerBasedEnv):
        # call super constructor
        super().__init__(cfg, env)

        self._joint_ids, self._joint_names = self._asset.find_joints(self.cfg.joint_names)
        self._num_joints = len(self._joint_ids)

        carb.log_info(
            f"Resolved joint names for the action term {self.__class__.__name__}:"
            f" {self._joint_names} [{self._joint_ids}]"
        )

        # Avoid indexing across all joints for efficiency
        if self._num_joints == self._asset.num_joints:
            self._joint_ids = slice(None)


        # create buffers
        self._raw_actions = torch.zeros(self.num_envs, self.action_dim, device=self.device)
        self._processed_actions = torch.zeros_like(self.raw_actions)

        self._wheel_radius = self.cfg.wheel_radius
        self._wheel_base = self.cfg.wheel_base
        self._max_linear_speed = self.cfg.max_linear_speed
        self._max_angular_speed = self.cfg.max_angular_speed
        self._max_wheel_speed = self.cfg.max_wheel_speed

        self._controller = DifferentialController(
            name="wheeled_robot_controller",
            wheel_radius=self._wheel_radius,
            wheel_base=self._wheel_base,
            max_linear_speed=self._max_linear_speed,
            max_angular_speed=self._max_angular_speed,
            max_wheel_speed=self._max_wheel_speed,
        )

        self.scale = cfg.scale


    """
    Properties.
    """

    @property
    def action_dim(self) -> int:
        # (throttle, steering)
        return 2

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions

    """
    Operations
    """

    def process_actions(self, actions: torch.Tensor):
        # store the raw actions
        self._raw_actions[:] = actions
        # no-processing of actions
        self._processed_actions[:] = self.scale * self._raw_actions[:]  # times scale

    def apply_actions(self):
        actions = self._processed_actions
        joint_actions = self._controller.forward(actions)
        self._asset.set_joint_velocity_target(joint_actions, joint_ids=self._joint_ids)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        self._raw_actions[env_ids] = 0.0


@configclass
class WheeledRobotActionTermCfg(ActionTermCfg):
    """Configuration for the cube action term."""

    class_type: type = WheeledRobotActionTerm
    """The class corresponding to the action term."""

    joint_names: List[str] = ["left_wheel_joint", "right_wheel_joint"]

    wheel_radius: float | torch.Tensor = 0.025
    wheel_base: float | torch.Tensor = 0.16
    max_linear_speed: float | torch.Tensor = 0.22
    max_angular_speed: float | torch.Tensor = 0.22
    max_wheel_speed: float | torch.Tensor = 8.0
    scale: float | torch.Tensor = 1.0