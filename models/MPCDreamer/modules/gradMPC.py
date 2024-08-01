import torch
from tensordict import TensorDict, TensorDictBase
from torch import nn

from torchrl.envs.common import EnvBase
from torchrl.modules.planners.common import MPCPlannerBase

from mpc.mpc.mpc import MPC


class gradMPCPlanner(MPCPlannerBase):
    """
    gradMPC
    """

    def __init__(
        self,
        env: EnvBase,
        planning_horizon: int,
        actor_module, 
        value_estimator,
        num_candidates: int,
        reward_key: str = ("next", "reward"),
        action_key: str = "action",
        mpc_config = None
    ):
        super().__init__(env=env, action_key=action_key)
        self.num_candidates = num_candidates
        self.reward_key = reward_key
        self.planning_horizon = planning_horizon
        self.actor_module = actor_module
        self.mpc_module = MPC(
            mpc_config,
            self.as_mpc_operation,
            shape=(self.planning_horizon, *self.env.action_spec.shape),
        )
        self.value_estimator = value_estimator
        
       
    def as_mpc_operation(self):
        def _calculate_return(inputs, start_tensordict):
            actions = self.env.action_spec.project(inputs)
            policy = _PrecomputedActionsSequentialSetter(actions)
            optim_tensordict = self.env.rollout(
                max_steps=self.planning_horizon,
                policy=policy,
                auto_reset=False,
                tensordict=start_tensordict,
            )

            self.value_estimator(optim_tensordict)
            value = optim_tensordict["value"]    
        
            return dict(
                loss = -value.sum(),
            )
        return _calculate_return
            

    def planning(self, tensordict: TensorDictBase) -> torch.Tensor:
        batch_size = tensordict.batch_size
        
        expanded_original_tensordict = (
            tensordict.unsqueeze(-1)
            .expand(*batch_size, self.num_candidates)
            .to_tensordict()
        )
        with torch.no_grad():
            init_tensordict = self.env.rollout(
                max_steps=self.planning_horizon,
                policy=self.actor_module,
                auto_reset=False,
                tensordict=expanded_original_tensordict.copy(),
            )
            init_actions = init_tensordict["action"]
        operation_kwargs = {"start_tensordict": expanded_original_tensordict}
        
        optimized, results = self.mpc_module(
            n_samples=self.num_candidates,
            operation_kwargs=operation_kwargs,
            init=init_actions,
        )
        
        # TODO: choose the one with the best return from the results
        
        return optimized[..., 0, 0, :]


class _PrecomputedActionsSequentialSetter:
    def __init__(self, actions):
        self.actions = actions
        self.cmpt = 0

    def __call__(self, tensordict):
        # checks that the step count is lower or equal to the horizon
        if self.cmpt >= self.actions.shape[-2]:
            raise ValueError("Precomputed actions sequence is too short")
        tensordict = tensordict.set("action", self.actions[..., self.cmpt, :])
        self.cmpt += 1
        return tensordict