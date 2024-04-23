from dataclasses import dataclass

import torch
from torch import nn
from torchrl.collectors import MultiaSyncDataCollector
from torchrl.data import LazyMemmapStorage, MultiStep, TensorDictReplayBuffer



from tensordict import TensorDict
from tensordict.nn import TensorDictModule
from tensordict.utils import NestedKey
from torchrl.objectives.common import LossModule
from losses import loss

@loss(
    name="loss_world_model", 
    keys=["loss_model_distance", "loss_model_reco", "loss_model_reward", "loss_latent_pred_distance"]
)
class ModelLoss(LossModule):
    """
    Dreamer Model Loss.
    """

    @dataclass
    class _AcceptedKeys:
        """
        Maintains default values for all configurable tensordict keys.
        """

        reward: NestedKey = "reward"
        true_reward: NestedKey = "true_reward"
        prior: NestedKey = "prior_logits"
        posterior: NestedKey = "posterior_logits"
        pixels: NestedKey = "pixels"
        reco_pixels: NestedKey = "reco_pixels"

    default_keys = _AcceptedKeys()

    def __init__(
        self,
        world_model: TensorDictModule,
        *,
        lambda_distance: float = 1.0,
        lambda_reco: float = 1.0,
        lambda_reward: float = 1.0,
    ):
        super().__init__()
        self.world_model = world_model
        self.lambda_distance = lambda_distance
        self.lambda_reward = lambda_reward
        self.lambda_reco = lambda_reco

    def _forward_value_estimator_keys(self, **kwargs) -> None:
        pass

    def forward(self, tensordict: TensorDict) -> torch.Tensor:
        
        tensordict = tensordict.copy()
        tensordict.rename_key_(
            ("next", self.tensor_keys.reward),
            ("next", self.tensor_keys.true_reward),
        )
        tensordict = self.world_model(tensordict)

        # latent distance loss
        latents = tensordict.get(("next", "belief"))
        targets = tensordict.get(("next", "state_target")).detach()
        latent_distance_loss = torch.nn.MSELoss(reduction="mean")(targets, latents)
        
        # latent pred distance loss
        latents = tensordict.get("state").detach()
        targets = tensordict.get(("next", "belief"))
        latent_pred_distance_loss = -torch.nn.MSELoss(reduction="mean")(targets, latents)
        
        # reconstruction loss
        reco_targets = tensordict.get(("next", self.tensor_keys.pixels))
        reco_outputs = tensordict.get(("next", self.tensor_keys.reco_pixels))
        reco_loss = torch.nn.MSELoss(reduction="mean")(reco_targets, reco_outputs)

        # reward predictor loss
        dist = self.world_model[-1].get_dist(tensordict)
        reward_loss = -dist.log_prob(tensordict.get(("next", self.tensor_keys.true_reward))).mean()

        return (
            TensorDict(
                {
                    "loss_model_distance": self.lambda_distance * latent_distance_loss,
                    "loss_model_reco": self.lambda_reco * reco_loss,
                    "loss_model_reward": self.lambda_reward * reward_loss,
                    "loss_latent_pred_distance": 0.25 * latent_pred_distance_loss
                },
                [],
            ), 
            tensordict.detach(),   
        )