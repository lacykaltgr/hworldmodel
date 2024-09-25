# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import torch
import torch.nn as nn
from tensordict.nn import InteractionType
from torchrl.envs.utils import ExplorationType, set_exploration_type, check_env_specs
from torchrl.modules import (
    MLP,
    SafeModule,
    SafeProbabilisticModule,
    SafeProbabilisticTensorDictSequential,
    SafeSequential,
    WorldModelWrapper,
)
from torchrl.data import (
    CompositeSpec,
    UnboundedContinuousTensorSpec,
)
from torchrl.modules.tensordict_module.world_models import WorldModelWrapper
from common.distributions import TruncNormalDist

from .. import _make_env, transform_env, get_activation
from .modules import (
    RSSMPriorV2, 
    RSSMPosteriorV2,
    ObsEncoder,
    DepthDecoder,
    DreamerActorV2,
    RSSMRollout,
    DreamerEnv
)
from .modules.actor import DreamerActorV2
import copy


class DreamerV2:
    
    def __init__(self, config, test_env, device):
        
        self.keys = dict(
            observation_in_key = "pixels" if config.env.from_pixels else "observation",
            observation_out_key = "reco_pixels" if config.env.from_pixels else "reco_observation",
            action_key = "action",
            value_key = "state_value",
        )
        self.slow_critic_update = 50   # TODO: make this configurable
        
        self.networks = self._init_networks(config, test_env)
        self.modules = self._init_modules(config, test_env, device)
        
    def update(self, step):
        with torch.no_grad():
            if 10*step % self.slow_critic_update == 0:
                print("Updating value target")
                self.modules["value_target"].load_state_dict(self.modules["value_model"].state_dict())
        
    def _init_networks(self, config, proof_env):
        activation = get_activation(config.networks.activation)
        action_spec = proof_env.action_spec
        
        hidden_dim = config.networks.hidden_dim
        state_vars = config.networks.state_vars
        state_classes = config.networks.state_dim
        rssm_dim = config.networks.rssm_hidden_dim
        
        return nn.ModuleDict(modules = dict(
                depth_encoder = ObsEncoder(),
                depth_decoder = DepthDecoder(),
                velocity_encoder = MLP(out_features=16, depth=1, num_cells=64, activation_class=activation),
                veolcity_decoder = MLP(out_features=6, depth=1, num_cells=64, activation_class=activation),
                command_encoder = MLP(out_features=16, depth=1, num_cells=64, activation_class=activation),
                command_decoder = MLP(out_features=4, depth=1, num_cells=64, activation_class=activation),
                gravity_encoder = MLP(out_features=16, depth=1, num_cells=64, activation_class=activation),
                gravity_decoder = MLP(out_features=3, depth=1, num_cells=64, activation_class=activation),

                joint_encoder = MLP(out_features=1024, depth=3, num_cells=1280, activation_class=activation),
                
                rssm_prior = RSSMPriorV2(hidden_dim=rssm_dim, rnn_hidden_dim=rssm_dim, state_vars=state_vars, state_classes=state_classes, action_spec=action_spec),
                rssm_posterior = RSSMPosteriorV2(hidden_dim=rssm_dim, state_vars=state_vars, state_classes=state_classes),
                reward_model = MLP(out_features=1, depth=2, num_cells=hidden_dim, activation_class=activation),
                actor_model = DreamerActorV2(out_features=action_spec.shape[-1], depth=2, num_cells=hidden_dim, activation_class=activation, std_min_val=0.1),
                value_model = MLP(out_features=1, depth=2, num_cells=hidden_dim, activation_class=activation),
            )
        )
    

    def _init_modules(self, config, proof_env, device):
        
        world_model = self._dreamer_make_world_model().to(device)
        actor_simulator = self._dreamer_make_actor_sim(proof_environment=proof_env).to(device)
        actor_realworld = self._dreamer_make_actor_real(proof_environment=proof_env).to(device)
        value_model, value_target  = self._dreamer_make_value_model()
        value_model, value_target = value_model.to(device), value_target.to(device)
        model_based_env = self._dreamer_make_mbenv(
            proof_env, state_dim=config.networks.state_dim, rssm_hidden_dim=config.networks.rssm_hidden_dim, device=device
        )
        
        
        # Initialize world model
        with torch.no_grad(), set_exploration_type(ExplorationType.RANDOM):
            tensordict = (proof_env.rollout(5, auto_cast_to_device=True).unsqueeze(-1).to(world_model.device))
            tensordict = tensordict.to_tensordict()
            if len(tensordict.shape) == 3:
                #for isaac_lab envs where paralellization is handled by the env
                tensordict = tensordict[0]
            world_model(tensordict)
            
        # Initialize model-based environment, actor_simulator, value_model
        check_env_specs(model_based_env)
        with torch.no_grad(), set_exploration_type(ExplorationType.RANDOM):
            tensordict = (
                model_based_env.fake_tensordict().unsqueeze(-1).to(value_model.device)
            )
            tensordict = tensordict
            tensordict = actor_simulator(tensordict)
            value_model(tensordict)
            value_target(tensordict)
        
        return nn.ModuleDict(modules=dict(
            world_model = world_model,
            model_based_env = model_based_env,
            actor_simulator = actor_simulator,
            value_model = value_model,
            value_target = value_target,
            actor_realworld = actor_realworld
        ))
        


    def _dreamer_make_value_model(self):

        value_model = SafeModule(
            self.networks["value_model"],
            in_keys=["state", "belief"],
            out_keys=[self.keys["value_key"]]
        )

        value_target = copy.deepcopy(value_model)

        return value_model, value_target


    def _dreamer_make_actor_sim(self, proof_environment):
        
        actor_simulator = SafeProbabilisticTensorDictSequential(
            SafeModule(
                self.networks["actor_model"],
                in_keys=["state", "belief"],
                out_keys=["loc", "scale"],
                spec=CompositeSpec(
                    **{
                        "loc": UnboundedContinuousTensorSpec(
                            proof_environment.action_spec.shape,
                            device=proof_environment.action_spec.device,
                        ),
                        "scale": UnboundedContinuousTensorSpec(
                            proof_environment.action_spec.shape,
                            device=proof_environment.action_spec.device,
                        ),
                    }
                ),
            ),
            SafeProbabilisticModule(
                in_keys=["loc", "scale"],
                out_keys=[self.keys["action_key"]],
                default_interaction_type=InteractionType.RANDOM,
                distribution_class=TruncNormalDist,
                distribution_kwargs={"a": -1, "b": 1},
                spec=CompositeSpec(**{self.keys["action_key"]: proof_environment.action_spec}),
            )
        )
        return actor_simulator


    def _dreamer_make_actor_real(self, proof_environment):
        # actor for real world: interacts with states ~ posterior
        # Out actor differs from the original paper where first they compute prior and posterior and then act on it
        # but we found that this approach worked better.
        actor_realworld = SafeSequential(
            SafeModule(
                self.networks["depth_encoder"],
                in_keys=["depth"],
                out_keys=["encoded_depth"],
            ),
            SafeModule(
                self.networks["velocity_encoder"],
                in_keys=["velocity"],
                out_keys=["encoded_velocity"],
            ),
            SafeModule(
                self.networks["command_encoder"],
                in_keys=["command"],
                out_keys=["encoded_command"],
            ),
            SafeModule(
                self.networks["gravity_encoder"],
                in_keys=["gravity"],
                out_keys=["encoded_gravity"],
            ),

            # SafeModule(
            #     self.networks["joints_encoder"],
            #     in_keys=["joints"],
            #     out_keys=["encoded_joints"],
            # ),
            # SafeModule(
            #     self.networks["actions_encoder"],
            #     in_keys=["actions"],
            #     out_keys=["encoded_actions"],
            # ),
            # SafeModule(
            #     self.networks["height_encoder"],
            #     in_keys=["height"],
            #     out_keys=["encoded_height"],
            # ),
            
            SafeModule(
                self.networks["joint_encoder"],
                in_keys=["encoded_depth", "encoded_velocity", "encoded_command", "encoded_gravity"], # , "encoded_joints", "encoded_actions", "encoded_height"
                out_keys=["encoded_latents"],
            ),  
            
            SafeModule(
                self.networks["rssm_posterior"],
                in_keys=["belief", "encoded_latents"],
                out_keys=[ "_", "state",],
            ),
            SafeProbabilisticTensorDictSequential(
                SafeModule(
                    self.networks["actor_model"],
                    in_keys=["state", "belief"],
                    out_keys=["loc", "scale"],
                    spec=CompositeSpec(
                        **{
                            "loc": UnboundedContinuousTensorSpec(
                                proof_environment.action_spec.shape,
                            ),
                            "scale": UnboundedContinuousTensorSpec(
                                proof_environment.action_spec.shape,
                            ),
                        }
                    ),
                ),
                SafeProbabilisticModule(
                    in_keys=["loc", "scale"],
                    out_keys=[self.keys["action_key"]],
                    default_interaction_type=InteractionType.RANDOM,
                    distribution_class=TruncNormalDist,
                    distribution_kwargs={"a": -1, "b": 1},
                    spec=CompositeSpec(
                        **{self.keys["action_key"]: proof_environment.action_spec.to("cpu")}
                    ),
                ),
            ),
            SafeModule(
                self.networks["rssm_prior"],
                in_keys=["state", "belief", self.keys["action_key"]],
                out_keys=["_", "_", ("next", "belief")], # we don't need the prior state
            ),
        )
        
        return actor_realworld
    


    
    def _dreamer_make_mbenv(self, test_env, use_decoder_in_env: bool = True, state_dim: int = 30, rssm_hidden_dim: int = 200, device: str = "cuda"):
        
        # MB environment
        if use_decoder_in_env:
            mb_env_obs_decoder = SafeModule(
                self.networks["depth_decoder"],
                in_keys=["state", "belief"],
                out_keys=["reco_depth"],
            )
        else:
            mb_env_obs_decoder = None

        transition_model = SafeSequential(
            SafeModule(
                self.networks["rssm_prior"],
                in_keys=["state", "belief", "action"],
                out_keys=["_", "state", "belief"],
            ),
        )

        reward_model = SafeModule(
                self.networks["reward_model"],
                in_keys=["state", "belief"],
                out_keys=["reward"],
        )

        model_based_env = DreamerEnv(
            world_model=WorldModelWrapper(
                transition_model,
                reward_model,
            ),
            prior_shape=torch.Size([state_dim]),
            belief_shape=torch.Size([rssm_hidden_dim]),
            obs_decoder=mb_env_obs_decoder,
            device=device
        )

        model_based_env.set_specs_from_env(test_env)
        
        return model_based_env


    def _dreamer_make_world_model(self):
        
        rssm_rollout = RSSMRollout(
            SafeModule(
                self.networks["rssm_prior"],
                in_keys=["state", "belief", "action"],
                out_keys=[("next", "prior_logits"), "_", ("next", "belief")],
            ),
            SafeModule(
                self.networks["rssm_posterior"],
                in_keys=[("next", "belief"), ("next", "encoded_latents")],
                out_keys=[("next", "posterior_logits"), ("next", "state")],
            ),
        )
    
    
        encoder = SafeSequential(                       
            SafeModule(
                self.networks["depth_encoder"],
                in_keys=[("next", "depth")],
                out_keys=[("next", "encoded_depth")],
            ),
            SafeModule(
                self.networks["velocity_encoder"],
                in_keys=[("next", "velocity")],
                out_keys=[("next", "encoded_velocity")],
            ),
            SafeModule(
                self.networks["command_encoder"],
                in_keys=[("next", "command")],
                out_keys=[("next", "encoded_command")],
            ),
            SafeModule(
                self.networks["gravity_encoder"],
                in_keys=[("next", "gravity")],
                out_keys=[("next", "encoded_gravity")],
            ),

            # SafeModule(
            #     self.networks["joints_encoder"],
            #     in_keys=[("next", "joints")],
            #     out_keys=[("next", "encoded_joints")],
            # ),
            # SafeModule(
            #     self.networks["actions_encoder"],
            #     in_keys=[("next", "actions")],
            #     out_keys=[("next", "encoded_actions")],
            # ),
            # SafeModule(
            #     self.networks["height_encoder"],
            #     in_keys=[("next", "height")],
            #     out_keys=[("next", "encoded_height")],
            # ),

            SafeModule(
                self.networks["joint_encoder"],
                in_keys=[("next", "encoded_depth"), ("next", "encoded_velocity"), ("next", "encoded_command"), ("next", "encoded_gravity")],
                # , ("next", "encoded_joints"), ("next", "encoded_actions"), ("next", "encoded_height")
                out_keys=[("next", "encoded_latents")],
            ),
        )
        
        decoder = SafeSequential(
            SafeModule(
                self.networks["depth_decoder"],
                in_keys=[("next", "state"), ("next", "belief")],
                out_keys=[("next", "reco_depth")],
            ),
            SafeModule(
                self.networks["veolcity_decoder"],
                in_keys=[("next", "state"), ("next", "belief")],
                out_keys=[("next", "reco_velocity")],
            ),
            SafeModule(
                self.networks["command_decoder"],
                in_keys=[("next", "state"), ("next", "belief")],
                out_keys=[("next", "reco_command")],
            ),
            SafeModule(
                self.networks["gravity_decoder"],
                in_keys=[("next", "state"), ("next", "belief")],
                out_keys=[("next", "reco_gravity")],
            ),

            # SafeModule(
            #     self.networks["joints_decoder"],
            #     in_keys=[("next", "state"), ("next", "belief")],
            #     out_keys=[("next", "reco_joints")],
            # ),
            # SafeModule(
            #     self.networks["actions_decoder"],
            #     in_keys=[("next", "state"), ("next", "belief")],
            #     out_keys=[("next", "reco_actions")],
            # ),
            # SafeModule(
            #     self.networks["height_decoder"],
            #     in_keys=[("next", "state"), ("next", "belief")],
            #     out_keys=[("next", "reco_height")],
            # ),
        )

        
        world_model = WorldModelWrapper(
            SafeSequential(encoder, rssm_rollout,decoder),
            reward_model = SafeModule(
                self.networks["reward_model"],
                in_keys=[("next", "state"), ("next", "belief")],
                out_keys=[("next", "reward")],
            ),
        )
        
        return world_model

