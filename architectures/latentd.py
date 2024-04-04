
class LatentD(ArchitectureConfig):
    
    def __init__(self, config, device):
        super(DreamerV1, self).__init__()
        test_env = make_env(config, device="cpu")
        test_env = transform_env(config, test_env, parallel_envs=1, dummy=True)
        
        self.keys = dict(
            observation_in_key = "pixels" if config.env.from_pixels else "observation",
            observation_out_key = "reco_pixels" if config.env.from_pixels else "reco_observation",
            action_key = config.architecture.action_key,
            value_key = config.architecture.value_key,
        )
        
        self.networks = self._init_networks(config, test_env)
        self.parts = self._init_modules(config, test_env, device)
        self.losses = self._init_losses(config)
        
        self.policy = self.parts["actor_realworld"]
        
        
    def _init_networks(self, config, proof_env):
        activation = get_activation(config.networks.activation)
        action_spec = proof_env.action_spec
        
        hidden_dim = config.networks.hidden_dim
        state_dim = config.networks.state_dim
        rssm_dim = config.networks.rssm_hidden_dim
        return nn.ModuleDict(modules = dict(
            encoder = (ObsEncoder() if config.env.from_pixels 
                    else MLP(out_features=64, depth=2, num_cells=hidden_dim, activation_class=activation)),
            decoder = (ObsDecoder() if config.env.from_pixels
                    else MLP(out_features=proof_env.observation_spec["observation"].shape[-1], depth=2, num_cells=hidden_dim, activation_class=activation)),
            rssm_prior = RSSMPrior(hidden_dim=hidden_dim,  rnn_hidden_dim=rssm_dim, state_dim=state_dim, action_spec=action_spec),
            rssm_posterior = RSSMPosterior(hidden_dim=rssm_dim, state_dim=state_dim),
            reward_model = MLP(out_features=1, depth=2, num_cells=hidden_dim, activation_class=activation),
            actor_model = DreamerActor(out_features=action_spec.shape[-1], depth=3, num_cells=hidden_dim, activation_class=activation),
            value_model = MLP(out_features=1, depth=3, num_cells=hidden_dim, activation_class=activation)
            )
        )
    

    def _init_modules(self, config, proof_env, device):
        
        world_model = self._dreamer_make_world_model()
        actor_simulator = self._dreamer_make_actor_sim(proof_environment=proof_env)
        actor_realworld = self._dreamer_make_actor_real(proof_environment=proof_env, exploration_noise=config.networks.exploration_noise)
        value_model = self._dreamer_make_value_model()
        model_based_env = self._dreamer_make_mbenv(
            proof_env, use_decoder_in_env=config.architecture.use_decoder_in_env, state_dim=config.networks.state_dim, rssm_hidden_dim=config.networks.rssm_hidden_dim
        )
        
        with torch.no_grad(), set_exploration_type(ExplorationType.RANDOM):
            tensordict = model_based_env.fake_tensordict().unsqueeze(-1).to(value_model.device)
            tensordict = actor_simulator(tensordict)
            value_model(tensordict)
            
        with torch.no_grad(), set_exploration_type(ExplorationType.RANDOM):
            tensordict = (proof_env.rollout(5, auto_cast_to_device=True).unsqueeze(-1).to(world_model.device))
            tensordict = tensordict.to_tensordict()
            world_model(tensordict)
        
        return nn.ModuleDict(modules=dict(
            world_model = world_model.to(device),
            model_based_env = model_based_env.to(device),
            actor_simulator = actor_simulator.to(device),
            value_model = value_model.to(device),
            actor_realworld = actor_realworld.to(device)
        ))
        
    def _init_losses(self, config):
        losses = nn.ModuleDict(dict(
            world_model = DreamerModelLoss(
                self.parts["world_model"]
            ).with_optimizer(params=self.parts["world_model"].parameters(), 
                             lr=config.optimization.world_model_lr, weight_decay=1e-6),
            
            actor = DreamerActorLoss(
                self.parts["actor_simulator"],
                self.parts["value_model"],
                self.parts["model_based_env"],
                imagination_horizon=config.optimization.imagination_horizon,
                discount_loss=False
            ).with_optimizer(params=self.parts["actor_simulator"].parameters(), 
                             lr=config.optimization.actor_lr, weight_decay=1e-6),
            
            value = DreamerValueLoss(
                self.parts["value_model"],
                discount_loss=False
            ).with_optimizer(params=self.parts["value_model"].parameters(), 
                             lr=config.optimization.value_lr, weight_decay=1e-6),
        ))
        #if config.env.backend == "gym" or config.env.backend == "gymnasium":
        #    losses["world_model"].set_keys(pixels="observation", reco_pixels="reco_observation")
        return losses



    def _dreamer_make_value_model(self):
        nets = self.networks
        value_key = self.keys["value_key"]
        
        value_model = SafeProbabilisticTensorDictSequential(
            SafeModule(
                nets["value_model"],
                in_keys=["state", "belief"],
                out_keys=["loc"],
            ),
            SafeProbabilisticModule(
                in_keys=["loc"],
                out_keys=[value_key],
                distribution_class=IndependentNormal,
                distribution_kwargs={"scale": 1.0, "event_dim": 1},
            ),
        )

        return value_model
