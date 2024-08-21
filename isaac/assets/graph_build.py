

class GraphBuilder:

    graph = ... # type: nx.Graph

    @staticmethod
    def command_function(env: ManagerBasedRLEnv, std: float, command_name: str) -> torch.Tensor:
        """Reward position tracking with tanh kernel."""
        command = env.command_manager.get_command(command_name)
        des_pos_b = command[:, :3]
        distance = torch.norm(des_pos_b, dim=1)
        return 1 - torch.tanh(distance / std)
    
    @staticmethod
    def reset_event_function(env_ids: Sequence[int] | None = None) -> None:
        self._raw_actions[env_ids] = 0.0
