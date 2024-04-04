from abc import ABC, abstractmethod
from typing import Any, Dict
from torch.nn import Module


class ArchitectureConfig(Module, ABC):
    networks: Any
    parts: Any
    losses: Any
    
    @abstractmethod
    def update(self, step):
        pass
