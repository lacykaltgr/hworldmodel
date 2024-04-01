from abc import ABC
from typing import Any, Dict
from torch.nn import Module


class ArchitectureConfig(Module, ABC):
    networks: Any
    parts: Any
    losses: Any
