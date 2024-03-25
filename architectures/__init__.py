from abc import ABC
from typing import Any, Dict


class ArchitectureConfig(ABC):
    networks: Any
    modules: Any
    losses: Any
