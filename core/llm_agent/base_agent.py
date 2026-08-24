from abc import ABC, abstractmethod
from typing import Any, Dict, List

class BaseAgent(ABC):
    def __init__(self, cfg):
        self.cfg = cfg

    @abstractmethod
    def reset(self, *args, **kwargs):
        """
        Reset LLM agent with the initial prompt.
        """
        pass

    @abstractmethod
    def decision_making(self, *args, **kwargs):
        """
        Decide the next step based on current prompt.
        """
        pass