from abc import ABC, abstractmethod
from typing import Any, Dict, List


class BaseEnv(ABC):
    """
    Abstract interface for simulation environments (WahEnv, AlfredEnv, etc.)
    """

    @abstractmethod
    def reset(self, task_data):
        """
        Reset environemnt with task information.
        """
        pass
    
    @abstractmethod
    def step(self, nl_action):
        """
        Attempt to execute a natural language action in the environment.
        - If the action is feasible, it is executed in the simulator.
        - If the action fails (due to preconditions or low-level controller failures), a feedback message is returned explaining the reason.
        - In either case, the environment's observation is updated accordingly.
        """
        pass

    @abstractmethod
    def get_visual_obs(self, camera_info):
        """
        Return raw visual observation (e.g., RGB image, depth image, etc.)
        """
        pass

    @abstractmethod
    def get_text_obs(self, nl_action=None):
        """
        Return current observation after executing the action.
        """
        pass

    @abstractmethod
    def get_graph_obs(self, visibility='full'):
        """
        Return the current state of the environment as a graph.
        """
        pass

    @abstractmethod
    def get_skill_set(self) -> List[str]:
        """
        Return list of skills the agent can currently attempt.
        """
        pass