from abc import ABC, abstractmethod

class BasePlanner(ABC):
    """
    Abstract base class for a task planner that coordinates between an environment and an LLM agent.
    """

    def __init__(self, cfg, env, llm_agent):
        """
        Initialize the planner with a simulation environment and an LLM-based decision-making agent.
        """
        self.cfg = cfg
        self.env = env
        self.llm_agent = llm_agent

    @abstractmethod
    def run(self, task_data):
        """
        Execute the full planning loop on a given task.
        This typically includes reset, perception, reasoning, and action steps.
        """
        pass

    @abstractmethod
    def collect_human(self, task_data):
        """
        Collect human demonstration or annotation data (e.g., for in-context learning or imitation).
        """
        pass

    @abstractmethod
    def collect_llm(self, task_data):
        """
        Run the LLM agent to generate data autonomously (e.g., for training, analysis, or evaluation).
        """
        pass