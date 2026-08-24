from abc import ABC, abstractmethod

class BaseRetriever(ABC):
    """
    Abstract base class for a retriever that coordinates between an working memory and an LLM agent.
    """

    def __init__(self, cfg, llm_agent, sg_interface):
        """
        Initialize the retriever with a working memory and an LLM-based decision-making agent.
        """
        self.cfg = cfg
        self.llm_agent = llm_agent
        self.sg_interface = sg_interface

    @abstractmethod
    def run(self, question):
        pass

    @abstractmethod
    def collect_human(self, task_data):
        pass

    @abstractmethod
    def collect_llm(self, task_data):
        pass