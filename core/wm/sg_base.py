from abc import ABC, abstractmethod

class BaseSG(ABC):
    """
    Abstract base class for a 3D Scene Graph.
    """
    @abstractmethod
    def convert_nx_graph(self, *args, **kwargs):
        return None
    
    @abstractmethod
    def update_nx_graph(self, *args, **kwargs):
        return None