from abc import ABC, abstractmethod

class BlockingMethod(ABC):
    @abstractmethod
    def block(self, x, y, k,verbose, controls):
        pass