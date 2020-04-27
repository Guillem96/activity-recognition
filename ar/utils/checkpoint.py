import abc

import torch
import torch.nn as nn


class SerializableModule(nn.Module, abc.ABC):
    """
    Abstract torch nn.Module along with serialize utils.
    """

    def __init__(self):
        super(SerializableModule, self).__init__()
    
    @abc.abstractmethod
    def config(self) -> dict:
        raise NotImplemented

    def save(self, path: str, **kwargs) -> dict:
        checkpoint = dict(config=self.config(), 
                          model=self.state_dict(), 
                          **kwargs)
        torch.save(checkpoint, path)

    @classmethod
    def load(cls, 
             path: str, 
             map_location: torch.device) -> 'SerializableModule':

        checkpoint = torch.load(path, map_location=map_location)
        
        instance = cls(**checkpoint.pop('config'))
        instance.to(map_location)
        instance.load_state_dict(checkpoint.pop('model'))

        return instance, checkpoint
