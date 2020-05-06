import abc
from typing import Any, Tuple, TypeVar, Type


import torch
import torch.nn as nn

T = TypeVar('T', bound='SerializableModule')


class SerializableModule(nn.Module, abc.ABC):
    """
    Abstract torch nn.Module along with serialize utils.
    """

    def __init__(self, *args, **kwargs):
        super(SerializableModule, self).__init__()
    
    @abc.abstractmethod
    def config(self) -> dict:
        raise NotImplemented

    def save(self, path: str, **kwargs: Any) -> None:
        checkpoint = dict(config=self.config(), 
                          model=self.state_dict(), 
                          **kwargs)
        torch.save(checkpoint, path)

    @classmethod
    def load(cls: Type[T], 
             path: str, 
             map_location: torch.device,
             **kwargs: Any) -> Tuple[T, dict]:

        checkpoint = torch.load(path, map_location=map_location)
        
        instance = cls(**checkpoint.pop('config'), **kwargs)
        instance.to(map_location)
        instance.load_state_dict(checkpoint.pop('model'))

        return instance, checkpoint
