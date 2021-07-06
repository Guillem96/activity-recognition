import abc
from pathlib import Path
from typing import Any
from typing import Callable
from typing import Dict
from typing import Optional
from typing import Tuple
from typing import Type
from typing import TypeVar

import requests
import torch
import torch.nn as nn

from ar.typing import PathLike

T = TypeVar('T', bound='SerializableModule')
_SaveFn = Callable[[Dict[str, Any], PathLike], None]


class SerializableModule(nn.Module, abc.ABC):
    """
    Abstract torch nn.Module along with serialize utils.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super(SerializableModule, self).__init__()

    @abc.abstractmethod
    def config(self) -> Dict[str, Any]:
        """Get the model configuration.
        
        This method should return a dictionary which the keys are the __init__
        method arguments are the values are the values to restore the instance
        from.

        Returns
        -------
        Dict[str, Any]
            Dictionary which the keys are equal to the __init__ parameters
            and the values are pickeable.
        """

    def save(self,
             path: PathLike,
             save_fn: Optional[_SaveFn] = None,
             **kwargs: Any) -> None:
        """Serializes the model and the provided kwargs

        Parameters
        ----------
        path: PathLike
            File to serialize the checkpoint
        save_fn: Optional[_SaveFn], defaults None
            Function to use to serialize the checkpoint. This is set when 
            saving with accelerator.save instead of torch.save. I left to None
            by default the method uses the `torch.save`.
        """
        cpu_kwargs = move_to_cpu(kwargs)
        checkpoint = dict(config=self.config(),
                          model=move_to_cpu(self.state_dict()),
                          **cpu_kwargs)
        save_fn = save_fn or torch.save
        save_fn(checkpoint, path)

    @classmethod
    def load(cls: Type[T],
             path: PathLike,
             map_location: Optional[torch.device] = None,
             **kwargs: Any) -> Tuple[T, Dict[str, Any]]:
        """Loads a checkpoint and also returns the remaining elements from it.

        Parameters
        ----------
        path: PathLike
            Path where the checkpoint is stored
        map_location : Optional[torch.device], defaults None
            Device where to load the checkpoint

        Returns
        -------
        Tuple[T, Dict[str, Any]]
            First element is the restored model instance. The second element
            are the remaining objects in the checkpoint.
        """
        checkpoint = torch.load(path, map_location=map_location)
        config = dict(checkpoint.pop('config'), **kwargs)

        instance = cls(**config)
        instance.to(map_location)
        instance.load_state_dict(checkpoint.pop('model'))

        return instance, checkpoint

    @classmethod
    def from_pretrained(cls: Type[T],
                        name_or_path: PathLike,
                        map_location: torch.device = torch.device('cpu'),
                        dst_file: Optional[PathLike] = None) -> T:
        """Loads a model from a name or a path.

        If user uses a name then the method checks if the name is available in
        the model registry and download the checkpoint automatically.

        Parameters
        ----------
        name_or_path: PathLike
            Pretrained model name or path
        map_location : Optional[torch.device], defaults torch.device('cpu')
            Device where to load the checkpoint
        dst_file: Optional[PathLike], defaults None
            Cache file to store the downladed checkpoint. If left to None the 
            method stores the checkpoints at ~.ar/ directory.

        Returns
        -------
        T
            Loaded model in evaluation mode.

        Raises
        ------
        ValueError
            If name and path does not exist
        """
        base_url = 'https://drive.google.com/uc?id='

        names_url = {
            'r2plus1-ucf-101': '1ZPK_wrzV2obnENLOKNQh8tecQCq1EKsx',
            'lrcn-ucf-101': '1hlVU5sqMIyXg57SJfi_36B1AJ2wCTp17',
            'fstcn-ucf-101': '15actfrtMsgzQOpObPGQZR9F3zAcMhsd2'
        }

        path = Path(name_or_path)
        name = str(name_or_path)

        if path.is_file():
            model, _ = cls.load(path, map_location=map_location)
        elif name in names_url:
            if dst_file is None:
                dst_file = Path.home() / '.ar' / (name + '.pt')
                dst_file.parent.mkdir(exist_ok=True)

            if not Path(dst_file).exists():
                import gdown
                gdown.download(f'{base_url}{names_url[name]}', str(dst_file))

            model, _ = cls.load(str(dst_file), map_location=map_location)
        else:
            available_names = ','.join(f'"{o}"' for o in names_url)
            raise ValueError(f'{name_or_path} is not a valid model reference.'
                             f'You can reference a model using a path or an '
                             f'available name. The available model names are: '
                             f'{available_names}')
        model.eval()
        return model


def move_to_cpu(o: Any) -> Any:
    if isinstance(o, dict):
        return {k: move_to_cpu(v) for k, v in o.items()}

    if isinstance(o, (list, tuple)):
        return type(o)(move_to_cpu(v) for v in o)

    if torch.is_tensor(o):
        return o.cpu()

    return o
