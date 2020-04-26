import time
from typing import Union
from collections import deque

import torch

Number = Union[int, float, torch.Tensor]


class LogValue(object):

    """
    Utility class to simplify the logging of a value. Keep tracks of the 
    last `window_size` values and can perform aggreagations such the mean
    or the median

    Parameters
    ----------
    name: str
        Value name
    window_size: int
        How many values should we keep track
    
    """
    def __init__(self, name: str, window_size: int):
        self.name = name
        self.window_size = window_size
        self.logs = deque(maxlen=window_size)
    
    def __call__(self, new_value: Number):
        """Append a new value to the tracking list"""
        self.logs.append(new_value)
    
    def __str__(self) -> str:
        return f'{self.name}: {self.mean.item():.4f}'

    def reset(self):
        """Remove the tracking queue"""
        self.logs.clear()

    @property
    def mean(self) -> torch.FloatTensor:
        if self.window_size == 1: 
            return torch.tensor(self.logs[0]).float()

        logs = torch.as_tensor(self.logs).float()
        return torch.mean(logs)
    
    @property
    def median(self) -> torch.FloatTensor:
        if self.window_size == 1: 
            return torch.tensor(self.logs[0]).float()

        logs = torch.as_tensor(self.logs).float()
        return torch.median(logs)


class ValuesLogger(object):
    """
    Utility class to simplify the activity of logging LogValues. This object
    has the capability of updating multiple `LogValues` at the same time and
    log them every ``print_freq`` steps. It also keeps track of the time
    elapsed between updates.

    Parameters
    ----------
    *values: sequence of LogValue
        LogValues to manage
    print_freq: int, default 10
        Print LogValues every print_freq steps
    header: str, default ''
        Text to add befor the values logs
    """
    def __init__(self, *values, print_freq: int = 10, header: str = ''):
        self.header = header
        self.print_freq = print_freq
        self.values = {v.name: v for v in values}
        self.values['time'] = LogValue(name='time_per_step', window_size=10)
        self.steps = 0
        self.inital_time = time.time()

    def __call__(self, **kwargs):
        self.steps += 1

        # Update the values
        for k, v in kwargs.items():
            self.values[k](v)

        elapsed = time.time() - self.inital_time
        self.values['time'](elapsed)
        
        # Log if is time to do so
        if self.steps % self.print_freq == 0:
            logs = ' '.join(str(v) for v in self.values.values())
            time_str = str(self.values['time'])
            header = self.header.format(step=self.steps)
            print(f'{header} {logs}')

        # Update initial time
        self.inital_time = time.time()
    
    def reset(self):
        """Reset all values and set steps to 0"""
        for v in self.values.values():
            v.reset()
        self.steps = 0
        self.inital_time = time.time()