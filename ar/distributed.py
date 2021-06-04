from contextlib import contextmanager
from typing import Generator

import accelerate


@contextmanager
def master_first(accelerator: accelerate.Accelerator) -> Generator:
    """Accelerate utility to let the main process first in a with block

    This has been incorporated in the most recent accelerate release as
    an accelerator method.

    Parameters
    ----------
    accelerator: accelerate.Accelerator
        HuggingFace accelerator object

    Yields
    -------
    Generator
        Waits until the main process has finished
    """
    if not accelerator.is_local_main_process:
        accelerator.wait_for_everyone()

    yield

    if accelerator.is_local_main_process:
        accelerator.wait_for_everyone()
