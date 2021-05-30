from contextlib import contextmanager
from typing import Generator

import accelerate


@contextmanager
def master_first(accelerator: accelerate.Accelerator) -> Generator:
    if not accelerator.is_local_main_process:
        accelerator.wait_for_everyone()

    yield

    if accelerator.is_local_main_process():
        accelerator.wait_for_everyone()
