Video Transformations (Data augmentation)
=========================================

You can quickly create tranformations for training and evaluation with the following functions.

.. autofunction:: ar.transforms.train_tfms
.. autofunction:: ar.transforms.valid_tfms

Object Oriented Transformations
-------------------------------

The Object Oriented Transformations inherit from `torch.nn.Module` which means
that they are callable objects. Thus, they can be combined using the `torchvision.transforms.Compose`
object or with the `torch.nn.Sequential`.

Note that if you wrap them with the torch sequential API you will be able to run
them on an accelerator other than a CPU.

.. code-block:: python

    import torchvision.transforms as T
    import ar.transforms as VT

    tfms = T.Compose([
        VT.VideoToTensor(),  # Mandatory to apply following transforms
        VT.VideoResize((256, 312)),
        VT.VideoRandomCrop((224, 224)),
        VT.VideoNormalize()
    ])

.. autoclass:: ar.transforms.Identity
    :members:

.. autoclass:: ar.transforms.OneOf
    :members:

.. autoclass:: ar.transforms.VideoCenterCrop
    :members:

.. autoclass:: ar.transforms.VideoNormalize
    :members:

.. autoclass:: ar.transforms.VideoRandomCrop
    :members:

.. autoclass:: ar.transforms.VideoRandomErase
    :members:

.. autoclass:: ar.transforms.VideoRandomHorizontalFlip
    :members:

.. autoclass:: ar.transforms.VideoResize
    :members:

.. autoclass:: ar.transforms.VideoToTensor
    :members:


Functional Transformations
--------------------------

Low level api to compute the transformations. The Object Oriented ones are classes
wrapping this raw functions.

.. automodule:: ar.transforms.functional
    :members:

