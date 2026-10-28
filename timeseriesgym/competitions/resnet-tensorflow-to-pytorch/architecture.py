"""The architecture check: the trainable parameters a ResNet's forward pass uses must number
exactly those of the Keras model of resnet.py built with the same arguments.

The accuracy test alone cannot tell a port from another model: on this data a ResNet without
its shortcuts, or a linear model, also reaches accuracy 1.0. Counting the parameters that get a
gradient from one forward and backward pass, rather than all registered parameters, also catches
layers that are defined but not used. Every honest port measured (a port with Keras' settings,
two Claude Code ports and the reference port) uses exactly the Keras model's 504,258 trainable
parameters with the grading arguments; a ResNet without its shortcuts uses 495,170.
"""

import torch


def _conv(c_in: int, c_out: int, kernel: int) -> int:
    return kernel * c_in * c_out + c_out  # Conv1D with a bias


def _batch_norm(c: int) -> int:
    return 2 * c  # gamma and beta; the moving mean and variance are not trainable


def _block(c_in: int, c_out: int, expand: bool) -> int:
    body = _conv(c_in, c_out, 8) + _conv(c_out, c_out, 5) + _conv(c_out, c_out, 3)
    shortcut = _conv(c_in, c_out, 1) + _batch_norm(c_out) if expand else _batch_norm(c_out)
    return body + 3 * _batch_norm(c_out) + shortcut


def keras_trainable_parameters(n_channels: int, n_feature_maps: int, n_classes: int) -> int:
    """Trainable parameters of Classifier_RESNET.build_model in resnet.py: three blocks (the first
    two with a 1x1 convolution shortcut, the third with a BatchNormalization shortcut), global
    average pooling and a dense softmax layer."""
    f = n_feature_maps
    blocks = _block(n_channels, f, True) + _block(f, 2 * f, True) + _block(2 * f, 2 * f, False)
    return blocks + 2 * f * n_classes + n_classes


def trainable_parameters_in_use(model: torch.nn.Module, x: torch.Tensor) -> int:
    """The number of trainable parameters that get a gradient from one forward and backward pass
    of `model` (in training mode) on `x`. The model is changed only in its gradients and its
    BatchNorm statistics, so the caller should pass a model it does not use for anything else."""
    params = list(torch.nn.Module.parameters(model))
    for p in params:
        p.grad = None
    model.train()
    out = model(x)
    weights = torch.linspace(-1.0, 1.0, out.numel()).reshape(out.shape)  # not constant: softmax
    (out * weights).sum().backward()
    return sum(p.numel() for p in params if p.requires_grad and p.grad is not None)
