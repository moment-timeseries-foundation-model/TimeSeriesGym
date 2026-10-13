"""Counts the samples a submitted ResNet is trained on, however `train` reads its data.

The grader asks for `n_epochs` passes over `train_data`. Counting reads of the dataset would
miss a submission that copies the tensors out once and loops over the copy, so the count is
taken at the model instead: while the counter is active, every call of an instance of the
submission's `ResNet` class (also a copy, a reloaded model or one built inside `train`) whose
output is back-propagated adds its batch size. Forward passes without back-propagation
(validation, predictions, BatchNorm statistics) add nothing.

Agent code runs in the grader's process, so code written to evade the count (for example by
removing the hook, or by training another class and copying its weights) is not stopped by it;
the count catches code that trains for more passes than asked, whatever loop it uses.
"""

from collections.abc import Callable

import torch


class TrainingSampleCounter:
    """`samples` is the number of back-propagated samples through instances of
    `model_class()` (None until the class exists) while the counter is entered."""

    def __init__(self, model_class: Callable[[], type | None]):
        self._model_class = model_class
        self._handle = None
        self.samples = 0

    def __enter__(self) -> "TrainingSampleCounter":
        self._handle = torch.nn.modules.module.register_module_forward_hook(self._on_forward)
        return self

    def __exit__(self, *exc) -> None:
        self._handle.remove()
        self._handle = None

    def _on_forward(self, module, args, output) -> None:
        cls = self._model_class()
        if not isinstance(cls, type) or not isinstance(module, cls):
            return
        if isinstance(output, torch.Tensor) and output.requires_grad and output.dim() > 0:
            output.register_hook(self._counter(int(output.shape[0])))

    def _counter(self, batch: int):
        def count(grad):
            if self._handle is not None:  # back-propagated while the counter is active
                self.samples += batch

        return count
