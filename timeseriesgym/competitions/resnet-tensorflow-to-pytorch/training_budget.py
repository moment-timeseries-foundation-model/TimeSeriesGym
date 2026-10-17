"""Counts the samples a submission trains on, however `train` reads its data and calls its model.

The grader asks for `n_epochs` passes over `train_data`. Counting reads of the dataset would
miss a submission that copies the tensors out once and loops over the copy, and counting calls
of the `ResNet` itself would miss one that trains through another method (`model.logits(x)`,
a submodule, `torch.func.functional_call`). The count is taken at the layers instead: while the
counter is active, the output of every call of a module with trainable parameters of its own
(a convolution, a batch normalization, a dense layer, ...) gets a gradient hook. Each backward
pass then counts once, with the median batch size of the layer outputs it reached, so a network
of many layers counts its batch once, and every backward pass counts, also several over one
forward pass. Forward passes without back-propagation (validation, predictions, BatchNorm
statistics) add nothing.

Agent code runs in the grader's process, so code written to evade the count is not stopped by
it; the grader also rejects a model that is better than a constant without any counted sample.
"""

import statistics
from collections import defaultdict

import torch


def _has_own_parameters(module: torch.nn.Module) -> bool:
    return any(p.requires_grad for p in torch.nn.Module.parameters(module, recurse=False))


class TrainingSampleCounter:
    """`samples`: the samples back-propagated through layers with parameters while entered."""

    def __init__(self):
        self._handle = None
        self._batches = defaultdict(list)  # backward pass (autograd graph task) -> batch sizes

    @property
    def samples(self) -> int:
        return int(sum(statistics.median(sizes) for sizes in self._batches.values()))

    def __enter__(self) -> "TrainingSampleCounter":
        self._handle = torch.nn.modules.module.register_module_forward_hook(self._on_forward)
        return self

    def __exit__(self, *exc) -> None:
        self._handle.remove()
        self._handle = None

    def _on_forward(self, module, args, output) -> None:
        if not isinstance(output, torch.Tensor) or not output.requires_grad or output.dim() == 0:
            return
        if _has_own_parameters(module):
            output.register_hook(self._counter(int(output.shape[0])))

    def _counter(self, batch: int):
        def count(grad):
            if self._handle is not None:  # back-propagated while the counter is active
                self._batches[torch._C._current_graph_task_id()].append(batch)

        return count
