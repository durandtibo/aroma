__all__ = ["MapOfTensorConverterIterDataPipe"]

from collections.abc import Hashable, MutableMapping
from typing import Any

import torch
from gravitorch.utils.format import str_indent
from torch import Tensor
from torch.utils.data import IterDataPipe


class MapOfTensorConverterIterDataPipe(IterDataPipe[MutableMapping[Hashable, Tensor]]):
    r"""Implements an ``IterDataPipe`` to convert lists or tuples to
    ``torch.Tensor``s in a mapping.

    Only the lists and tuples of numerical values are converted to
    ``torch.Tensor``.

    Args:
        datapipe (``IterDataPipe``): Specifies an
            ``IterDataPipe`` of mappings.
    """

    def __init__(self, datapipe: IterDataPipe[MutableMapping[Hashable, Any]]) -> None:
        self._datapipe = datapipe

    def __iter__(self) -> IterDataPipe[MutableMapping[Hashable, Tensor]]:
        for data in self._datapipe:
            for key, value in data.items():
                if isinstance(value, (list, tuple)):
                    try:
                        data[key] = torch.as_tensor(value)
                    except (ValueError, TypeError):
                        # Keep the original value because it is not possible to convert it to a
                        # torch.Tensor
                        pass
            yield data

    def __len__(self) -> int:
        return len(self._datapipe)

    def __str__(self) -> str:
        return f"{self.__class__.__qualname__}(\n  datapipe={str_indent(self._datapipe)},\n)"
