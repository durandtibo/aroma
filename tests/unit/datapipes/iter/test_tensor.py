from __future__ import annotations

from unittest.mock import Mock

import torch
from coola import objects_are_equal
from pytest import raises
from torch.utils.data.datapipes.iter import IterableWrapper

from aroma.datapipes.iter import MapOfTensorConverter

##########################################
#     Tests for MapOfTensorConverter     #
##########################################


def test_map_of_tensor_converter_str() -> None:
    assert str(MapOfTensorConverter(IterableWrapper([]))).startswith(
        "MapOfTensorConverterIterDataPipe("
    )


def test_map_of_tensor_converter_iter() -> None:
    assert objects_are_equal(
        tuple(
            MapOfTensorConverter(
                IterableWrapper(
                    [
                        {"int": 1, "float": 10.5, "bool": True},
                        {"int": [1, 2, 3], "float": [10.5, 5.5], "bool": [True, False, True]},
                        {"int": (1, 2, 3), "float": (10.5, 5.5), "bool": (True, False, True)},
                        {"str1": ["a", 1, "c"], "str2": ["a", "b", "c"]},
                    ]
                )
            )
        ),
        tuple(
            [
                {"int": 1, "float": 10.5, "bool": True},
                {
                    "int": torch.tensor([1, 2, 3], dtype=torch.long),
                    "float": torch.tensor([10.5, 5.5], dtype=torch.float),
                    "bool": torch.tensor([True, False, True], dtype=torch.bool),
                },
                {
                    "int": torch.tensor([1, 2, 3], dtype=torch.long),
                    "float": torch.tensor([10.5, 5.5], dtype=torch.float),
                    "bool": torch.tensor([True, False, True], dtype=torch.bool),
                },
                {"str1": ["a", 1, "c"], "str2": ["a", "b", "c"]},
            ]
        ),
    )


def test_map_of_tensor_converter_len() -> None:
    assert len(MapOfTensorConverter(Mock(__len__=Mock(return_value=5)))) == 5


def test_map_of_tensor_converter_no_len() -> None:
    with raises(TypeError, match="object of type .* has no len()"):
        len(MapOfTensorConverter(Mock()))
