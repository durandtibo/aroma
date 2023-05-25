from pathlib import Path
from unittest.mock import Mock, patch

import torch
from coola import objects_are_allclose, objects_are_equal
from gravitorch.utils.io import save_pickle
from pytest import mark, raises
from redcat import BatchDict, BatchedTensorSeq

from aroma.datasets.neurawkes import (
    Annotation,
    get_event_data,
    get_event_examples,
    load_pickle2,
    prepare_example,
)

####################################
#     Tests for get_event_data     #
####################################


@mark.parametrize("split", ("train", "test"))
def test_get_event_data(tmp_path: Path, split: str) -> None:
    load_mock = Mock(
        return_value={
            split: [
                [
                    {"time_since_start": 118.0, "time_since_last_event": 118.0, "type_event": 0},
                    {"time_since_start": 177.0, "time_since_last_event": 59.0, "type_event": 1},
                    {"time_since_start": 261.0, "time_since_last_event": 84.0, "type_event": 1},
                ],
                [
                    {"time_since_start": 25.0, "time_since_last_event": 25.0, "type_event": 0},
                    {"time_since_start": 25.0, "time_since_last_event": 0.0, "type_event": 1},
                ],
            ]
        }
    )
    with patch("aroma.datasets.neurawkes.load_pickle2", load_mock):
        assert objects_are_allclose(
            get_event_data(tmp_path, split),
            (
                BatchDict(
                    {
                        Annotation.EVENT_TYPE_INDEX: BatchedTensorSeq(
                            torch.tensor([[0, 1, 1], [0, 1, -1]], dtype=torch.long)
                        ),
                        Annotation.START_TIME: BatchedTensorSeq(
                            torch.tensor(
                                [[118.0, 177.0, 261.0], [25.0, 25.0, float("nan")]],
                                dtype=torch.float,
                            )
                        ),
                    }
                ),
                {},
            ),
            equal_nan=True,
        )
        load_mock.assert_called_once_with(tmp_path.joinpath(f"{split}.pkl"))


########################################
#     Tests for get_event_examples     #
########################################


@mark.parametrize("split", ("train", "test"))
def test_get_event_examples(tmp_path: Path, split: str) -> None:
    load_mock = Mock(
        return_value={
            split: [
                [
                    {"time_since_start": 118.0, "time_since_last_event": 118.0, "type_event": 0},
                    {"time_since_start": 177.0, "time_since_last_event": 59.0, "type_event": 1},
                    {"time_since_start": 261.0, "time_since_last_event": 84.0, "type_event": 1},
                ],
                [
                    {"time_since_start": 25.0, "time_since_last_event": 25.0, "type_event": 0},
                    {"time_since_start": 25.0, "time_since_last_event": 0.0, "type_event": 1},
                ],
            ]
        }
    )
    with patch("aroma.datasets.neurawkes.load_pickle2", load_mock):
        assert objects_are_equal(
            get_event_examples(tmp_path, split),
            tuple(
                [
                    {
                        Annotation.EVENT_TYPE_INDEX: torch.tensor([0, 1, 1], dtype=torch.long),
                        Annotation.START_TIME: torch.tensor(
                            [118.0, 177.0, 261.0], dtype=torch.float
                        ),
                    },
                    {
                        Annotation.EVENT_TYPE_INDEX: torch.tensor([0, 1], dtype=torch.long),
                        Annotation.START_TIME: torch.tensor([25.0, 25.0], dtype=torch.float),
                    },
                ]
            ),
        )
        load_mock.assert_called_once_with(tmp_path.joinpath(f"{split}.pkl"))


def test_get_event_examples_incorrect_split(tmp_path: Path) -> None:
    with raises(RuntimeError, match="Incorrect split 'incorrect'."):
        get_event_examples(tmp_path, "incorrect")


#####################################
#     Tests for prepare_example     #
#####################################


def test_prepare_example() -> None:
    assert prepare_example({"type_event": [1, 2, 3], "time_since_start": [1, 7, 9]}) == {
        Annotation.EVENT_TYPE_INDEX: [1, 2, 3],
        Annotation.START_TIME: [1, 7, 9],
    }


def test_prepare_example_extra_key() -> None:
    assert prepare_example(
        {"type_event": [1, 2, 3], "time_since_start": [1, 7, 9], "key": "abc"}
    ) == {
        Annotation.EVENT_TYPE_INDEX: [1, 2, 3],
        Annotation.START_TIME: [1, 7, 9],
    }


##################################
#     Tests for load_pickle2     #
##################################


def test_load_pickle2(tmp_path: Path) -> None:
    path = tmp_path.joinpath("data.pkl")
    save_pickle([1, 2, 3], path)
    assert load_pickle2(path) == [1, 2, 3]
