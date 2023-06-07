from __future__ import annotations

from unittest.mock import Mock

import torch
from pytest import mark, raises
from redcat import BaseBatch, BatchDict, BatchedTensorSeq

from aroma.preprocessing import add_inter_times_, compute_inter_times
from aroma.preprocessing.intertimes import compute_inter_times_tensor

############################################
#     Tests for add_batch_inter_times_     #
############################################


@mark.parametrize("time_key", ("time", "time0"))
@mark.parametrize("inter_times_key", ("inter", "inter0"))
def test_add_inter_times_(time_key: str, inter_times_key: str) -> None:
    batch = BatchDict(
        {
            time_key: BatchedTensorSeq(
                torch.tensor(
                    [
                        [[1.0], [31.0], [151.0], [429.0], [576.0], [706.0]],
                        [[1.0], [48.0], [216.0], [566.0], [0.0], [0.0]],
                    ],
                    dtype=torch.float,
                )
            )
        }
    )
    add_inter_times_(batch, time_key=time_key, inter_times_key=inter_times_key)
    assert batch.equal(
        BatchDict(
            {
                time_key: BatchedTensorSeq(
                    torch.tensor(
                        [
                            [[1.0], [31.0], [151.0], [429.0], [576.0], [706.0]],
                            [[1.0], [48.0], [216.0], [566.0], [0.0], [0.0]],
                        ],
                        dtype=torch.float,
                    )
                ),
                inter_times_key: BatchedTensorSeq(
                    torch.tensor(
                        [
                            [[0.0], [30.0], [120.0], [278.0], [147.0], [130.0]],
                            [[0.0], [47.0], [168.0], [350.0], [-566.0], [0.0]],
                        ],
                        dtype=torch.float,
                    )
                ),
            }
        )
    )


def test_add_inter_times_incorrect_type() -> None:
    with raises(TypeError, match="Incorrect batch type for time."):
        add_inter_times_(
            BatchDict({"time": Mock(spec=BaseBatch)}),
            time_key="time",
            inter_times_key="inter_times",
        )


#########################################
#     Tests for compute_inter_times     #
#########################################


def test_compute_inter_times_tensor_seq_batch_batch_first() -> None:
    assert compute_inter_times(
        BatchedTensorSeq(torch.tensor([[1, 2, 3, 4, 5], [1, 4, 5, 7, 8]]))
    ).equal(BatchedTensorSeq(torch.tensor([[0, 1, 1, 1, 1], [0, 3, 1, 2, 1]])))


def test_compute_inter_times_tensor_seq_batch_sequence_first() -> None:
    assert compute_inter_times(
        BatchedTensorSeq(
            torch.tensor([[1, 1], [2, 4], [3, 5], [4, 7], [5, 8]]), batch_dim=1, seq_dim=0
        )
    ).equal(BatchedTensorSeq(torch.tensor([[0, 1, 1, 1, 1], [0, 3, 1, 2, 1]])))


def test_compute_inter_times_incorrect_type() -> None:
    with raises(TypeError, match="Incorrect type:"):
        compute_inter_times(Mock(spec=BaseBatch))


################################################
#     Tests for compute_inter_times_tensor     #
################################################


def test_compute_inter_times_tensor_1d() -> None:
    assert compute_inter_times_tensor(torch.tensor([1, 4, 5, 2, 1])).equal(
        torch.tensor([0, 3, 1, -3, -1])
    )


def test_compute_inter_times_tensor_batch_first() -> None:
    assert compute_inter_times_tensor(
        torch.tensor([[1, 2, 3, 4, 5], [1, 4, 5, 7, 8]]), batch_first=True
    ).equal(torch.tensor([[0, 1, 1, 1, 1], [0, 3, 1, 2, 1]]))


def test_compute_inter_times_tensor_sequence_first() -> None:
    assert compute_inter_times_tensor(torch.tensor([[1, 1], [2, 4], [3, 5], [4, 7], [5, 8]])).equal(
        torch.tensor([[0, 0], [1, 3], [1, 1], [1, 2], [1, 1]])
    )
