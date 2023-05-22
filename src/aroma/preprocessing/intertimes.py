__all__ = ["add_inter_times_", "compute_inter_times_tensor", "compute_inter_times"]

from redcat import BatchDict, BatchedTensorSeq
from torch import Tensor


def add_inter_times_(
    batch: BatchDict,
    time_key: str,
    inter_times_key: str,
) -> None:
    r"""Computes and adds the inter-times to the current batch.

    Note: this function modifies the input batch.

    Args:
        batch (``BatchDict``): Specifies the batch with the time
            data.
        time_key (str): Specifies the key associated to the time
            values in the batch.
        inter_times_key (str): Specifies the key used to store the
            inter-times in the batch.

    Example usage:

    .. code-block:: python

        >>> import torch
        >>> from redcat import BatchDict, BatchedTensorSeq
        >>> from aroma.preprocessing import add_inter_times_
        >>> batch = BatchDict(
        ...     {
        ...         "time": BatchedTensorSeq(
        ...             torch.tensor(
        ...                 [
        ...                     [[1.0], [31.0], [151.0], [429.0], [576.0], [706.0]],
        ...                     [[1.0], [48.0], [216.0], [566.0], [0.0], [0.0]],
        ...                 ],
        ...                 dtype=torch.float,
        ...             )
        ...         )
        ...     }
        ... )
        >>> add_inter_times_(batch, time_key="time", inter_times_key="inter_times")
        >>> batch
        BatchDict(
          (time) tensor([[[1.], [31.], [151.], [429.], [576.], [706.]],
                         [[1.], [48.], [216.], [566.], [  0.], [  0.]]], batch_dim=0, seq_dim=1)
          (inter_times) tensor([[[0.], [30.], [120.], [278.], [ 147.], [130.]],
                                [[0.], [47.], [168.], [350.], [-566.], [  0.]]], batch_dim=0, seq_dim=1)
        )
    """
    time = batch[time_key]
    if not isinstance(time, BatchedTensorSeq):
        raise TypeError(
            f"Incorrect batch type for time. Expecting `BatchedTensorSeq` but received {type(time)}"
        )
    batch[inter_times_key] = compute_inter_times(time)


def compute_inter_times(time: BatchedTensorSeq) -> BatchedTensorSeq:
    r"""Computes the inter-times given a batch of time value sequences.

    Args:
        time (``BatchedTensorSeq``): Specifies a batch of time value
            sequences.

    Returns:
        ``BatchedTensorSeq``: A batch with the inter-times.

    Raises:
        TypeError if the input is not a ``BatchedTensorSeq``.
    """
    if isinstance(time, BatchedTensorSeq):
        return BatchedTensorSeq(
            compute_inter_times_tensor(time.align_to_batch_seq().data, batch_first=True)
        )
    raise TypeError(f"Incorrect type: {type(time)}. The supported type is BatchedTensorSeq")


def compute_inter_times_tensor(time: Tensor, batch_first: bool = False) -> Tensor:
    r"""Computes the inter-times given time value sequences.

    Args:
        time (``torch.Tensor`` of shape
            ``(batch_size, sequence_length, *)`` if `
            `batch_first=True`` or ``(sequence_length, *)``
            otherwise): Specifies the batch of time value sequences.
        batch_first (bool, optional): Indicates if the first
            dimension is the batch or the sequence. If ``True``, the
            input sequences should have the shape
            ``(batch_size, sequence_length, *)``,
            otherwise ``(sequence_length, *)``. Default: ``False``

    Returns:
        ``torch.Tensor`` of shape ``(batch_size, sequence_length, *)``
            if ``batch_first=True`` or ``(sequence_length, *)``: The
            computed inter-times.
    """
    if batch_first:
        return time.diff(dim=1, prepend=time[:, :1])
    return time.diff(dim=0, prepend=time[:1])
