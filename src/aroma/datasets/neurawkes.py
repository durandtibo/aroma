__all__ = ["Annotation", "load_event_data", "load_event_examples"]

import logging
import pickle
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from gravitorch.datapipes.iter import DictOfListConverter, SourceWrapper
from gravitorch.utils.mapping import convert_to_dict_of_lists
from gravitorch.utils.path import sanitize_path
from redcat import BatchDict
from redcat.tensorseq import from_sequences
from torch.utils.data.datapipes.iter import Mapper

from aroma.datapipes.iter import MapOfTensorConverter

logger = logging.getLogger(__name__)


MISSING_EVENT_TYPE = -1
MISSING_START_TIME = float("nan")


@dataclass
class Annotation:
    EVENT_TYPE_INDEX: str = "event_type_index"
    INTER_TIMES: str = "inter_times"  # Time since the start of the last event
    START_TIME: str = "start_time"  # Time since start


def load_event_data(path: Path, split: str) -> tuple[BatchDict, dict]:
    r"""Loads the neurawkes event sequences data and metadata.

    Please follow the instructions from
    https://github.com/hongyuanmei/neurawkes/tree/master/data
    to download the data.

    Args:
        path (``Path``): Specifies the path to the directory with the
            neurawkes files.
        split (str): Specifies the dataset split name.

    Returns:
        tuple: A tuple with the data and metadata.

    Example usage:

    .. code-block:: python

        >>> from pathlib import Path
        >>> from aroma.datasets.neurawkes import load_event_data
        >>> data, metadata = load_event_data(Path('/path/to/data/neurawkes/data_retweet'), 'train')
        >>> data, metadata = load_event_data(Path('/path/to/data/neurawkes/data_so/fold1'), 'train')
    """
    examples = load_event_examples(path, split)
    logger.info(f"Creating a batch of {len(examples):,} examples...")
    batch = convert_to_dict_of_lists(examples)
    return (
        BatchDict(
            {
                Annotation.EVENT_TYPE_INDEX: from_sequences(
                    batch[Annotation.EVENT_TYPE_INDEX], padding_value=MISSING_EVENT_TYPE
                ),
                Annotation.START_TIME: from_sequences(
                    batch[Annotation.START_TIME], padding_value=MISSING_START_TIME
                ),
            }
        ),
        {},
    )


def load_event_examples(path: Path, split: str) -> tuple[dict, ...]:
    r"""Loads the neurawkes event-based examples for a given dataset
    split.

    Args:
        path (``Path``): Specifies the path to the directory with the
            neurawkes files.
        split (str): Specifies the dataset split name.

    Returns:
        tuple: The examples.

    Example usage:

    .. code-block:: python

        >>> from pathlib import Path
        >>> from aroma.datasets.neurawkes import load_event_examples
        >>> examples = load_event_examples(Path('/path/to/data/neurawkes/data_retweet'), 'train')
        >>> len(examples)
        20000
        >>> examples[0]
        {'event_type_index': tensor([1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0,  # noqa: E501,B950
                 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0,
                 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1]),
         'start_time': tensor([1.5000e+01, 1.6000e+01, 1.9000e+01, 2.3000e+01, 2.3000e+01, 2.5000e+01,
                 2.8000e+01, 3.0000e+01, 3.2000e+01, 3.3000e+01, 3.4000e+01, 3.6000e+01,
                 3.9000e+01, 4.0000e+01, 4.3000e+01, 4.4000e+01, 4.6000e+01, 4.8000e+01,
                 5.2000e+01, 5.4000e+01, 5.5000e+01, 5.6000e+01, 5.7000e+01, 5.8000e+01,
                 5.9000e+01, 6.0000e+01, 6.1000e+01, 7.4000e+01, 7.5000e+01, 7.7000e+01,
                 7.9000e+01, 8.0000e+01, 8.1000e+01, 8.6000e+01, 8.7000e+01, 8.8000e+01,
                 8.9000e+01, 9.0000e+01, 9.2000e+01, 9.4000e+01, 9.5000e+01, 1.1100e+02,
                 1.1300e+02, 1.1700e+02, 1.1900e+02, 1.5700e+02, 1.6200e+02, 1.6500e+02,
                 1.6600e+02, 1.7000e+02, 1.7100e+02, 1.7700e+02, 1.9100e+02, 1.9500e+02,
                 1.9600e+02, 1.9800e+02, 1.9900e+02, 2.0100e+02, 2.0300e+02, 2.0600e+02,
                 2.0800e+02, 2.1100e+02, 2.1500e+02, 2.1700e+02, 2.1900e+02, 2.2000e+02,
                 2.2500e+02, 2.3200e+02, 2.3400e+02, 2.3500e+02, 2.3900e+02, 2.4100e+02,
                 2.4300e+02, 2.4600e+02, 2.4800e+02, 2.5400e+02, 2.5500e+02, 2.5700e+02,
                 2.5900e+02, 2.6300e+02, 3.1800e+02, 4.6000e+02, 5.9100e+02, 6.9500e+02,
                 9.0000e+02, 3.9225e+04, 3.9270e+04, 3.9328e+04])}
    """
    valid_splits = {"train", "dev", "test", "test1"}
    if split not in valid_splits:
        raise RuntimeError(f"Incorrect split '{split}'. The valid splits are {valid_splits}.")
    path = sanitize_path(path)
    path = path.joinpath(f"{split}.pkl")
    logger.info(f"Loading data from {path}...")
    data = load_pickle2(path)
    logger.info("Preparing examples...")
    dp = SourceWrapper(data[split])
    dp = DictOfListConverter(dp)
    dp = MapOfTensorConverter(dp)
    dp = Mapper(dp, fn=prepare_example)
    return tuple(dp)


def prepare_example(example: Mapping) -> dict:
    r"""Prepares a single example.

    Args:
        example (``Mapping``): Specifies an example.
            The mapping should have the keys: ``'type_event'`` and
            ``'time_since_start'``.

    Returns:
        dict: The prepared example. The dictionary has two keys:
            ``'event_type_index'`` and ``'start_time'``.
    """
    return {
        Annotation.EVENT_TYPE_INDEX: example["type_event"],
        Annotation.START_TIME: example["time_since_start"],
    }


def load_pickle2(path: Path) -> Any:
    r"""Loads the data from a pickle file.

    The data is stored as numpy arrays.
    Pickle incompatibility of numpy arrays between Python 2 and 3:
    https://stackoverflow.com/a/41366785

    Args:
        path (``Path``): Specifies the path to the pickle file.

    Returns:
        The data from the pickle file.
    """
    with Path.open(path, "rb") as f:
        return pickle.load(f, encoding="latin1")


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    path = Path("~/Downloads/NeuralHawkesData/data_retweet")
    batch, metadata = load_event_data(path, "train")
    print(batch)
    print(metadata)

    from aroma.preprocessing import add_inter_times_

    add_inter_times_(batch, time_key=Annotation.START_TIME, inter_times_key=Annotation.INTER_TIMES)
    print(batch)
