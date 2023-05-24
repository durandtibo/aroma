r"""This module contains code to prepare/preprocess the Breakfast data.

Information about the Breakfast dataset can be found in the following
paper:

The Language of Actions: Recovering the Syntax and Semantics of Goal-
Directed Human Activities. Kuehne, Arslan, and Serre. CVPR 2014.

Project page:

https://serre-lab.clps.brown.edu/resource/breakfast-actions-dataset/

Data can be downloaded at

https://serre-lab.clps.brown.edu/resource/breakfast-actions-dataset/#Downloads

The documentation assumes the data are downloaded in the directory `/path/to/data/breakfast/`.
"""

__all__ = [
    "ActionIndexAdderIterDataPipe",
    "Annotation",
    "COOKING_ACTIVITIES",
    "DATASET_SPLITS",
    "DuplicateExampleRemoverIterDataPipe",
    "TxtAnnotationReaderIterDataPipe",
    "create_action_vocabulary",
    "download_annotations",
    "filter_batch_by_dataset_split",
    "filter_examples_by_dataset_split",
    "load_action_annotation",
    "load_event_data",
    "load_event_examples",
    "parse_action_annotation_lines",
    "remove_duplicate_examples",
]

import logging
import tarfile
from collections import Counter
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path

import torch
from coola import objects_are_equal
from gravitorch.datapipes.iter import FileFilter, PathLister, SourceWrapper
from gravitorch.utils.format import str_indent
from gravitorch.utils.mapping import convert_to_dict_of_lists
from redcat import BatchDict, BatchList
from redcat.tensorseq import from_sequences
from torch.utils.data import IterDataPipe

from aroma.utils.download import download_drive_file
from aroma.utils.vocab import Vocabulary

logger = logging.getLogger(__name__)

MISSING_ACTION_INDEX = -1
MISSING_START_TIME = float("nan")
MISSING_END_TIME = MISSING_START_TIME

URLS = {
    "segmentation_coarse": "https://drive.google.com/open?id=1R3z_CkO1uIOhu4y2Nh0pCHjQQ2l-Ab9E",
    "segmentation_fine": "https://drive.google.com/open?id=1Alg_xjefEFOOpO_6_RnelWiNqbJlKhVF",
}


@dataclass
class Annotation:
    ACTION: str = "action"
    ACTION_INDEX: str = "action_index"
    ACTION_VOCAB: str = "action_vocab"
    COOKING_ACTIVITY: str = "cooking_activity"
    END_TIME: str = "end_time"
    INTER_TIMES: str = "inter_times"
    PERSON_ID: str = "person_id"
    START_TIME: str = "start_time"


COOKING_ACTIVITIES = (
    "cereals",
    "coffee",
    "friedegg",
    "juice",
    "milk",
    "pancake",
    "salat",
    "sandwich",
    "scrambledegg",
    "tea",
)

NUM_COOKING_ACTIVITIES = {
    "cereals": 214,
    "coffee": 100,
    "friedegg": 198,
    "juice": 187,
    "milk": 224,
    "pancake": 173,
    "salat": 185,
    "sandwich": 197,
    "scrambledegg": 188,
    "tea": 223,
}

PART1 = tuple(f"P{i:02d}" for i in range(3, 16))
PART2 = tuple(f"P{i:02d}" for i in range(16, 29))
PART3 = tuple(f"P{i:02d}" for i in range(29, 42))
PART4 = tuple(f"P{i:02d}" for i in range(42, 55))

DATASET_SPLITS = {
    "minitrain1": sorted(PART2 + PART3),
    "minitrain2": sorted(PART3 + PART4),
    "minitrain3": sorted(PART1 + PART4),
    "minitrain4": sorted(PART1 + PART2),
    "minival1": sorted(PART4),
    "minival2": sorted(PART1),
    "minival3": sorted(PART2),
    "minival4": sorted(PART3),
    "test1": sorted(PART1),
    "test2": sorted(PART2),
    "test3": sorted(PART3),
    "test4": sorted(PART4),
    "train1": sorted(PART2 + PART3 + PART4),
    "train2": sorted(PART1 + PART3 + PART4),
    "train3": sorted(PART1 + PART2 + PART4),
    "train4": sorted(PART1 + PART2 + PART3),
}


def download_annotations(path: Path, force_download: bool = False) -> None:
    r"""Downloads the Breakfast annotations.

    Args:
        path (``pathlib.Path``): Specifies the path where to store the
            downloaded data.
        force_download (bool, optional): If ``True``, the annotations
            are downloaded everytime this function is called.
            If ``False``, the annotations are downloaded only if the
            given path does not contain the annotation data.
            Default: ``False``

    Example usage:

    .. code-block:: python

        >>> from pathlib import Path
        >>> from aroma.datasets.breakfast import download_annotations
        >>> download_annotations(Path('/path/to/data'))
        >>> list(path.iterdir())
        [PosixPath('//path/to/data/segmentation_coarse'),
         PosixPath('/path/to/data/segmentation_fine')]
    """
    logger.info("Downloading Breakfast dataset annotations...")
    for name, url in URLS.items():
        if not path.joinpath(name).is_dir() or force_download:
            tar_file = path.joinpath(f"{name}.tar.gz")
            download_drive_file(url, tar_file, quiet=False, fuzzy=True)
            tarfile.open(tar_file).extractall(path)
            tar_file.unlink(missing_ok=True)


def load_event_data(path: Path, remove_duplicate: bool = True) -> tuple[BatchDict, dict]:
    r"""Loads the event data and metadata for Breakfast.

    Args:
        path (``pathlib.Path``): Specifies the directory where the
            dataset annotations are stored.
        remove_duplicate (bool, optional): If ``True``, the duplicate
            examples are removed.

    Returns:
        tuple: A tuple with the data and metadata.

    Example usage:

    .. code-block:: python

        >>> from pathlib import Path
        >>> from aroma.datasets.breakfast import load_event_data
        # Dataset built with coarse annotations and without duplicate sequence events.
        >>> data, metadata = load_event_data(Path('/path/to/data/breakfast/segmentation_coarse/'))
        >>> data.keys()
        dict_keys(['action_index', 'cooking_activity', 'end_time', 'person_id', 'start_time'])
        >>> data.batch_size
        508
        >>> metadata
        {'action_vocab': Vocabulary(
          counter=Counter({'SIL': 1016, 'pour_milk': 199, 'cut_fruit': 176, ..., 'stir_tea': 2}),
          index_to_token=('SIL', 'pour_milk', 'cut_fruit',..., 'stir_tea'),
          token_to_index={'SIL': 0, 'pour_milk': 1, 'cut_fruit': 2, ..., 'stir_tea': 47},
        )}
        # Dataset built with coarse annotations and with duplicate sequence events.
        >>> data, metadata = load_event_data(
        ...     Path('/path/to/data/breakfast/segmentation_coarse/'),
        ...     remove_duplicate=False,
        ... )
        >>> data.batch_size
        1712
        # Dataset built with coarse annotations and without duplicate sequence events.
        >>> data, metadata = load_event_data(Path('/path/to/data/breakfast/segmentation_fine/'))
        >>> data.batch_size
        257
        >>> metadata
        {'action_vocab': Vocabulary(
          counter=Counter({'garbage': 774, 'move': 649, ..., 'carry_capSalt': 3}),
          index_to_token=('garbage', 'move', 'carry_knife', ..., 'carry_capSalt'),
          token_to_index={'garbage': 0, 'move': 1, 'carry_knife': 2, ..., 'carry_capSalt': 177},
        )}
    """
    examples = load_event_examples(path=path, remove_duplicate=remove_duplicate)
    action_vocab = create_action_vocabulary(examples)
    batch = convert_to_dict_of_lists(
        tuple(ActionIndexAdderIterDataPipe(SourceWrapper(examples), action_vocab))
    )
    return (
        BatchDict(
            {
                Annotation.ACTION_INDEX: from_sequences(
                    batch[Annotation.ACTION_INDEX], padding_value=MISSING_ACTION_INDEX
                ),
                Annotation.COOKING_ACTIVITY: BatchList(batch[Annotation.COOKING_ACTIVITY]),
                Annotation.END_TIME: from_sequences(
                    batch[Annotation.END_TIME], padding_value=MISSING_END_TIME
                ),
                Annotation.PERSON_ID: BatchList(batch[Annotation.PERSON_ID]),
                Annotation.START_TIME: from_sequences(
                    batch[Annotation.START_TIME], padding_value=MISSING_START_TIME
                ),
            }
        ),
        {Annotation.ACTION_VOCAB: action_vocab},
    )


def load_event_examples(path: Path, remove_duplicate: bool = True) -> tuple[dict, ...]:
    r"""Loads all the event-based examples.

    Args:
        path (``pathlib.Path``): Specifies the path where the
            annotations files are.
        remove_duplicate (bool, optional): If ``True``, the duplicate
            examples are removed.

    Returns:
        tuple: The examples. Each example is a dictionary with the
            following keys: ``'action'``, ``'cooking_activity'``,
            ``'end_time'``, ``'person_id'``, ``'start_time'``.

    Example usage:

    .. code-block:: python

        >>> from pathlib import Path
        >>> from aroma.datasets.breakfast import load_event_examples
        >>> examples = load_event_examples(Path("/path/to/data/breakfast/segmentation_coarse"))
        >>> len(examples)
        508
        >>> examples[0]
        {'action': ('SIL',
          'take_bowl',
          'pour_cereals',
          'pour_milk',
          'stir_cereals',
          'SIL'),
         'start_time': tensor([[  1.],
                 [ 31.],
                 [151.],
                 [429.],
                 [576.],
                 [706.]]),
         'end_time': tensor([[ 30.],
                 [150.],
                 [428.],
                 [575.],
                 [705.],
                 [836.]]),
         'person_id': 'P03',
         'cooking_activity': 'cereals'}
        # To keep duplicate examples
        >>> examples = load_event_examples(
        ...     Path("/path/to/data/breakfast/segmentation_coarse"),
        ...     remove_duplicate=False,
        ... )
        >>> len(examples)
        1712
        >>> examples = load_event_examples(Path("/path/to/data/breakfast/segmentation_fine"))
        >>> len(examples)
        257
        >>> examples[0]
        {'action': ('garbage',
          'reach_cabinet',
          'open_cabinet',
          'reach_bowl',
          'carry_bowl',
          'move',
          'close_cabinet',
          'move',
          'turn',
          'reach_cereal',
          'carry_cereal',
          'wait',
          'hold_bowl',
          'pour_cereal',
          'carry_cereal',
          'reach_milk',
          'carry_milk',
          'screwopen_capMilk',
          'pour_milk',
          'screwclose_capMilk',
          'carry_milk',
          'shift',
          'carry_spoon',
          'stir_cereal',
          'wait',
          'reach_cereal',
          'carry_cereal',
          'garbage'),
         'start_time': tensor([[  1.],
                 [ 54.],
                 [ 64.],
                 [ 81.],
                 [127.],
                 [140.],
                 [152.],
                 [178.],
                 [185.],
                 [202.],
                 [228.],
                 [264.],
                 [277.],
                 [285.],
                 [395.],
                 [417.],
                 [424.],
                 [440.],
                 [446.],
                 [538.],
                 [561.],
                 [574.],
                 [617.],
                 [635.],
                 [705.],
                 [785.],
                 [792.],
                 [831.]]),
         'end_time': tensor([[ 53.],
                 [ 63.],
                 [ 80.],
                 [126.],
                 [140.],
                 [151.],
                 [178.],
                 [185.],
                 [201.],
                 [227.],
                 [263.],
                 [276.],
                 [284.],
                 [394.],
                 [416.],
                 [423.],
                 [439.],
                 [445.],
                 [537.],
                 [560.],
                 [573.],
                 [616.],
                 [634.],
                 [704.],
                 [784.],
                 [791.],
                 [830.],
                 [835.]]),
         'person_id': 'P03',
         'cooking_activity': 'cereals'}
    """
    dp = PathLister(SourceWrapper([path]), pattern="**/*.txt")
    dp = FileFilter(dp)
    dp = TxtAnnotationReaderIterDataPipe(dp)
    if remove_duplicate:
        dp = DuplicateExampleRemoverIterDataPipe(dp)
    examples = tuple(dp)
    logger.info(f"Found {len(examples):,} examples (remove_duplicate={remove_duplicate})")
    return examples


def filter_batch_by_dataset_split(
    batch: BatchDict,
    split: str,
    person_id_key: str = Annotation.PERSON_ID,
) -> BatchDict:
    r"""Filters the data in a batch for a given dataset split.

    Args:
        batch (``BatchDict``): Specifies the batch of examples.
        split (str): Specifies the dataset split.
        person_id_key (str, optional): Specifies the key used to store
            the person ID. The dataset splits are organized by
            person ID. Default: ``'person_id'``

    Returns:
        ``BatchDict``: The batch of examples for the given dataset
            split.

    Example usage:

    .. code-block:: python

        >>> from pathlib import Path
        >>> from aroma.datasets.breakfast import (
        ...     filter_batch_by_dataset_split,
        ...     load_event_data,
        ... )
        >>> data, metadata = load_event_data(Path('/path/to/data/breakfast/segmentation_coarse/'))
        >>> filter_batch_by_dataset_split(data, 'train1').batch_size
        386
        >>> filter_batch_by_dataset_split(data, 'test1').batch_size
        122
    """
    indices = []
    valid_person_ids = set(DATASET_SPLITS[split])
    for i, person_id in enumerate(batch[person_id_key].data):
        if person_id in valid_person_ids:
            indices.append(i)
    return batch.index_select_along_batch(indices)


def filter_examples_by_dataset_split(
    examples: Iterable[dict],
    split: str,
    person_id_key: str = Annotation.PERSON_ID,
) -> list[dict]:
    r"""Filters the examples for a given dataset split.

    Args:
        examples (``Iterable``): Specifies an iterable of examples.
        split (str): Specifies the dataset split.
        person_id_key (str, optional): Specifies the key used to store
            the person ID. The dataset splits are organized by person
            ID. Default: ``'person_id'``

    Returns:
        list: The examples for the given dataset split.
    """
    filtered_examples = []
    valid_person_ids = set(DATASET_SPLITS[split])
    for example in examples:
        if example[person_id_key] in valid_person_ids:
            filtered_examples.append(example)
    logger.info(f"Found {len(filtered_examples):,} examples for split `{split}`")
    return filtered_examples


def load_action_annotation(path: Path) -> dict:
    r"""Loads action annotations from a text file.

    Args:
        path (``pathlib.Path``): Specifies the file with the annotations.

    Returns:
        dict: A dictionary with the sequence of actions, the start
            time and end time of each action.
    """
    if path.suffix != ".txt":
        raise ValueError(f"This function can only parse `.txt` files but received {path.suffix}")
    logger.info(f"Reading {path}...")
    with Path.open(path) as file:
        lines = [x.strip() for x in file.readlines()]
    annotations = parse_action_annotation_lines(lines)
    annotations[Annotation.PERSON_ID] = path.stem.split("_", maxsplit=1)[0]
    annotations[Annotation.COOKING_ACTIVITY] = path.stem.rsplit("_", maxsplit=1)[-1]
    return annotations


def parse_action_annotation_lines(lines: Sequence) -> dict:
    r"""Parses action annotation lines and returns a dictionary with the
    data.

    Args:
        lines (sequence): Specifies the lines to parse.

    Returns:
        dict: A dictionary with the sequence of actions, the start
            time and end time of each action.
    """
    actions = []
    start_time = []
    end_time = []
    for line in lines:
        pair_time, action = line.strip().split()
        actions.append(action)
        start, end = pair_time.split("-")
        start_time.append(float(start))
        end_time.append(float(end))
    return {
        Annotation.ACTION: tuple(actions),
        Annotation.START_TIME: torch.tensor(start_time, dtype=torch.float).unsqueeze(dim=1),
        Annotation.END_TIME: torch.tensor(end_time, dtype=torch.float).unsqueeze(dim=1),
    }


def remove_duplicate_examples(examples: Iterable[dict]) -> list[dict]:
    r"""Removes duplicate examples.

    Args:
         examples (``Iterable``): Specifies an iterable of examples.

    Returns:
        list: The list of examples without duplicate.
    """
    examples = sorted(
        examples,
        key=lambda item: (item[Annotation.COOKING_ACTIVITY], item[Annotation.PERSON_ID]),
    )
    num_original_examples = len(examples)
    for i in range(num_original_examples - 1, 0, -1):
        if objects_are_equal(examples[i - 1], examples[i]):
            examples.pop(i)
    logger.info(
        f"{num_original_examples - len(examples):,} duplicate examples have been removed: "
        f"{num_original_examples:,} -> {len(examples)}"
    )
    return examples


def create_action_vocabulary(
    examples: Iterable[dict], action_key: str = Annotation.ACTION
) -> Vocabulary:
    r"""Creates a vocabulary of actions from the examples.

    Args:
        examples (``Iterable``): Specifies the examples.
        action_key (str, optional): Specifies the key associated to
            the actions. Default: ``"action"``

    Returns:
        ``Vocabulary``: The action vocabulary. To have a deterministic
            behavior, the vocabulary is sorted by count.
    """
    logger.info("Creating the vocabulary of actions...")
    counter = Counter()
    for example in examples:
        counter.update(example[action_key])
    vocab = Vocabulary(counter).sort_by_count()
    logger.info(f"Found {len(vocab):,} actions")
    return vocab


class DuplicateExampleRemoverIterDataPipe(IterDataPipe[dict]):
    r"""Implements an ``IterDataPipe`` to remove duplicate examples.

    Args:
        datapipe (``IterDataPipe``): Specifies the source
            ``IterDataPipe``.
    """

    def __init__(self, datapipe: IterDataPipe[dict]) -> None:
        self._datapipe = datapipe

    def __iter__(self) -> Iterator[dict]:
        yield from remove_duplicate_examples(self._datapipe)

    def __len__(self) -> int:
        return len(self._datapipe)

    def __str__(self) -> str:
        return f"{self.__class__.__qualname__}(\n" f"  datapipe={str_indent(self._datapipe)},\n)"


class TxtAnnotationReaderIterDataPipe(IterDataPipe[dict]):
    r"""Implements an ``IterDataPipe`` to read text annotations.

    Args:
        datapipe (``IterDataPipe``): Specifies the source
            ``IterDataPipe``.
    """

    def __init__(self, datapipe: IterDataPipe[Path]) -> None:
        self._datapipe = datapipe

    def __iter__(self) -> Iterator[dict]:
        for path in self._datapipe:
            yield load_action_annotation(path)

    def __len__(self) -> int:
        return len(self._datapipe)

    def __str__(self) -> str:
        return f"{self.__class__.__qualname__}(\n" f"  datapipe={str_indent(self._datapipe)},\n)"


class ActionIndexAdderIterDataPipe(IterDataPipe[dict]):
    r"""Implements an ``IterDataPipe`` that adds the action index feature in
    each example.

    Args:
        datapipe (``IterDataPipe``): Specifies the source
            DataPipe.
        vocab (``Vocabulary``): Specifies the vocabulary of actions.
        action_key (str, optional): Specifies the key with the
            sequence of actions in the source DataPipe.
        action_index_key (str, optional): Specifies the key for the
            generated action index feature.
    """

    def __init__(
        self,
        datapipe: IterDataPipe[dict],
        vocab: Vocabulary,
        action_key: str = Annotation.ACTION,
        action_index_key: str = Annotation.ACTION_INDEX,
    ) -> None:
        self._datapipe = datapipe
        self._vocab = vocab
        self._action_key = str(action_key)
        self._action_index_key = str(action_index_key)

    def __iter__(self) -> Iterator[dict]:
        for data in self._datapipe:
            data[self._action_index_key] = torch.tensor(
                [self._vocab.get_index(action) for action in data[self._action_key]],
                dtype=torch.long,
            )
            yield data

    def __len__(self) -> int:
        return len(self._datapipe)

    def __str__(self) -> str:
        return (
            f"{self.__class__.__qualname__}(\n"
            f"  vocab={self._vocab},\n"
            f"  action_key={self._action_key},\n"
            f"  action_index_key={self._action_index_key},\n"
            f"  datapipe={str_indent(self._datapipe)},\n)"
        )


# if __name__ == "__main__":
#     logging.basicConfig(level=logging.DEBUG)
#
#     annotation_path = Path("/Users/thibaut/Downloads/segmentation_coarse")
#     # annotation_path = Path("/Users/thibaut/Downloads/segmentation_fine")
#     batch, metadata = load_event_data(annotation_path)
#     print(batch)
#     print(metadata)
#
#     from aroma.preprocessing import add_inter_times_
#
#     add_inter_times_(batch, time_key=Annotation.START_TIME, inter_times_key=Annotation.INTER_TIMES)
#     print(batch)
