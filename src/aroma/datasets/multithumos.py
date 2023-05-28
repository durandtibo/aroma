r"""This module contains code to prepare/preprocess the MultiTHUMOS data.

Information about the MultiTHUMOS dataset can be found in the following
paper:

Every Moment Counts: Dense Detailed Labeling of Actions in Complex
Videos. Yeung S., Russakovsky O., Jin N., Andriluka M., Mori G., Fei-Fei
L. IJCV 2017 (

http://arxiv.org/pdf/1507.05738)

Project page: http://ai.stanford.edu/~syyeung/everymoment.html
"""

__all__ = [
    "Annotation",
    "create_action_vocabulary",
    "download_annotations",
    "filter_test_examples",
    "filter_validation_examples",
    "load_event_data",
    "load_event_examples",
    "is_annotation_path_ready",
    "load_all_event_annotations_per_video",
    "load_single_event_annotations_per_video",
    "prepare_event_example",
    "sort_examples_by_video_id",
]

import logging
import zipfile
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory

from gravitorch.utils.io import load_text
from gravitorch.utils.mapping import convert_to_dict_of_lists
from gravitorch.utils.path import sanitize_path
from redcat import BatchDict, BatchList
from redcat.tensorseq import from_sequences
from torch.hub import download_url_to_file
from torch.utils.data.datapipes.iter import IterableWrapper
from tqdm import tqdm

from aroma.datapipes.iter import MapOfTensorConverter
from aroma.utils.vocab import Vocabulary

logger = logging.getLogger(__name__)

ANNOTATION_URL = "http://ai.stanford.edu/~syyeung/resources/multithumos.zip"


@dataclass
class Annotation:
    ACTION: str = "action"
    ACTION_INDEX: str = "action_index"
    ACTION_VOCAB: str = "action_vocab"
    END_TIME: str = "end_time"
    INTER_TIMES: str = "inter_times"
    START_TIME: str = "start_time"
    VIDEO_ID: str = "video_id"


def load_event_data(path: Path, split: str = "all") -> tuple[BatchDict, dict]:
    r"""Loads the event data and metadata for MultiTHUMOS.

    Args:
        path (``pathlib.Path``): Specifies the directory where the
            dataset annotations are stored.

    Returns:
        tuple: A tuple with the data and metadata.

    Example usage:

    .. code-block:: pycon

        >>> from pathlib import Path
        >>> from aroma.datasets.multithumos import load_event_data
        >>> path = Path("/path/to/data/")
        >>> data, metadata = load_event_data(path)  # Load all the data i.e. val+test
        >>> data.summary()
        BatchDict(
          (video_id) BatchList(batch_size=413)
          (action_index) BatchedTensorSeq(dtype=torch.int64, shape=torch.Size([413, 1235]), device=cpu, batch_dim=0, seq_dim=1)
          (start_time) BatchedTensorSeq(dtype=torch.float32, shape=torch.Size([413, 1235]), device=cpu, batch_dim=0, seq_dim=1)
          (end_time) BatchedTensorSeq(dtype=torch.float32, shape=torch.Size([413, 1235]), device=cpu, batch_dim=0, seq_dim=1)
        )
        >>> metadata
        {'action_vocab': Vocabulary(
          counter=Counter({'BaseballPitch': 1, 'BasketballBlock': 1, 'BasketballDribble': 1, ...}),
          index_to_token=('BaseballPitch', 'BasketballBlock', 'BasketballDribble', ...),
          token_to_index={'BaseballPitch': 0, 'BasketballBlock': 1, 'BasketballDribble': 2, ...},
        )}
        >>> data, metadata = load_event_data(path, split="val")
        >>> data.summary()
        BatchDict(
          (video_id) BatchList(batch_size=200)
          (action_index) BatchedTensorSeq(dtype=torch.int64, shape=torch.Size([200, 622]), device=cpu, batch_dim=0, seq_dim=1)
          (start_time) BatchedTensorSeq(dtype=torch.float32, shape=torch.Size([200, 622]), device=cpu, batch_dim=0, seq_dim=1)
          (end_time) BatchedTensorSeq(dtype=torch.float32, shape=torch.Size([200, 622]), device=cpu, batch_dim=0, seq_dim=1)
        )
        >>> data, metadata = load_event_data(path, split="test")
        >>> data.summary()
        BatchDict(
          (video_id) BatchList(batch_size=213)
          (action_index) BatchedTensorSeq(dtype=torch.int64, shape=torch.Size([213, 1235]), device=cpu, batch_dim=0, seq_dim=1)
          (start_time) BatchedTensorSeq(dtype=torch.float32, shape=torch.Size([213, 1235]), device=cpu, batch_dim=0, seq_dim=1)
          (end_time) BatchedTensorSeq(dtype=torch.float32, shape=torch.Size([213, 1235]), device=cpu, batch_dim=0, seq_dim=1)
        )
    """
    if split not in (valid_splits := {"all", "val", "test"}):
        raise RuntimeError(f"Incorrect split: {split}. Valid split names are {valid_splits}")
    path = sanitize_path(path)
    action_vocab = create_action_vocabulary(path)
    examples = load_event_examples(path=path, action_vocab=action_vocab)
    if split == "val":
        examples = filter_validation_examples(examples)
    elif split == "test":
        examples = filter_test_examples(examples)
    batch = convert_to_dict_of_lists(examples)
    metadata = {Annotation.ACTION_VOCAB: action_vocab}
    return (
        BatchDict(
            {
                Annotation.VIDEO_ID: BatchList(batch[Annotation.VIDEO_ID]),
                Annotation.ACTION_INDEX: from_sequences(batch[Annotation.ACTION_INDEX]),
                Annotation.START_TIME: from_sequences(batch[Annotation.START_TIME]),
                Annotation.END_TIME: from_sequences(batch[Annotation.END_TIME]),
            }
        ),
        metadata,
    )


def create_action_vocabulary(path: Path) -> Vocabulary:
    r"""Creates the action vocabulary.

    Args:
        path (``pathlib.Path``): Specifies the directory where the
            dataset annotations are stored. The directory should
            contain the file ``class_list.txt``.

    Returns:
        ``Vocabulary``: A vocabulary of 65 actions. The vocabulary is
            sorted by alphabetical order of tokens (i.e. action name).

    Example usage:

    .. code-block:: pycon

        >>> from pathlib import Path
        >>> from aroma.datasets.multithumos import download_annotations
        >>> create_action_vocabulary(Path("/path/to/data/"))
        Vocabulary(vocab_size=65)
    """
    path = sanitize_path(path)
    file = path.joinpath("class_list.txt")
    logger.info(f"Reading actions from {file}...")
    vocab = Vocabulary(
        Counter([line.split(" ", maxsplit=1)[1] for line in load_text(file).splitlines()])
    )
    if vocab.get_vocab_size() != 65:
        raise RuntimeError(f"Incorrect vocabulary size: ({vocab.get_vocab_size():,} but expect 65")
    return vocab.sort_by_token()  # Sort by token to have a deterministic behavior


def download_annotations(path: Path, force_download: bool = False) -> None:
    r"""Downloads the MultiTHUMOS annotations.

    Internally, this function downloads the annotations in a temporary
    directory, then extracts the files from the download zip files in
    the temporary directory, and finally moves the extracted files to
    the given path.

    Args:
        path (``pathlib.Path``): Specifies the path where to store the
            MultiTHUMOS data.
        force_download (bool, optional): If ``True``, the annotations
            are downloaded everytime this function is calleed.
            If ``False``, the annotations are downloaded only if the
            given path does not contain the annotation data.
            Default: ``False``

    Example usage:

    .. code-block:: pycon

        >>> from pathlib import Path
        >>> from aroma.datasets.multithumos import download_annotations
        >>> path = Path("/path/to/data")
        >>> download_annotations(path)
        >>> list(path.iterdir())
        [PosixPath('/path/to/data/multithumos/README'),
         PosixPath('/path/to/data/multithumos/annotations'),
         PosixPath('/path/to/data/multithumos/class_list.txt')]
    """
    path = sanitize_path(path)
    if not is_annotation_path_ready(path) or force_download:
        with TemporaryDirectory() as tmpdir:
            print(tmpdir)
            tmp_path = Path(tmpdir)
            zip_file = tmp_path.joinpath("multithumos.zip.tmp")
            logger.info(f"Downloading MultiTHUMOS annotations data in {zip_file}...")
            download_url_to_file(
                ANNOTATION_URL, zip_file.as_posix(), progress=True, hash_prefix=None
            )

            logger.info(f"Extracting {zip_file} in {tmp_path}...")
            with zipfile.ZipFile(zip_file, "r") as zip_ref:
                zip_ref.extractall(tmp_path)

            logger.info(f"Moving extracted files to {path}...")
            path.mkdir(parents=True, exist_ok=True)
            tmp_path.joinpath("multithumos/README").rename(path.joinpath("README"))
            tmp_path.joinpath("multithumos/class_list.txt").rename(path.joinpath("class_list.txt"))
            tmp_path.joinpath("multithumos/annotations").rename(path.joinpath("annotations"))

    logger.info(f"MultiTHUMOS annotation data are available in {path}")


def is_annotation_path_ready(path: Path) -> bool:
    r"""Indicates if the given path contains the MultiTHUMOS annotation
    data.

    Args:
        path (``pathlib.Path``): Specifies the path to check.

    Returns:
        bool: ``True`` if the path contains the MultiTHUMOS data,
            otherwise ``False``.

    Example usage:

    .. code-block:: pycon

        >>> from pathlib import Path
        >>> from aroma.datasets.multithumos import is_annotation_path_ready
        >>> is_annotation_path_ready(Path("/path/to/data/"))
        True  # or False
    """
    path = sanitize_path(path)
    if not path.joinpath("README").is_file():
        return False
    if not path.joinpath("class_list.txt").is_file():
        return False
    if not path.joinpath("annotations").is_dir():
        return False
    return len(tuple(path.joinpath("annotations").glob("*.txt"))) == 65


def load_event_examples(path: Path, action_vocab: Vocabulary) -> tuple[dict, ...]:
    r"""Gets all the event-based examples.

    Args:
        path (``pathlib.Path``): Specifies the path to the MultiTHUMOS
            annotations.
        action_vocab (``Vocabulary``): Specifies the vocabulary of
            actions.

    Returns:
        tuple: The event-based examples. Each example is represented by
            a dictionary with the following keys: ``'video_id'``,
            ``'action_index'``, ``'start_time'``, and ``'end_time'``.

    Example usage:

    .. code-block:: pycon

        >>> from pathlib import Path
        >>> from aroma.datasets.multithumos import (
        ...     create_action_vocabulary,
        ...     load_event_examples,
        ... )
        >>> path = Path("/path/to/data/")
        >>> action_vocab = create_action_vocabulary(path)
        >>> examples = load_event_examples(path, action_vocab)
        >>> len(examples)
        413
        >>> examples[0]
        {'action_index': tensor([49, 16, 34, 53, 17, 36, 36, 43, 53, 16, 62, 16, 43, 34, 53, 17, 36, 53,  # noqa: E501,B950
                 16, 17]),
         'start_time': tensor([3.0000e-02, 2.0000e-01, 2.3000e-01, 3.3000e-01, 1.0000e+00, 3.0000e+00,  # noqa: E501,B950
                 6.3300e+00, 9.3300e+00, 1.1330e+01, 1.1400e+01, 1.2300e+01, 1.8600e+01,
                 1.8670e+01, 1.9330e+01, 1.9670e+01, 2.0800e+01, 2.6430e+01, 2.8100e+01,
                 2.8300e+01, 3.0300e+01]),
         'end_time': tensor([ 1.1000,  1.1000,  0.9300,  1.1300,  1.5000,  4.7000,  8.0700, 11.3300,
                 12.7000, 12.2000, 15.5000, 20.8000, 19.2700, 20.3700, 21.1000, 22.3000,
                 28.0000, 29.5700, 29.7000, 31.7000]),
         'video_id': 'video_test_0000004'}
    """
    annotations_per_video = load_all_event_annotations_per_video(
        path=path, action_vocab=action_vocab
    )
    datapipe = IterableWrapper(
        tuple(
            prepare_event_example(video_id, events)
            for video_id, events in annotations_per_video.items()
        )
    )
    datapipe = MapOfTensorConverter(datapipe)
    return tuple(tqdm(datapipe, desc="Preparing action examples"))


def load_all_event_annotations_per_video(
    path: Path, action_vocab: Vocabulary
) -> dict[str, list[dict]]:
    r"""Loads the event annotations for all action categories and
    organizes them by video ID.

    Args:
        path (``pathlib.Path``): Specifies the path to the MultiTHUMOS
            annotations.
        action_vocab (``Vocabulary``): Specifies the vocabulary of
            actions.

    Returns:
        dict: A dictionary with the list of events for each video ID.
            The keys are the video IDs, and the values are the list
            of events associated to each video ID. The events are not
            sorted, so the order can be quite random. Each event is
            represented by a dictionary with the following keys:
            ``'action_index'``, ``'start_time'``, and ``'end_time'``.

    Example usage:

    .. code-block:: pycon

        >>> from pathlib import Path
        >>> from aroma.datasets.multithumos import (
        ...     create_action_vocabulary,
        ...     load_all_event_annotations_per_video,
        ... )
        >>> path = Path("/path/to/data/")
        >>> vocab = create_action_vocabulary(path)
        >>> annotations = load_all_event_annotations_per_video(path, vocab)
        >>> annotations["video_validation_0000902"]
        [{'action_index': 1, 'start_time': 2.97, 'end_time': 3.6},
         {'action_index': 1, 'start_time': 4.54, 'end_time': 5.07},
         {'action_index': 1, 'start_time': 20.22, 'end_time': 20.49},
         ...,
         {'action_index': 62, 'start_time': 541.54, 'end_time': 546.55}]
    """
    path = sanitize_path(path)
    annotations = defaultdict(list)
    for action in action_vocab.get_index_to_token():
        action_annotations = load_single_event_annotations_per_video(
            path=path.joinpath(f"annotations/{action}.txt"),
            action_index=action_vocab.get_index(action),
        )
        logger.info(f"Found {len(action_annotations):,} videos for `{action}`")
        for video_id, events in action_annotations.items():
            annotations[video_id].extend(events)
    logger.info(f"Found {len(annotations):,} videos")
    return dict(sorted(annotations.items()))


def load_single_event_annotations_per_video(path: Path, action_index: int) -> dict[str, list[dict]]:
    r"""Loads the event annotations for a single action category and
    organizes them by video ID.

    Args:
        path (``pathlib.Path``): Specifies the path to the
            annotation file for a given action category.
        action_index (int): Specifies the action index. The value is
            used to represent the action category, so it should be
            unique.

    Returns:
        dict: A dictionary with the list of events for each video ID.
            The keys are the video IDs, and the values are the list of
            events associated to each video ID. The events are not
            sorted, so the order can be quite random. Each event is
            represented by a dictionary with the following keys:
            ``'action_index'``, ``'start_time'``, and ``'end_time'``.

    Example usage:

    .. code-block:: pycon

        >>> from pathlib import Path
        >>> from aroma.datasets.multithumos import load_single_event_annotations_per_video
        >>> path = Path("/path/to/data/multithumos/annotations/BasketballBlock.txt")
        >>> annotations = load_single_event_annotations_per_video(path, action_id=1)
        >>> annotations["video_validation_0000902"]
        [{'action_index': 1, 'start_time': 2.97, 'end_time': 3.6},
         {'action_index': 1, 'start_time': 4.54, 'end_time': 5.07},
         {'action_index': 1, 'start_time': 20.22, 'end_time': 20.49},
         {'action_index': 1, 'start_time': 20.72, 'end_time': 21.02},
         {'action_index': 1, 'start_time': 30.5, 'end_time': 30.86}]
    """
    path = sanitize_path(path)
    logger.info(f"Loading annotations from {path}...")
    raw_text = load_text(path)
    annotations = defaultdict(list)
    for line in raw_text.splitlines():
        video_id, start_time, end_time = line.split(" ")
        annotations[video_id].append(
            {
                Annotation.ACTION_INDEX: action_index,
                Annotation.START_TIME: float(start_time),
                Annotation.END_TIME: float(end_time),
            }
        )
    return dict(sorted(annotations.items()))


def prepare_event_example(video_id: str, events: Sequence[Mapping]) -> dict:
    r"""Prepares an example which is represented a sequence of events.

    This function sorts the events by starting time.
    If two events have the same starting time, they are also sorted by
    action index.

    Args:
        video_id (str): Specifies the video ID which is used to
            identify the example.
        events (sequence): Specifies the sequence of events
            associated to the example. The events can be in any
            order. Each event should have at least the following keys:
            ``'start_time'`` and ``'action index'``.

    Returns:
        dict: The prepared example.

    Example usage:

    .. code-block:: pycon

        >>> from aroma.datasets.multithumos import prepare_event_example
        >>> prepare_event_example(
        ...     "video_0",
        ...     [
        ...         {"start_time": 5.1, "action_index": 3, "end_time": 5.7},
        ...         {"start_time": 7.8, "action_index": 3, "end_time": 9.2},
        ...         {"start_time": 1.2, "action_index": 5, "end_time": 3.1},
        ...     ],
        ... )
        {'start_time': [1.2, 5.1, 7.8],
         'action_index': [5, 3, 3],
         'end_time': [3.1, 5.7, 9.2],
         'video_id': 'video_0'}
    """
    events: list = sorted(
        events, key=lambda item: (item[Annotation.START_TIME], item[Annotation.ACTION_INDEX])
    )
    example: dict = convert_to_dict_of_lists(events)
    example[Annotation.VIDEO_ID] = video_id
    return example


def sort_examples_by_video_id(
    examples: Sequence[Mapping], descending: bool = False
) -> list[Mapping]:
    r"""Sorts the examples by video ID.

    Args:
        examples (sequence): Specifies the examples to sort.
        descending (bool, optional): If ``False``, the examples are
            sorted by ascending order. If ``True``, the examples are
            sorted by descending order. Default: ``True``

    Returns:
        list: The examples sorted by video ID.
    """
    return sorted(examples, key=lambda item: item[Annotation.VIDEO_ID], reverse=descending)


def filter_test_examples(examples: Sequence[Mapping]) -> list[Mapping]:
    r"""Filters and keep only the test examples.

    Args:
        examples (sequence): Specifies the examples to filter.
            Each example should have the key ``'video_id'``.

    Returns:
        list: The test examples.
    """
    return [
        example for example in examples if example[Annotation.VIDEO_ID].startswith("video_test")
    ]


def filter_validation_examples(examples: Sequence[Mapping]) -> list[Mapping]:
    r"""Filters and keep only the validation examples.

    Args:
        examples (sequence): Specifies the examples to filter.
            Each example should have the key ``'video_id'``.

    Returns:
        list: The validation examples.
    """
    return [
        example
        for example in examples
        if example[Annotation.VIDEO_ID].startswith("video_validation")
    ]


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    path = Path("~/Downloads/multithumos")
    download_annotations(path)
    vocab = create_action_vocabulary(path)
    print(vocab)
    print(vocab.get_index_to_token())

    examples = load_event_examples(path, action_vocab=vocab)
    print(examples[:10])

    data, metadata = load_event_data(path, split="val")
    print(data.summary())
    print(data)
    print(metadata)
