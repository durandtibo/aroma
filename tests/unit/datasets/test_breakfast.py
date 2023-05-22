from collections import Counter
from pathlib import Path
from unittest.mock import Mock

import torch
from coola import objects_are_equal
from gravitorch.datapipes.iter import SourceWrapper
from gravitorch.utils.io import save_text
from pytest import fixture, mark, raises
from redcat import BatchDict, BatchedTensor, BatchedTensorSeq, BatchList
from torch.utils.data import IterDataPipe

from aroma.datasets import load_breakfast_events
from aroma.datasets.breakfast import (
    DATASET_SPLITS,
    MISSING_ACTION_INDEX,
    MISSING_END_TIME,
    MISSING_START_TIME,
    ActionIndexAdderIterDataPipe,
    Annotation,
    DuplicateExampleRemoverIterDataPipe,
    TxtAnnotationReaderIterDataPipe,
    create_action_vocabulary,
    filter_batch_by_dataset_split,
    filter_examples_by_dataset_split,
    load_action_annotation,
    load_event_data,
    load_event_examples,
    parse_action_annotation_lines,
    remove_duplicate_examples,
)
from aroma.utils.vocab import Vocabulary


@fixture
def action_vocab() -> Vocabulary:
    return Vocabulary(
        Counter(["SIL", "take_bowl", "pour_cereals", "pour_milk", "stir_cereals", "spoon_powder"])
    )


def create_text_files(path: Path) -> None:
    save_text(
        "1-30 SIL  \n"
        "31-150 take_bowl  \n"
        "151-428 pour_cereals  \n"
        "429-575 pour_milk  \n"
        "576-705 stir_cereals  \n"
        "706-836 SIL\n",
        path.joinpath("P03_cam01_P03_cereals.txt"),
    )
    save_text(  # duplicate example
        "1-30 SIL  \n"
        "31-150 take_bowl  \n"
        "151-428 pour_cereals  \n"
        "429-575 pour_milk  \n"
        "576-705 stir_cereals  \n"
        "706-836 SIL\n",
        path.joinpath("P03_cam02_P03_cereals.txt"),
    )
    save_text(
        "1-47 SIL  \n" "48-215 pour_milk  \n" "216-565 spoon_powder  \n" "566-747 SIL  \n",
        path.joinpath("milk/P54_webcam02_P54_milk.txt"),
    )


####################################
#     Tests for DATASET_SPLITS     #
####################################


@mark.parametrize("index", (1, 2, 3, 4))
def test_dataset_splits_test(index: int) -> None:
    assert len(DATASET_SPLITS[f"test{index}"]) == 13


@mark.parametrize("index", (1, 2, 3, 4))
def test_dataset_splits_minival(index: int) -> None:
    assert len(DATASET_SPLITS[f"minival{index}"]) == 13


@mark.parametrize("index", (1, 2, 3, 4))
def test_dataset_splits_minitrain(index: int) -> None:
    assert len(DATASET_SPLITS[f"minitrain{index}"]) == 26


@mark.parametrize("index", (1, 2, 3, 4))
def test_dataset_splits_train(index: int) -> None:
    assert len(DATASET_SPLITS[f"train{index}"]) == 39


@mark.parametrize("index", (1, 2, 3, 4))
def test_dataset_splits_separate_train_test(index: int) -> None:
    assert len(set(DATASET_SPLITS[f"train{index}"] + DATASET_SPLITS[f"test{index}"])) == 52


@mark.parametrize("index", (1, 2, 3, 4))
def test_dataset_splits_separate_minitrain_minival_test(index: int) -> None:
    assert (
        len(
            set(
                DATASET_SPLITS[f"minitrain{index}"]
                + DATASET_SPLITS[f"minival{index}"]
                + DATASET_SPLITS[f"test{index}"]
            )
        )
        == 52
    )


def test_dataset_splits_separate_minival() -> None:
    assert (
        len(
            set(
                DATASET_SPLITS["minival1"]
                + DATASET_SPLITS["minival2"]
                + DATASET_SPLITS["minival3"]
                + DATASET_SPLITS["minival4"]
            )
        )
        == 52
    )


def test_dataset_splits_separate_test() -> None:
    assert (
        len(
            set(
                DATASET_SPLITS["test1"]
                + DATASET_SPLITS["test2"]
                + DATASET_SPLITS["test3"]
                + DATASET_SPLITS["test4"]
            )
        )
        == 52
    )


###########################################
#     Tests for load_breakfast_events     #
###########################################


def test_load_breakfast_events(tmp_path: Path) -> None:
    create_text_files(tmp_path)
    batch, metadata = load_breakfast_events(tmp_path)
    assert batch.allclose(
        BatchDict(
            {
                Annotation.ACTION_INDEX: BatchedTensorSeq(
                    torch.tensor(
                        [
                            [0, 2, 5, 1, 3, 0],
                            [0, 1, 4, 0, MISSING_ACTION_INDEX, MISSING_ACTION_INDEX],
                        ],
                        dtype=torch.long,
                    )
                ),
                Annotation.COOKING_ACTIVITY: BatchList[str](["cereals", "milk"]),
                Annotation.END_TIME: BatchedTensorSeq(
                    torch.tensor(
                        [
                            [[30.0], [150.0], [428.0], [575.0], [705.0], [836.0]],
                            [
                                [47.0],
                                [215.0],
                                [565.0],
                                [747.0],
                                [MISSING_END_TIME],
                                [MISSING_END_TIME],
                            ],
                        ],
                        dtype=torch.float,
                    )
                ),
                Annotation.PERSON_ID: BatchList[str](["P03", "P54"]),
                Annotation.START_TIME: BatchedTensorSeq(
                    torch.tensor(
                        [
                            [[1.0], [31.0], [151.0], [429.0], [576.0], [706.0]],
                            [
                                [1.0],
                                [48.0],
                                [216.0],
                                [566.0],
                                [MISSING_START_TIME],
                                [MISSING_START_TIME],
                            ],
                        ],
                        dtype=torch.float,
                    )
                ),
            }
        ),
        equal_nan=True,
    )
    assert metadata[Annotation.ACTION_VOCAB].equal(
        Vocabulary(
            Counter(
                {
                    "SIL": 4,
                    "pour_milk": 2,
                    "take_bowl": 1,
                    "stir_cereals": 1,
                    "spoon_powder": 1,
                    "pour_cereals": 1,
                }
            )
        )
    )


#####################################
#     Tests for load_event_data     #
#####################################


def test_load_event_data_remove_duplicate_examples(tmp_path: Path) -> None:
    create_text_files(tmp_path)
    batch, metadata = load_event_data(tmp_path)
    assert batch.allclose(
        BatchDict(
            {
                Annotation.ACTION_INDEX: BatchedTensorSeq(
                    torch.tensor(
                        [
                            [0, 2, 5, 1, 3, 0],
                            [0, 1, 4, 0, MISSING_ACTION_INDEX, MISSING_ACTION_INDEX],
                        ],
                        dtype=torch.long,
                    )
                ),
                Annotation.COOKING_ACTIVITY: BatchList[str](["cereals", "milk"]),
                Annotation.END_TIME: BatchedTensorSeq(
                    torch.tensor(
                        [
                            [[30.0], [150.0], [428.0], [575.0], [705.0], [836.0]],
                            [
                                [47.0],
                                [215.0],
                                [565.0],
                                [747.0],
                                [MISSING_END_TIME],
                                [MISSING_END_TIME],
                            ],
                        ],
                        dtype=torch.float,
                    )
                ),
                Annotation.PERSON_ID: BatchList[str](["P03", "P54"]),
                Annotation.START_TIME: BatchedTensorSeq(
                    torch.tensor(
                        [
                            [[1.0], [31.0], [151.0], [429.0], [576.0], [706.0]],
                            [
                                [1.0],
                                [48.0],
                                [216.0],
                                [566.0],
                                [MISSING_START_TIME],
                                [MISSING_START_TIME],
                            ],
                        ],
                        dtype=torch.float,
                    )
                ),
            }
        ),
        equal_nan=True,
    )
    assert metadata[Annotation.ACTION_VOCAB].equal(
        Vocabulary(
            Counter(
                {
                    "SIL": 4,
                    "pour_milk": 2,
                    "take_bowl": 1,
                    "stir_cereals": 1,
                    "spoon_powder": 1,
                    "pour_cereals": 1,
                }
            )
        )
    )


def test_load_event_data_keep_duplicate_examples(tmp_path: Path) -> None:
    create_text_files(tmp_path)
    batch, metadata = load_event_data(tmp_path, remove_duplicate=False)
    assert batch.allclose(
        BatchDict(
            {
                Annotation.ACTION_INDEX: BatchedTensorSeq(
                    torch.tensor(
                        [
                            [0, 2, 4, 1, 3, 0],
                            [0, 2, 4, 1, 3, 0],
                            [0, 1, 5, 0, MISSING_ACTION_INDEX, MISSING_ACTION_INDEX],
                        ],
                        dtype=torch.long,
                    )
                ),
                Annotation.COOKING_ACTIVITY: BatchList[str](["cereals", "cereals", "milk"]),
                Annotation.END_TIME: BatchedTensorSeq(
                    torch.tensor(
                        [
                            [[30.0], [150.0], [428.0], [575.0], [705.0], [836.0]],
                            [[30.0], [150.0], [428.0], [575.0], [705.0], [836.0]],
                            [
                                [47.0],
                                [215.0],
                                [565.0],
                                [747.0],
                                [MISSING_END_TIME],
                                [MISSING_END_TIME],
                            ],
                        ],
                        dtype=torch.float,
                    )
                ),
                Annotation.PERSON_ID: BatchList[str](["P03", "P03", "P54"]),
                Annotation.START_TIME: BatchedTensorSeq(
                    torch.tensor(
                        [
                            [[1.0], [31.0], [151.0], [429.0], [576.0], [706.0]],
                            [[1.0], [31.0], [151.0], [429.0], [576.0], [706.0]],
                            [
                                [1.0],
                                [48.0],
                                [216.0],
                                [566.0],
                                [MISSING_START_TIME],
                                [MISSING_START_TIME],
                            ],
                        ],
                        dtype=torch.float,
                    )
                ),
            }
        ),
        equal_nan=True,
    )
    assert metadata[Annotation.ACTION_VOCAB].equal(
        Vocabulary(
            Counter(
                {
                    "SIL": 6,
                    "pour_milk": 3,
                    "take_bowl": 2,
                    "stir_cereals": 2,
                    "pour_cereals": 2,
                    "spoon_powder": 1,
                }
            )
        )
    )


#########################################
#     Tests for load_event_examples     #
#########################################


def test_load_event_examples_remove_duplicate_examples(tmp_path: Path) -> None:
    create_text_files(tmp_path)
    assert objects_are_equal(
        load_event_examples(tmp_path),
        (
            {
                Annotation.ACTION: (
                    "SIL",
                    "take_bowl",
                    "pour_cereals",
                    "pour_milk",
                    "stir_cereals",
                    "SIL",
                ),
                Annotation.COOKING_ACTIVITY: "cereals",
                Annotation.END_TIME: torch.tensor(
                    [[30.0], [150.0], [428.0], [575.0], [705.0], [836.0]], dtype=torch.float
                ),
                Annotation.PERSON_ID: "P03",
                Annotation.START_TIME: torch.tensor(
                    [[1.0], [31.0], [151.0], [429.0], [576.0], [706.0]], dtype=torch.float
                ),
            },
            {
                Annotation.ACTION: ("SIL", "pour_milk", "spoon_powder", "SIL"),
                Annotation.COOKING_ACTIVITY: "milk",
                Annotation.END_TIME: torch.tensor(
                    [[47.0], [215.0], [565.0], [747.0]], dtype=torch.float
                ),
                Annotation.PERSON_ID: "P54",
                Annotation.START_TIME: torch.tensor(
                    [[1.0], [48.0], [216.0], [566.0]], dtype=torch.float
                ),
            },
        ),
    )


def test_load_event_examples_keep_duplicate_examples(tmp_path: Path) -> None:
    create_text_files(tmp_path)
    assert objects_are_equal(
        load_event_examples(tmp_path, remove_duplicate=False),
        (
            {
                Annotation.ACTION: (
                    "SIL",
                    "take_bowl",
                    "pour_cereals",
                    "pour_milk",
                    "stir_cereals",
                    "SIL",
                ),
                Annotation.COOKING_ACTIVITY: "cereals",
                Annotation.END_TIME: torch.tensor(
                    [[30.0], [150.0], [428.0], [575.0], [705.0], [836.0]], dtype=torch.float
                ),
                Annotation.PERSON_ID: "P03",
                Annotation.START_TIME: torch.tensor(
                    [[1.0], [31.0], [151.0], [429.0], [576.0], [706.0]], dtype=torch.float
                ),
            },
            {
                Annotation.ACTION: (
                    "SIL",
                    "take_bowl",
                    "pour_cereals",
                    "pour_milk",
                    "stir_cereals",
                    "SIL",
                ),
                Annotation.COOKING_ACTIVITY: "cereals",
                Annotation.END_TIME: torch.tensor(
                    [[30.0], [150.0], [428.0], [575.0], [705.0], [836.0]], dtype=torch.float
                ),
                Annotation.PERSON_ID: "P03",
                Annotation.START_TIME: torch.tensor(
                    [[1.0], [31.0], [151.0], [429.0], [576.0], [706.0]], dtype=torch.float
                ),
            },
            {
                Annotation.ACTION: ("SIL", "pour_milk", "spoon_powder", "SIL"),
                Annotation.COOKING_ACTIVITY: "milk",
                Annotation.END_TIME: torch.tensor(
                    [[47.0], [215.0], [565.0], [747.0]], dtype=torch.float
                ),
                Annotation.PERSON_ID: "P54",
                Annotation.START_TIME: torch.tensor(
                    [[1.0], [48.0], [216.0], [566.0]], dtype=torch.float
                ),
            },
        ),
    )


###################################################
#     Tests for filter_batch_by_dataset_split     #
###################################################


def test_filter_batch_by_dataset_split_test1() -> None:
    assert filter_batch_by_dataset_split(
        BatchDict(
            {
                Annotation.PERSON_ID: BatchList[str](["P03", "P13", "P23", "P33", "P03"]),
                "key": BatchedTensor(torch.arange(5)),
            }
        ),
        split="test1",
    ).equal(
        BatchDict(
            {
                Annotation.PERSON_ID: BatchList[str](["P03", "P13", "P03"]),
                "key": BatchedTensor(torch.tensor([0, 1, 4])),
            }
        )
    )


def test_filter_batch_by_dataset_split_train1() -> None:
    assert filter_batch_by_dataset_split(
        BatchDict(
            {
                Annotation.PERSON_ID: BatchList[str](["P03", "P13", "P23", "P33", "P03"]),
                "key": BatchedTensor(torch.arange(5)),
            }
        ),
        split="train1",
    ).equal(
        BatchDict(
            {
                Annotation.PERSON_ID: BatchList[str](["P23", "P33"]),
                "key": BatchedTensor(torch.tensor([2, 3])),
            }
        )
    )


@mark.parametrize("person_id_key", ("person", "ID"))
def test_filter_batch_by_dataset_split_custom_person_id_key(person_id_key: str) -> None:
    assert filter_batch_by_dataset_split(
        BatchDict(
            {
                person_id_key: BatchList[str](["P03", "P13", "P23", "P33", "P03"]),
                "key": BatchedTensor(torch.arange(5)),
            }
        ),
        split="test1",
        person_id_key=person_id_key,
    ).equal(
        BatchDict(
            {
                person_id_key: BatchList[str](["P03", "P13", "P03"]),
                "key": BatchedTensor(torch.tensor([0, 1, 4])),
            }
        )
    )


######################################################
#     Tests for filter_examples_by_dataset_split     #
######################################################


def test_filter_examples_by_dataset_split_empty() -> None:
    assert filter_examples_by_dataset_split([], split="test1") == []


def test_filter_examples_by_dataset_split_test1() -> None:
    assert filter_examples_by_dataset_split(
        [
            {Annotation.PERSON_ID: "P03", "key": 0},
            {Annotation.PERSON_ID: "P13", "key": 1},
            {Annotation.PERSON_ID: "P23", "key": 2},
            {Annotation.PERSON_ID: "P33", "key": 3},
            {Annotation.PERSON_ID: "P03", "key": 4},
        ],
        split="test1",
    ) == [
        {Annotation.PERSON_ID: "P03", "key": 0},
        {Annotation.PERSON_ID: "P13", "key": 1},
        {Annotation.PERSON_ID: "P03", "key": 4},
    ]


def test_filter_examples_by_dataset_split_train1() -> None:
    assert filter_examples_by_dataset_split(
        [
            {Annotation.PERSON_ID: "P03", "key": 0},
            {Annotation.PERSON_ID: "P13", "key": 1},
            {Annotation.PERSON_ID: "P23", "key": 2},
            {Annotation.PERSON_ID: "P33", "key": 3},
            {Annotation.PERSON_ID: "P03", "key": 4},
        ],
        split="train1",
    ) == [{Annotation.PERSON_ID: "P23", "key": 2}, {Annotation.PERSON_ID: "P33", "key": 3}]


@mark.parametrize("person_id_key", ("person", "ID"))
def test_filter_examples_by_dataset_split_custom_person_id_key(person_id_key: str) -> None:
    assert filter_examples_by_dataset_split(
        [
            {person_id_key: "P03", "key": 0},
            {person_id_key: "P13", "key": 1},
            {person_id_key: "P23", "key": 2},
            {person_id_key: "P33", "key": 3},
            {person_id_key: "P03", "key": 4},
        ],
        split="test1",
        person_id_key=person_id_key,
    ) == [
        {person_id_key: "P03", "key": 0},
        {person_id_key: "P13", "key": 1},
        {person_id_key: "P03", "key": 4},
    ]


############################################
#     Tests for load_action_annotation     #
############################################


def test_load_action_annotation_incorrect_extension() -> None:
    with raises(ValueError):
        load_action_annotation(Mock(spec=Path))


def test_load_action_annotation(tmp_path: Path) -> None:
    path = tmp_path.joinpath("P03_cam01_P03_cereals.txt")
    save_text(
        "1-30 SIL  \n"
        "31-150 take_bowl  \n"
        "151-428 pour_cereals  \n"
        "429-575 pour_milk  \n"
        "576-705 stir_cereals  \n"
        "706-836 SIL\n",
        path,
    )
    assert objects_are_equal(
        load_action_annotation(path),
        {
            Annotation.ACTION: (
                "SIL",
                "take_bowl",
                "pour_cereals",
                "pour_milk",
                "stir_cereals",
                "SIL",
            ),
            Annotation.COOKING_ACTIVITY: "cereals",
            Annotation.END_TIME: torch.tensor(
                [[30.0], [150.0], [428.0], [575.0], [705.0], [836.0]], dtype=torch.float
            ),
            Annotation.PERSON_ID: "P03",
            Annotation.START_TIME: torch.tensor(
                [[1.0], [31.0], [151.0], [429.0], [576.0], [706.0]], dtype=torch.float
            ),
        },
    )


###################################################
#     Tests for parse_action_annotation_lines     #
###################################################


def test_parse_action_annotation_lines_empty() -> None:
    assert objects_are_equal(
        parse_action_annotation_lines([]),
        {
            Annotation.ACTION: tuple(),
            Annotation.START_TIME: torch.zeros(0, 1, dtype=torch.float),
            Annotation.END_TIME: torch.zeros(0, 1, dtype=torch.float),
        },
    )


def test_parse_annotation_lines() -> None:
    assert objects_are_equal(
        parse_action_annotation_lines(
            [
                "1-30 SIL  \n",
                "31-150 take_bowl  \n",
                "151-428 pour_cereals  \n",
                "429-575 pour_milk",
                "576-705 stir_cereals  \n",
                "706-836 SIL  \n",
            ]
        ),
        {
            Annotation.ACTION: (
                "SIL",
                "take_bowl",
                "pour_cereals",
                "pour_milk",
                "stir_cereals",
                "SIL",
            ),
            Annotation.START_TIME: torch.tensor(
                [[1.0], [31.0], [151.0], [429.0], [576.0], [706.0]], dtype=torch.float
            ),
            Annotation.END_TIME: torch.tensor(
                [[30.0], [150.0], [428.0], [575.0], [705.0], [836.0]], dtype=torch.float
            ),
        },
    )


###############################################
#     Tests for remove_duplicate_examples     #
###############################################


def test_remove_duplicate_examples_empty() -> None:
    assert remove_duplicate_examples([]) == []


def test_remove_duplicate_examples() -> None:
    assert objects_are_equal(
        remove_duplicate_examples(
            [
                {
                    Annotation.ACTION: (
                        "SIL",
                        "take_bowl",
                        "pour_cereals",
                        "pour_milk",
                        "stir_cereals",
                        "SIL",
                    ),
                    Annotation.COOKING_ACTIVITY: "cereals",
                    Annotation.PERSON_ID: "P03",
                },
                {  # Different person
                    Annotation.ACTION: (
                        "SIL",
                        "take_bowl",
                        "pour_cereals",
                        "pour_milk",
                        "stir_cereals",
                        "SIL",
                    ),
                    Annotation.COOKING_ACTIVITY: "cereals",
                    Annotation.PERSON_ID: "P01",
                },
                {  # Different cooking activity
                    Annotation.ACTION: (
                        "SIL",
                        "take_bowl",
                        "pour_cereals",
                        "pour_milk",
                        "stir_cereals",
                        "SIL",
                    ),
                    Annotation.COOKING_ACTIVITY: "milk",
                    Annotation.PERSON_ID: "P03",
                },
                {  # Duplicate
                    Annotation.ACTION: (
                        "SIL",
                        "take_bowl",
                        "pour_cereals",
                        "pour_milk",
                        "stir_cereals",
                        "SIL",
                    ),
                    Annotation.COOKING_ACTIVITY: "cereals",
                    Annotation.PERSON_ID: "P03",
                },
                {  # Different sequence of actions
                    Annotation.ACTION: ("SIL", "take_bowl", "pour_cereals", "SIL"),
                    Annotation.COOKING_ACTIVITY: "cereals",
                    Annotation.PERSON_ID: "P03",
                },
                {  # Extra keys
                    Annotation.ACTION: (
                        "SIL",
                        "take_bowl",
                        "pour_cereals",
                        "pour_milk",
                        "stir_cereals",
                        "SIL",
                    ),
                    Annotation.COOKING_ACTIVITY: "cereals",
                    Annotation.PERSON_ID: "P03",
                    Annotation.START_TIME: torch.tensor(
                        [[1.0], [31.0], [151.0], [429.0], [576.0], [706.0]], dtype=torch.float
                    ),
                },
            ]
        ),
        [
            {
                Annotation.ACTION: (
                    "SIL",
                    "take_bowl",
                    "pour_cereals",
                    "pour_milk",
                    "stir_cereals",
                    "SIL",
                ),
                Annotation.COOKING_ACTIVITY: "cereals",
                Annotation.PERSON_ID: "P01",
            },
            {
                Annotation.ACTION: (
                    "SIL",
                    "take_bowl",
                    "pour_cereals",
                    "pour_milk",
                    "stir_cereals",
                    "SIL",
                ),
                Annotation.COOKING_ACTIVITY: "cereals",
                Annotation.PERSON_ID: "P03",
            },
            {
                Annotation.ACTION: ("SIL", "take_bowl", "pour_cereals", "SIL"),
                Annotation.COOKING_ACTIVITY: "cereals",
                Annotation.PERSON_ID: "P03",
            },
            {
                Annotation.ACTION: (
                    "SIL",
                    "take_bowl",
                    "pour_cereals",
                    "pour_milk",
                    "stir_cereals",
                    "SIL",
                ),
                Annotation.COOKING_ACTIVITY: "cereals",
                Annotation.PERSON_ID: "P03",
                Annotation.START_TIME: torch.tensor(
                    [[1.0], [31.0], [151.0], [429.0], [576.0], [706.0]], dtype=torch.float
                ),
            },
            {
                Annotation.ACTION: (
                    "SIL",
                    "take_bowl",
                    "pour_cereals",
                    "pour_milk",
                    "stir_cereals",
                    "SIL",
                ),
                Annotation.COOKING_ACTIVITY: "milk",
                Annotation.PERSON_ID: "P03",
            },
        ],
    )


##############################################
#     Tests for create_action_vocabulary     #
##############################################


def test_create_action_vocabulary_empty() -> None:
    assert create_action_vocabulary([]).equal(Vocabulary(Counter({})))


def test_create_action_vocabulary() -> None:
    assert create_action_vocabulary(
        [
            {Annotation.ACTION: ("action1", "action2", "action3", "action1")},
            {Annotation.ACTION: ("action4", "action2", "action3", "action1")},
            {Annotation.ACTION: tuple()},
        ]
    ).equal(Vocabulary(Counter({"action1": 3, "action3": 2, "action2": 2, "action4": 1})))


@mark.parametrize("action_key", ("action", "a"))
def test_create_action_vocabulary_action_key(action_key: str) -> None:
    assert create_action_vocabulary(
        [
            {action_key: ("action1", "action2", "action3", "action1")},
            {action_key: ("action4", "action2", "action3", "action1")},
            {action_key: tuple()},
        ],
        action_key=action_key,
    ).equal(Vocabulary(Counter({"action1": 3, "action3": 2, "action2": 2, "action4": 1})))


#########################################################
#     Tests for DuplicateExampleRemoverIterDataPipe     #
#########################################################


def test_duplicate_example_remover_iter_datapipe_str() -> None:
    assert str(DuplicateExampleRemoverIterDataPipe(Mock(spec=IterDataPipe))).startswith(
        "DuplicateExampleRemoverIterDataPipe("
    )


def test_duplicate_example_remover_iter_datapipe_iter() -> None:
    assert objects_are_equal(
        list(
            DuplicateExampleRemoverIterDataPipe(
                SourceWrapper(
                    [
                        {
                            Annotation.ACTION: (
                                "SIL",
                                "take_bowl",
                                "pour_cereals",
                                "pour_milk",
                                "stir_cereals",
                                "SIL",
                            ),
                            Annotation.COOKING_ACTIVITY: "cereals",
                            Annotation.PERSON_ID: "P03",
                        },
                        {  # Different person
                            Annotation.ACTION: (
                                "SIL",
                                "take_bowl",
                                "pour_cereals",
                                "pour_milk",
                                "stir_cereals",
                                "SIL",
                            ),
                            Annotation.COOKING_ACTIVITY: "cereals",
                            Annotation.PERSON_ID: "P01",
                        },
                        {  # Different cooking activity
                            Annotation.ACTION: (
                                "SIL",
                                "take_bowl",
                                "pour_cereals",
                                "pour_milk",
                                "stir_cereals",
                                "SIL",
                            ),
                            Annotation.COOKING_ACTIVITY: "milk",
                            Annotation.PERSON_ID: "P03",
                        },
                        {  # Duplicate
                            Annotation.ACTION: (
                                "SIL",
                                "take_bowl",
                                "pour_cereals",
                                "pour_milk",
                                "stir_cereals",
                                "SIL",
                            ),
                            Annotation.COOKING_ACTIVITY: "cereals",
                            Annotation.PERSON_ID: "P03",
                        },
                        {  # Different sequence of actions
                            Annotation.ACTION: ("SIL", "take_bowl", "pour_cereals", "SIL"),
                            Annotation.COOKING_ACTIVITY: "cereals",
                            Annotation.PERSON_ID: "P03",
                        },
                        {  # Extra keys
                            Annotation.ACTION: (
                                "SIL",
                                "take_bowl",
                                "pour_cereals",
                                "pour_milk",
                                "stir_cereals",
                                "SIL",
                            ),
                            Annotation.COOKING_ACTIVITY: "cereals",
                            Annotation.PERSON_ID: "P03",
                            Annotation.START_TIME: torch.tensor(
                                [[1.0], [31.0], [151.0], [429.0], [576.0], [706.0]],
                                dtype=torch.float,
                            ),
                        },
                    ]
                )
            )
        ),
        [
            {
                Annotation.ACTION: (
                    "SIL",
                    "take_bowl",
                    "pour_cereals",
                    "pour_milk",
                    "stir_cereals",
                    "SIL",
                ),
                Annotation.COOKING_ACTIVITY: "cereals",
                Annotation.PERSON_ID: "P01",
            },
            {
                Annotation.ACTION: (
                    "SIL",
                    "take_bowl",
                    "pour_cereals",
                    "pour_milk",
                    "stir_cereals",
                    "SIL",
                ),
                Annotation.COOKING_ACTIVITY: "cereals",
                Annotation.PERSON_ID: "P03",
            },
            {
                Annotation.ACTION: ("SIL", "take_bowl", "pour_cereals", "SIL"),
                Annotation.COOKING_ACTIVITY: "cereals",
                Annotation.PERSON_ID: "P03",
            },
            {
                Annotation.ACTION: (
                    "SIL",
                    "take_bowl",
                    "pour_cereals",
                    "pour_milk",
                    "stir_cereals",
                    "SIL",
                ),
                Annotation.COOKING_ACTIVITY: "cereals",
                Annotation.PERSON_ID: "P03",
                Annotation.START_TIME: torch.tensor(
                    [[1.0], [31.0], [151.0], [429.0], [576.0], [706.0]], dtype=torch.float
                ),
            },
            {
                Annotation.ACTION: (
                    "SIL",
                    "take_bowl",
                    "pour_cereals",
                    "pour_milk",
                    "stir_cereals",
                    "SIL",
                ),
                Annotation.COOKING_ACTIVITY: "milk",
                Annotation.PERSON_ID: "P03",
            },
        ],
    )


def test_duplicate_example_remover_iter_datapipe_len() -> None:
    assert len(DuplicateExampleRemoverIterDataPipe(Mock(__len__=Mock(return_value=5)))) == 5


def test_duplicate_example_remover_iter_datapipe_no_len() -> None:
    with raises(TypeError):
        len(DuplicateExampleRemoverIterDataPipe(Mock()))


#####################################################
#     Tests for TxtAnnotationReaderIterDataPipe     #
#####################################################


def test_txt_annotation_reader_iter_datapipe_str() -> None:
    assert str(TxtAnnotationReaderIterDataPipe(Mock(spec=IterDataPipe))).startswith(
        "TxtAnnotationReaderIterDataPipe("
    )


def test_txt_annotation_reader_iter_datapipe_iter(tmp_path: Path) -> None:
    create_text_files(tmp_path)
    assert objects_are_equal(
        list(
            TxtAnnotationReaderIterDataPipe(
                SourceWrapper(
                    [
                        tmp_path.joinpath("P03_cam01_P03_cereals.txt"),
                        tmp_path.joinpath("milk/P54_webcam02_P54_milk.txt"),
                    ]
                )
            )
        ),
        [
            {
                Annotation.ACTION: (
                    "SIL",
                    "take_bowl",
                    "pour_cereals",
                    "pour_milk",
                    "stir_cereals",
                    "SIL",
                ),
                Annotation.COOKING_ACTIVITY: "cereals",
                Annotation.END_TIME: torch.tensor(
                    [[30.0], [150.0], [428.0], [575.0], [705.0], [836.0]], dtype=torch.float
                ),
                Annotation.PERSON_ID: "P03",
                Annotation.START_TIME: torch.tensor(
                    [[1.0], [31.0], [151.0], [429.0], [576.0], [706.0]], dtype=torch.float
                ),
            },
            {
                Annotation.ACTION: ("SIL", "pour_milk", "spoon_powder", "SIL"),
                Annotation.COOKING_ACTIVITY: "milk",
                Annotation.END_TIME: torch.tensor(
                    [[47.0], [215.0], [565.0], [747.0]], dtype=torch.float
                ),
                Annotation.PERSON_ID: "P54",
                Annotation.START_TIME: torch.tensor(
                    [[1.0], [48.0], [216.0], [566.0]], dtype=torch.float
                ),
            },
        ],
    )


def test_txt_annotation_reader_iter_datapipe_len() -> None:
    assert len(TxtAnnotationReaderIterDataPipe(Mock(__len__=Mock(return_value=5)))) == 5


def test_txt_annotation_reader_iter_datapipe_no_len() -> None:
    with raises(TypeError):
        len(TxtAnnotationReaderIterDataPipe(Mock()))


##################################################
#     Tests for ActionIndexAdderIterDataPipe     #
##################################################


def test_action_index_adder_iter_datapipe_str(action_vocab: Vocabulary) -> None:
    assert str(ActionIndexAdderIterDataPipe(SourceWrapper([]), vocab=action_vocab)).startswith(
        "ActionIndexAdderIterDataPipe("
    )


def test_action_index_adder_iter_datapipe_iter(action_vocab: Vocabulary) -> None:
    assert objects_are_equal(
        list(
            ActionIndexAdderIterDataPipe(
                SourceWrapper(
                    [
                        {
                            Annotation.ACTION: (
                                "SIL",
                                "take_bowl",
                                "pour_cereals",
                                "pour_milk",
                                "stir_cereals",
                                "SIL",
                            )
                        },
                        {Annotation.ACTION: ("SIL", "pour_milk", "spoon_powder", "SIL")},
                    ]
                ),
                vocab=action_vocab,
            )
        ),
        [
            {
                Annotation.ACTION: (
                    "SIL",
                    "take_bowl",
                    "pour_cereals",
                    "pour_milk",
                    "stir_cereals",
                    "SIL",
                ),
                Annotation.ACTION_INDEX: torch.tensor([0, 1, 2, 3, 4, 0], dtype=torch.long),
            },
            {
                Annotation.ACTION: ("SIL", "pour_milk", "spoon_powder", "SIL"),
                Annotation.ACTION_INDEX: torch.tensor([0, 3, 5, 0], dtype=torch.long),
            },
        ],
    )


@mark.parametrize("action_key", (Annotation.ACTION, f"{Annotation.ACTION_INDEX}0"))
@mark.parametrize("action_index_key", (Annotation.ACTION, f"{Annotation.ACTION_INDEX}0"))
def test_action_index_adder_iter_datapipe_iter_custom_keys(
    action_vocab: Vocabulary,
    action_key: str,
    action_index_key: str,
) -> None:
    assert objects_are_equal(
        list(
            ActionIndexAdderIterDataPipe(
                SourceWrapper(
                    [
                        {
                            action_key: (
                                "SIL",
                                "take_bowl",
                                "pour_cereals",
                                "pour_milk",
                                "stir_cereals",
                                "SIL",
                            )
                        },
                        {action_key: ("SIL", "pour_milk", "spoon_powder", "SIL")},
                    ]
                ),
                vocab=action_vocab,
                action_key=action_key,
                action_index_key=action_index_key,
            )
        ),
        [
            {
                action_key: (
                    "SIL",
                    "take_bowl",
                    "pour_cereals",
                    "pour_milk",
                    "stir_cereals",
                    "SIL",
                ),
                action_index_key: torch.tensor([0, 1, 2, 3, 4, 0], dtype=torch.long),
            },
            {
                action_key: ("SIL", "pour_milk", "spoon_powder", "SIL"),
                action_index_key: torch.tensor([0, 3, 5, 0], dtype=torch.long),
            },
        ],
    )


def test_action_index_adder_iter_datapipe_iter_empty(action_vocab: Vocabulary) -> None:
    assert list(ActionIndexAdderIterDataPipe(SourceWrapper([]), vocab=action_vocab)) == []


def test_action_index_adder_iter_datapipe_len(action_vocab: Vocabulary) -> None:
    assert (
        len(ActionIndexAdderIterDataPipe(Mock(__len__=Mock(return_value=5)), vocab=action_vocab))
        == 5
    )


def test_action_index_adder_iter_datapipe_no_len(action_vocab: Vocabulary) -> None:
    with raises(TypeError):
        len(ActionIndexAdderIterDataPipe(Mock(), vocab=action_vocab))
