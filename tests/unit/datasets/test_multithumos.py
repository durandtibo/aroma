from collections import Counter
from pathlib import Path
from unittest.mock import Mock, patch
from zipfile import ZipFile

import torch
from coola import objects_are_equal
from gravitorch.utils.io import save_text
from pytest import mark, raises
from redcat import BatchDict, BatchedTensorSeq, BatchList

from aroma.datasets.multithumos import (
    ANNOTATION_URL,
    Annotation,
    create_action_vocabulary,
    download_annotations,
    fetch_event_data,
    filter_test_examples,
    filter_validation_examples,
    is_annotation_path_ready,
    load_all_event_annotations_per_video,
    load_event_data,
    load_event_examples,
    load_single_event_annotations_per_video,
    prepare_event_example,
    sort_examples_by_video_id,
)
from aroma.utils.vocab import Vocabulary

######################################
#     Tests for fetch_event_data     #
######################################


def test_fetch_event_data_default(tmp_path: Path) -> None:
    batch = BatchDict(
        {
            Annotation.VIDEO_ID: BatchList[str](["video_0", "video_1"]),
            Annotation.ACTION_INDEX: BatchedTensorSeq(
                torch.tensor([[5, 3, 3], [1, 2, 0]], dtype=torch.long)
            ),
            Annotation.START_TIME: BatchedTensorSeq(
                torch.tensor([[1.2, 5.1, 7.8], [5.1, 15.1, 0]], dtype=torch.float)
            ),
            Annotation.END_TIME: BatchedTensorSeq(
                torch.tensor([[2.2, 6.1, 8.8], [5.9, 19.5, 0]], dtype=torch.float)
            ),
        }
    )
    load_mock = Mock(return_value=(batch, {}))
    with patch("aroma.datasets.multithumos.download_annotations") as download_mock:
        with patch("aroma.datasets.multithumos.load_event_data", load_mock):
            assert objects_are_equal(fetch_event_data(tmp_path), (batch, {}))
            download_mock.assert_called_once_with(tmp_path, False)
            load_mock.assert_called_once_with(tmp_path, "all")


@mark.parametrize("split", ("all", "val", "test"))
@mark.parametrize("force_download", (True, False))
def test_fetch_event_data(tmp_path: Path, split: str, force_download: bool) -> None:
    batch = BatchDict(
        {
            Annotation.VIDEO_ID: BatchList[str](["video_0", "video_1"]),
            Annotation.ACTION_INDEX: BatchedTensorSeq(
                torch.tensor([[5, 3, 3], [1, 2, 0]], dtype=torch.long)
            ),
            Annotation.START_TIME: BatchedTensorSeq(
                torch.tensor([[1.2, 5.1, 7.8], [5.1, 15.1, 0]], dtype=torch.float)
            ),
            Annotation.END_TIME: BatchedTensorSeq(
                torch.tensor([[2.2, 6.1, 8.8], [5.9, 19.5, 0]], dtype=torch.float)
            ),
        }
    )
    load_mock = Mock(return_value=(batch, {}))
    with patch("aroma.datasets.multithumos.download_annotations") as download_mock:
        with patch("aroma.datasets.multithumos.load_event_data", load_mock):
            assert objects_are_equal(
                fetch_event_data(tmp_path, split=split, force_download=force_download), (batch, {})
            )
            download_mock.assert_called_once_with(tmp_path, force_download)
            load_mock.assert_called_once_with(tmp_path, split)


#####################################
#     Tests for load_event_data     #
#####################################


def test_load_event_data_all(tmp_path: Path) -> None:
    action_vocab = Mock(spec=Vocabulary)
    create_vocab_mock = Mock(return_value=action_vocab)
    event_examples_mock = Mock(
        return_value=(
            {
                Annotation.VIDEO_ID: "video_0",
                Annotation.ACTION_INDEX: torch.tensor([5, 3, 3], dtype=torch.long),
                Annotation.START_TIME: torch.tensor([1.2, 5.1, 7.8], dtype=torch.float),
                Annotation.END_TIME: torch.tensor([2.2, 6.1, 8.8], dtype=torch.float),
            },
            {
                Annotation.VIDEO_ID: "video_1",
                Annotation.ACTION_INDEX: torch.tensor([1, 2], dtype=torch.long),
                Annotation.START_TIME: torch.tensor([5.1, 15.1], dtype=torch.float),
                Annotation.END_TIME: torch.tensor([5.9, 19.5], dtype=torch.float),
            },
        )
    )
    with patch(
        "aroma.datasets.multithumos.create_action_vocabulary",
        create_vocab_mock,
    ):
        with patch(
            "aroma.datasets.multithumos.load_event_examples",
            event_examples_mock,
        ):
            data, metadata = load_event_data(tmp_path)
            create_vocab_mock.assert_called_once_with(tmp_path)
            event_examples_mock.assert_called_once_with(path=tmp_path, action_vocab=action_vocab)
            assert data.equal(
                BatchDict(
                    {
                        Annotation.VIDEO_ID: BatchList[str](["video_0", "video_1"]),
                        Annotation.ACTION_INDEX: BatchedTensorSeq(
                            torch.tensor([[5, 3, 3], [1, 2, 0]], dtype=torch.long)
                        ),
                        Annotation.START_TIME: BatchedTensorSeq(
                            torch.tensor([[1.2, 5.1, 7.8], [5.1, 15.1, 0]], dtype=torch.float)
                        ),
                        Annotation.END_TIME: BatchedTensorSeq(
                            torch.tensor([[2.2, 6.1, 8.8], [5.9, 19.5, 0]], dtype=torch.float)
                        ),
                    }
                )
            )
            assert metadata == {Annotation.ACTION_VOCAB: action_vocab}


def test_load_event_data_val(tmp_path: Path) -> None:
    action_vocab = Mock(spec=Vocabulary)
    create_vocab_mock = Mock(return_value=action_vocab)
    event_examples_mock = Mock(
        return_value=(
            {
                Annotation.VIDEO_ID: "video_test_1",
                Annotation.ACTION_INDEX: torch.tensor([1, 2], dtype=torch.long),
                Annotation.START_TIME: torch.tensor([5.1, 15.1], dtype=torch.float),
                Annotation.END_TIME: torch.tensor([5.9, 19.5], dtype=torch.float),
            },
            {
                Annotation.VIDEO_ID: "video_validation_0",
                Annotation.ACTION_INDEX: torch.tensor([5, 3, 3], dtype=torch.long),
                Annotation.START_TIME: torch.tensor([1.2, 5.1, 7.8], dtype=torch.float),
                Annotation.END_TIME: torch.tensor([2.2, 6.1, 8.8], dtype=torch.float),
            },
            {
                Annotation.VIDEO_ID: "video_validation_1",
                Annotation.ACTION_INDEX: torch.tensor([0, 1, 2, 3], dtype=torch.long),
                Annotation.START_TIME: torch.tensor([5.1, 15.1, 25.1, 35.1], dtype=torch.float),
                Annotation.END_TIME: torch.tensor([5.9, 19.5, 29.5, 39.5], dtype=torch.float),
            },
        )
    )
    with patch(
        "aroma.datasets.multithumos.create_action_vocabulary",
        create_vocab_mock,
    ):
        with patch(
            "aroma.datasets.multithumos.load_event_examples",
            event_examples_mock,
        ):
            data, metadata = load_event_data(tmp_path, split="val")
            create_vocab_mock.assert_called_once_with(tmp_path)
            event_examples_mock.assert_called_once_with(path=tmp_path, action_vocab=action_vocab)
            assert data.equal(
                BatchDict(
                    {
                        Annotation.VIDEO_ID: BatchList[str](
                            ["video_validation_0", "video_validation_1"]
                        ),
                        Annotation.ACTION_INDEX: BatchedTensorSeq(
                            torch.tensor([[5, 3, 3, 0], [0, 1, 2, 3]], dtype=torch.long)
                        ),
                        Annotation.START_TIME: BatchedTensorSeq(
                            torch.tensor(
                                [[1.2, 5.1, 7.8, 0.0], [5.1, 15.1, 25.1, 35.1]],
                                dtype=torch.float,
                            )
                        ),
                        Annotation.END_TIME: BatchedTensorSeq(
                            torch.tensor(
                                [[2.2, 6.1, 8.8, 0.0], [5.9, 19.5, 29.5, 39.5]],
                                dtype=torch.float,
                            )
                        ),
                    }
                )
            )
            assert metadata == {Annotation.ACTION_VOCAB: action_vocab}


def test_load_event_data_test(tmp_path: Path) -> None:
    action_vocab = Mock(spec=Vocabulary)
    create_vocab_mock = Mock(return_value=action_vocab)
    event_examples_mock = Mock(
        return_value=(
            {
                Annotation.VIDEO_ID: "video_validation_0",
                Annotation.ACTION_INDEX: torch.tensor([1, 2], dtype=torch.long),
                Annotation.START_TIME: torch.tensor([5.1, 15.1], dtype=torch.float),
                Annotation.END_TIME: torch.tensor([5.9, 19.5], dtype=torch.float),
            },
            {
                Annotation.VIDEO_ID: "video_test_0",
                Annotation.ACTION_INDEX: torch.tensor([5, 3, 3], dtype=torch.long),
                Annotation.START_TIME: torch.tensor([1.2, 5.1, 7.8], dtype=torch.float),
                Annotation.END_TIME: torch.tensor([2.2, 6.1, 8.8], dtype=torch.float),
            },
            {
                Annotation.VIDEO_ID: "video_test_1",
                Annotation.ACTION_INDEX: torch.tensor([0, 1, 2, 3], dtype=torch.long),
                Annotation.START_TIME: torch.tensor([5.1, 15.1, 25.1, 35.1], dtype=torch.float),
                Annotation.END_TIME: torch.tensor([5.9, 19.5, 29.5, 39.5], dtype=torch.float),
            },
        )
    )
    with patch(
        "aroma.datasets.multithumos.create_action_vocabulary",
        create_vocab_mock,
    ):
        with patch(
            "aroma.datasets.multithumos.load_event_examples",
            event_examples_mock,
        ):
            data, metadata = load_event_data(tmp_path, split="test")
            create_vocab_mock.assert_called_once_with(tmp_path)
            event_examples_mock.assert_called_once_with(path=tmp_path, action_vocab=action_vocab)
            assert data.equal(
                BatchDict(
                    {
                        Annotation.VIDEO_ID: BatchList[str](["video_test_0", "video_test_1"]),
                        Annotation.ACTION_INDEX: BatchedTensorSeq(
                            torch.tensor([[5, 3, 3, 0], [0, 1, 2, 3]], dtype=torch.long)
                        ),
                        Annotation.START_TIME: BatchedTensorSeq(
                            torch.tensor(
                                [[1.2, 5.1, 7.8, 0.0], [5.1, 15.1, 25.1, 35.1]],
                                dtype=torch.float,
                            )
                        ),
                        Annotation.END_TIME: BatchedTensorSeq(
                            torch.tensor(
                                [[2.2, 6.1, 8.8, 0.0], [5.9, 19.5, 29.5, 39.5]],
                                dtype=torch.float,
                            )
                        ),
                    }
                )
            )
            assert metadata == {Annotation.ACTION_VOCAB: action_vocab}


def test_load_event_data_incorrect_split() -> None:
    with raises(RuntimeError, match="Incorrect split:"):
        load_event_data(Mock(spec=Path), split="incorrect")


##############################################
#     Tests for create_action_vocabulary     #
##############################################


def test_create_action_vocabulary(tmp_path: Path) -> None:
    save_text(
        "\n".join([f"{i} class{i:02d}" for i in range(65)]), tmp_path.joinpath("class_list.txt")
    )
    vocab = create_action_vocabulary(tmp_path)
    assert vocab.get_vocab_size() == 65
    assert vocab.get_index_to_token() == tuple([f"class{i:02d}" for i in range(65)])


def test_create_action_vocabulary_incorrect_size(tmp_path: Path) -> None:
    save_text(
        "\n".join([f"{i} class{i:02d}" for i in range(5)]), tmp_path.joinpath("class_list.txt")
    )
    with raises(RuntimeError):
        create_action_vocabulary(tmp_path)


##########################################
#     Tests for download_annotations     #
##########################################


def create_zip_file(path: Path) -> None:
    r"""Create a zip file that contains a single text file.

    Args:
        path (``pathlib.Path``): Specifies the filepath of the zip file to create.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(path, "w") as zfile:
        zfile.writestr("multithumos/README", "")
        zfile.writestr("multithumos/class_list.txt", "")
        zfile.writestr("multithumos/annotations", "")


@mark.parametrize("force_download", (True, False))
def test_download_annotations(tmp_path: Path, force_download: bool) -> None:
    tmpdir = tmp_path.joinpath("tmp")
    with patch("aroma.datasets.multithumos.download_url_to_file") as download_mock:
        with patch(
            "aroma.datasets.multithumos.TemporaryDirectory.__enter__",
            Mock(return_value=tmpdir),
        ):
            zip_file = tmpdir.joinpath("multithumos.zip.tmp")
            create_zip_file(
                zip_file
            )  # Create a zip file manually because the download function is mocked.
            download_annotations(tmp_path.joinpath("multithumos"), force_download=force_download)
            download_mock.assert_called_once_with(
                ANNOTATION_URL, zip_file.as_posix(), progress=True, hash_prefix=None
            )
            assert tmp_path.joinpath("multithumos/README").is_file()
            assert tmp_path.joinpath("multithumos/class_list.txt").is_file()
            assert tmp_path.joinpath("multithumos/annotations").is_file()


def test_download_annotations_already_exists_force_download_false(tmp_path: Path) -> None:
    with patch(
        "aroma.datasets.multithumos.is_annotation_path_ready",
        Mock(return_value=True),
    ):
        download_annotations(tmp_path)
        # The file should not exist because the download step is skipped
        assert not tmp_path.joinpath("multithumos/README").is_file()


def test_download_annotations_already_exists_force_download_true(tmp_path: Path) -> None:
    tmpdir = tmp_path.joinpath("tmp")
    with patch(
        "aroma.datasets.multithumos.is_annotation_path_ready",
        Mock(return_value=True),
    ):
        with patch(
            "aroma.datasets.multithumos.TemporaryDirectory.__enter__",
            Mock(return_value=tmpdir),
        ):
            with patch("aroma.datasets.multithumos.download_url_to_file") as download_mock:
                zip_file = tmpdir.joinpath("multithumos.zip.tmp")
                create_zip_file(
                    zip_file
                )  # Create a zip file manually because the download function is mocked.
                download_annotations(tmp_path.joinpath("multithumos"), force_download=True)
                download_mock.assert_called_once_with(
                    ANNOTATION_URL, zip_file.as_posix(), progress=True, hash_prefix=None
                )
                assert tmp_path.joinpath("multithumos/README").is_file()
                assert tmp_path.joinpath("multithumos/class_list.txt").is_file()
                assert tmp_path.joinpath("multithumos/annotations").is_file()


##############################################
#     Tests for is_annotation_path_ready     #
##############################################


def test_is_annotation_path_ready_true(tmp_path: Path) -> None:
    save_text("", tmp_path.joinpath("README"))
    save_text("", tmp_path.joinpath("class_list.txt"))
    for i in range(65):
        save_text("", tmp_path.joinpath(f"annotations/{i + 1}.txt"))
    assert is_annotation_path_ready(tmp_path)


def test_is_annotation_path_ready_false_missing_readme(tmp_path: Path) -> None:
    assert not is_annotation_path_ready(tmp_path)


def test_is_annotation_path_ready_false_missing_class_list(tmp_path: Path) -> None:
    save_text("", tmp_path.joinpath("README.txt"))
    assert not is_annotation_path_ready(tmp_path)


def test_is_annotation_path_ready_false_missing_annotations(tmp_path: Path) -> None:
    save_text("", tmp_path.joinpath("README"))
    save_text("", tmp_path.joinpath("class_list.txt"))
    assert not is_annotation_path_ready(tmp_path)


def test_is_annotation_path_ready_false_missing_annotation_file(tmp_path: Path) -> None:
    save_text("", tmp_path.joinpath("README"))
    save_text("", tmp_path.joinpath("README.txt"))
    for i in range(64):
        save_text("", tmp_path.joinpath(f"annotations/{i + 1}.txt"))
    assert not is_annotation_path_ready(tmp_path)


#########################################
#     Tests for load_event_examples     #
#########################################


def test_load_event_examples() -> None:
    path = Mock(spec=Path)
    action_vocab = Mock(spec=Vocabulary)
    load_mock = Mock(
        return_value={
            "video_0": [
                {Annotation.START_TIME: 5.1, Annotation.ACTION_INDEX: 3, Annotation.END_TIME: 6.1},
                {Annotation.START_TIME: 7.8, Annotation.ACTION_INDEX: 3, Annotation.END_TIME: 8.8},
                {Annotation.START_TIME: 1.2, Annotation.ACTION_INDEX: 5, Annotation.END_TIME: 2.2},
            ],
            "video_1": [
                {Annotation.START_TIME: 5.1, Annotation.ACTION_INDEX: 3, Annotation.END_TIME: 5.7},
                {Annotation.START_TIME: 5.1, Annotation.ACTION_INDEX: 2, Annotation.END_TIME: 5.8},
                {Annotation.START_TIME: 5.1, Annotation.ACTION_INDEX: 1, Annotation.END_TIME: 5.9},
            ],
        }
    )
    with patch(
        "aroma.datasets.multithumos.load_all_event_annotations_per_video",
        load_mock,
    ):
        assert objects_are_equal(
            load_event_examples(path, action_vocab),
            (
                {
                    Annotation.VIDEO_ID: "video_0",
                    Annotation.START_TIME: torch.tensor([1.2, 5.1, 7.8], dtype=torch.float),
                    Annotation.ACTION_INDEX: torch.tensor([5, 3, 3], dtype=torch.long),
                    Annotation.END_TIME: torch.tensor([2.2, 6.1, 8.8], dtype=torch.float),
                },
                {
                    Annotation.VIDEO_ID: "video_1",
                    Annotation.START_TIME: torch.tensor([5.1, 5.1, 5.1], dtype=torch.float),
                    Annotation.ACTION_INDEX: torch.tensor([1, 2, 3], dtype=torch.long),
                    Annotation.END_TIME: torch.tensor([5.9, 5.8, 5.7], dtype=torch.float),
                },
            ),
        )
        load_mock.assert_called_once_with(path=path, action_vocab=action_vocab)


def test_load_event_examples_empty() -> None:
    path = Mock(spec=Path)
    action_vocab = Mock(spec=Vocabulary)
    load_mock = Mock(return_value={})
    with patch(
        "aroma.datasets.multithumos.load_all_event_annotations_per_video",
        load_mock,
    ):
        assert load_event_examples(path, action_vocab) == tuple()
        load_mock.assert_called_once_with(path=path, action_vocab=action_vocab)


##########################################################
#     Tests for load_all_event_annotations_per_video     #
##########################################################


def test_load_all_event_annotations_per_video(tmp_path: Path) -> None:
    vocab = Vocabulary(Counter(["action1", "action2"]))
    save_text(
        "\n".join(
            [
                "video_validation_0000266 72.80 76.40",
                "video_validation_0000681 44.00 50.90",
                "video_validation_0000682 1.50 5.40",
                "video_validation_0000682 79.30 83.90",
            ]
        ),
        tmp_path.joinpath("annotations/action1.txt"),
    )
    save_text(
        "\n".join(
            [
                "video_validation_0000902 2.97 3.60",
                "video_validation_0000902 4.54 5.07",
                "video_validation_0000902 20.22 20.49",
                "video_validation_0000682 17.57 18.33",
            ]
        ),
        tmp_path.joinpath("annotations/action2.txt"),
    )
    assert objects_are_equal(
        load_all_event_annotations_per_video(tmp_path, vocab),
        {
            "video_validation_0000266": [
                {
                    Annotation.ACTION_INDEX: 0,
                    Annotation.START_TIME: 72.80,
                    Annotation.END_TIME: 76.40,
                },
            ],
            "video_validation_0000681": [
                {
                    Annotation.ACTION_INDEX: 0,
                    Annotation.START_TIME: 44.00,
                    Annotation.END_TIME: 50.90,
                },
            ],
            "video_validation_0000682": [
                {
                    Annotation.ACTION_INDEX: 0,
                    Annotation.START_TIME: 1.50,
                    Annotation.END_TIME: 5.40,
                },
                {
                    Annotation.ACTION_INDEX: 0,
                    Annotation.START_TIME: 79.30,
                    Annotation.END_TIME: 83.90,
                },
                {
                    Annotation.ACTION_INDEX: 1,
                    Annotation.START_TIME: 17.57,
                    Annotation.END_TIME: 18.33,
                },
            ],
            "video_validation_0000902": [
                {
                    Annotation.ACTION_INDEX: 1,
                    Annotation.START_TIME: 2.97,
                    Annotation.END_TIME: 3.60,
                },
                {
                    Annotation.ACTION_INDEX: 1,
                    Annotation.START_TIME: 4.54,
                    Annotation.END_TIME: 5.07,
                },
                {
                    Annotation.ACTION_INDEX: 1,
                    Annotation.START_TIME: 20.22,
                    Annotation.END_TIME: 20.49,
                },
            ],
        },
    )


#############################################################
#     Tests for load_single_event_annotations_per_video     #
#############################################################


def test_load_single_event_annotations_per_video(tmp_path: Path) -> None:
    save_text(
        "\n".join(
            [
                "video_validation_0000266 72.80 76.40",
                "video_validation_0000681 44.00 50.90",
                "video_validation_0000682 1.50 5.40",
                "video_validation_0000682 79.30 83.90",
            ]
        ),
        tmp_path.joinpath("action.txt"),
    )
    assert objects_are_equal(
        load_single_event_annotations_per_video(tmp_path.joinpath("action.txt"), action_index=42),
        {
            "video_validation_0000266": [
                {
                    Annotation.ACTION_INDEX: 42,
                    Annotation.START_TIME: 72.80,
                    Annotation.END_TIME: 76.40,
                },
            ],
            "video_validation_0000681": [
                {
                    Annotation.ACTION_INDEX: 42,
                    Annotation.START_TIME: 44.00,
                    Annotation.END_TIME: 50.90,
                },
            ],
            "video_validation_0000682": [
                {
                    Annotation.ACTION_INDEX: 42,
                    Annotation.START_TIME: 1.50,
                    Annotation.END_TIME: 5.40,
                },
                {
                    Annotation.ACTION_INDEX: 42,
                    Annotation.START_TIME: 79.30,
                    Annotation.END_TIME: 83.90,
                },
            ],
        },
    )


###########################################
#     Tests for prepare_event_example     #
###########################################


def test_prepare_event_example_empty() -> None:
    assert prepare_event_example("video_0", []) == {Annotation.VIDEO_ID: "video_0"}


def test_prepare_event_example() -> None:
    assert prepare_event_example(
        "video_0",
        [
            {Annotation.START_TIME: 5.1, Annotation.ACTION_INDEX: 3},
            {Annotation.START_TIME: 7.8, Annotation.ACTION_INDEX: 3},
            {Annotation.START_TIME: 1.2, Annotation.ACTION_INDEX: 5},
        ],
    ) == {
        Annotation.VIDEO_ID: "video_0",
        Annotation.START_TIME: [1.2, 5.1, 7.8],
        Annotation.ACTION_INDEX: [5, 3, 3],
    }


def test_prepare_event_example_extra_keys() -> None:
    assert prepare_event_example(
        "video_0",
        [
            {Annotation.START_TIME: 5.1, Annotation.ACTION_INDEX: 3, Annotation.END_TIME: 5.7},
            {Annotation.START_TIME: 7.8, Annotation.ACTION_INDEX: 3, Annotation.END_TIME: 9.2},
            {Annotation.START_TIME: 1.2, Annotation.ACTION_INDEX: 5, Annotation.END_TIME: 3.1},
        ],
    ) == {
        Annotation.VIDEO_ID: "video_0",
        Annotation.START_TIME: [1.2, 5.1, 7.8],
        Annotation.ACTION_INDEX: [5, 3, 3],
        Annotation.END_TIME: [3.1, 5.7, 9.2],
    }


def test_prepare_event_example_same_start_time() -> None:
    assert prepare_event_example(
        "video_0",
        [
            {Annotation.START_TIME: 5.1, Annotation.ACTION_INDEX: 3},
            {Annotation.START_TIME: 5.1, Annotation.ACTION_INDEX: 2},
            {Annotation.START_TIME: 5.1, Annotation.ACTION_INDEX: 1},
        ],
    ) == {
        Annotation.VIDEO_ID: "video_0",
        Annotation.START_TIME: [5.1, 5.1, 5.1],
        Annotation.ACTION_INDEX: [1, 2, 3],
    }


###############################################
#     Tests for sort_examples_by_video_id     #
###############################################


def test_sort_examples_by_video_id() -> None:
    assert sort_examples_by_video_id(
        [
            {Annotation.VIDEO_ID: "video_10"},
            {Annotation.VIDEO_ID: "video_00"},
            {Annotation.VIDEO_ID: "video_07"},
        ],
    ) == [
        {Annotation.VIDEO_ID: "video_00"},
        {Annotation.VIDEO_ID: "video_07"},
        {Annotation.VIDEO_ID: "video_10"},
    ]


def test_sort_examples_by_video_id_descending_true() -> None:
    assert sort_examples_by_video_id(
        [
            {Annotation.VIDEO_ID: "video_10"},
            {Annotation.VIDEO_ID: "video_00"},
            {Annotation.VIDEO_ID: "video_07"},
        ],
        descending=True,
    ) == [
        {Annotation.VIDEO_ID: "video_10"},
        {Annotation.VIDEO_ID: "video_07"},
        {Annotation.VIDEO_ID: "video_00"},
    ]


##########################################
#     Tests for filter_test_examples     #
##########################################


def test_filter_test_examples() -> None:
    assert filter_test_examples(
        [
            {Annotation.VIDEO_ID: "video_test_0000179", "other": "abc"},
            {Annotation.VIDEO_ID: "video_validation_0000910", "other": "def"},
            {Annotation.VIDEO_ID: "video_test_0001257", "other": "ghj"},
        ]
    ) == [
        {Annotation.VIDEO_ID: "video_test_0000179", "other": "abc"},
        {Annotation.VIDEO_ID: "video_test_0001257", "other": "ghj"},
    ]


def test_filter_test_examples_empty() -> None:
    assert filter_test_examples([]) == []


################################################
#     Tests for filter_validation_examples     #
################################################


def test_filter_validation_examples() -> None:
    assert filter_validation_examples(
        [
            {Annotation.VIDEO_ID: "video_test_0000179", "other": "abc"},
            {Annotation.VIDEO_ID: "video_validation_0000910", "other": "def"},
            {Annotation.VIDEO_ID: "video_test_0001257", "other": "ghj"},
        ]
    ) == [{Annotation.VIDEO_ID: "video_validation_0000910", "other": "def"}]


def test_filter_validation_examples_empty() -> None:
    assert filter_validation_examples([]) == []
