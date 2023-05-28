import logging
from collections import Counter

from coola import EqualityTester
from pytest import LogCaptureFixture, mark

from aroma.utils.vocab import Vocabulary, VocabularyEqualityOperator

################################
#     Tests for Vocabulary     #
################################


@mark.parametrize("size", (1, 2, 3))
def test_vocabulary_len(size: int) -> None:
    assert len(Vocabulary(Counter({i: 10 for i in range(size)}))) == size


def test_vocabulary_repr() -> None:
    assert repr(Vocabulary(Counter({}))).startswith("Vocabulary(")


def test_vocabulary_str() -> None:
    assert str(Vocabulary(Counter({}))).startswith("Vocabulary(")


def test_vocabulary_equal_true() -> None:
    assert Vocabulary(Counter({"b": 3, "a": 1, "c": 2})).equal(
        Vocabulary(Counter({"b": 3, "a": 1, "c": 2}))
    )


def test_vocabulary_equal_false_different_order() -> None:
    assert not Vocabulary(Counter({"b": 3, "a": 1, "c": 2})).equal(
        Vocabulary(Counter({"a": 1, "b": 3, "c": 2}))
    )


def test_vocabulary_equal_false_different_count() -> None:
    assert not Vocabulary(Counter({"b": 3, "a": 1, "c": 2})).equal(
        Vocabulary(Counter({"b": 4, "a": 1, "c": 2}))
    )


def test_vocabulary_equal_false_different_tokens() -> None:
    assert not Vocabulary(Counter({"b": 3, "a": 1, "c": 2})).equal(
        Vocabulary(Counter({"d": 3, "a": 1, "c": 2}))
    )


def test_vocabulary_equal_false_different_type() -> None:
    assert not Vocabulary(Counter({"b": 3, "a": 1, "c": 2})).equal({})


def test_vocabulary_get_index() -> None:
    vocab = Vocabulary(Counter({"b": 3, "a": 1, "c": 2}))
    assert vocab.get_index("a") == 1
    assert vocab.get_index("b") == 0
    assert vocab.get_index("c") == 2


def test_vocabulary_get_index_to_token() -> None:
    assert Vocabulary(Counter({"b": 3, "a": 1, "c": 2})).get_index_to_token() == ("b", "a", "c")


def test_vocabulary_get_token() -> None:
    vocab = Vocabulary(Counter({"b": 3, "a": 1, "c": 2}))
    assert vocab.get_token(0) == "b"
    assert vocab.get_token(1) == "a"
    assert vocab.get_token(2) == "c"


def test_vocabulary_get_token_to_index() -> None:
    assert Vocabulary(Counter({"b": 3, "a": 1, "c": 2})).get_token_to_index() == {
        "b": 0,
        "a": 1,
        "c": 2,
    }


def test_vocabulary_get_vocab_size() -> None:
    assert Vocabulary(Counter({"b": 3, "a": 1, "c": 2})).get_vocab_size() == 3


def test_vocabulary_load_state_dict() -> None:
    vocab = Vocabulary(Counter({}))
    vocab.load_state_dict(
        {
            "index_to_token": ("b", "a", "c"),
            "token_to_index": {"b": 0, "a": 1, "c": 2},
            "counter": Counter({"b": 3, "a": 1, "c": 2}),
        }
    )
    assert vocab.state_dict() == {
        "index_to_token": ("b", "a", "c"),
        "token_to_index": {"b": 0, "a": 1, "c": 2},
        "counter": Counter({"b": 3, "a": 1, "c": 2}),
    }


def test_vocabulary_state_dict() -> None:
    assert Vocabulary(Counter({"b": 3, "a": 1, "c": 2})).state_dict() == {
        "index_to_token": ("b", "a", "c"),
        "token_to_index": {"b": 0, "a": 1, "c": 2},
        "counter": Counter({"b": 3, "a": 1, "c": 2}),
    }


def test_vocabulary_add() -> None:
    vocab1 = Vocabulary(Counter({"b": 3, "a": 1, "c": 2}))
    vocab2 = Vocabulary(Counter({"b": 3, "d": 7}))
    assert vocab1.add(vocab2).get_index_to_token() == ("b", "a", "c", "d")
    assert vocab2.add(vocab1).get_index_to_token() == ("b", "d", "a", "c")


def test_vocabulary_sub() -> None:
    vocab1 = Vocabulary(Counter({"b": 3, "a": 1, "c": 2}))
    vocab2 = Vocabulary(Counter({"b": 3, "d": 7}))
    assert vocab1.sub(vocab2).get_index_to_token() == ("a", "c")


def test_vocabulary_sort_by_count() -> None:
    assert Vocabulary(Counter({"b": 3, "a": 1, "c": 2})).sort_by_count().get_index_to_token() == (
        "b",
        "c",
        "a",
    )


def test_vocabulary_sort_by_count_descending_false() -> None:
    assert Vocabulary(Counter({"b": 3, "a": 1, "c": 2})).sort_by_count(
        descending=False
    ).get_index_to_token() == (
        "a",
        "c",
        "b",
    )


def test_vocabulary_sort_by_count_duplicate_count() -> None:
    assert Vocabulary(Counter({"b": 1, "a": 1, "c": 1})).sort_by_count().get_index_to_token() == (
        "c",
        "b",
        "a",
    )


def test_vocabulary_sort_by_token() -> None:
    assert Vocabulary(Counter({"b": 3, "a": 1, "c": 2})).sort_by_token().get_index_to_token() == (
        "a",
        "b",
        "c",
    )


def test_vocabulary_sort_by_token_descending_true() -> None:
    assert Vocabulary(Counter({"b": 3, "a": 1, "c": 2})).sort_by_token(
        descending=True
    ).get_index_to_token() == (
        "c",
        "b",
        "a",
    )


def test_vocabulary_most_common_2() -> None:
    assert Vocabulary(Counter({"b": 3, "a": 1, "c": 2})).most_common(2).get_index_to_token() == (
        "b",
        "c",
    )


def test_vocabulary_most_common_100() -> None:
    assert Vocabulary(Counter({"b": 3, "a": 1, "c": 2})).most_common(100).get_index_to_token() == (
        "b",
        "c",
        "a",
    )


################################################
#     Tests for VocabularyEqualityOperator     #
################################################


def test_registered_vocabulary_comparators() -> None:
    assert isinstance(EqualityTester.registry[Vocabulary], VocabularyEqualityOperator)


def test_vocabulary_equality_operator_str() -> None:
    assert str(VocabularyEqualityOperator()) == "VocabularyEqualityOperator()"


def test_vocabulary_equality_operator_equal_true() -> None:
    assert VocabularyEqualityOperator().equal(
        EqualityTester(),
        Vocabulary(Counter({"b": 3, "a": 1, "c": 2})),
        Vocabulary(Counter({"b": 3, "a": 1, "c": 2})),
    )


def test_vocabulary_equality_operator_equal_true_show_difference(caplog: LogCaptureFixture) -> None:
    with caplog.at_level(logging.INFO):
        assert VocabularyEqualityOperator().equal(
            tester=EqualityTester(),
            object1=Vocabulary(Counter({"b": 3, "a": 1, "c": 2})),
            object2=Vocabulary(Counter({"b": 3, "a": 1, "c": 2})),
            show_difference=True,
        )
        assert not caplog.messages


def test_vocabulary_equality_operator_equal_false_different_value() -> None:
    assert not VocabularyEqualityOperator().equal(
        EqualityTester(),
        Vocabulary(Counter({"b": 3, "a": 1, "c": 2})),
        Vocabulary(Counter({"b": 3, "a": 1, "c": 3})),
    )


def test_vocabulary_equality_operator_equal_false_different_value_show_difference(
    caplog: LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.INFO):
        assert not VocabularyEqualityOperator().equal(
            tester=EqualityTester(),
            object1=Vocabulary(Counter({"b": 3, "a": 1, "c": 2})),
            object2=Vocabulary(Counter({"b": 3, "a": 1, "c": 3})),
            show_difference=True,
        )
        assert caplog.messages[0].startswith("`Vocabulary` objects are different")


def test_vocabulary_equality_operator_equal_false_different_type() -> None:
    assert not VocabularyEqualityOperator().equal(
        EqualityTester(), Vocabulary(Counter({"b": 3, "a": 1, "c": 2})), 42
    )


def test_vocabulary_equality_operator_equal_false_different_type_show_difference(
    caplog: LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.INFO):
        assert not VocabularyEqualityOperator().equal(
            tester=EqualityTester(),
            object1=Vocabulary(Counter({"b": 3, "a": 1, "c": 2})),
            object2=42,
            show_difference=True,
        )
        assert caplog.messages[0].startswith("object2 is not a `Vocabulary` object")
