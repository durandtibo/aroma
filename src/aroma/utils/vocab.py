from __future__ import annotations

__all__ = ["Vocabulary"]

import logging
from collections import Counter
from collections.abc import Hashable
from typing import Any, Generic, TypeVar

from coola import (
    BaseEqualityOperator,
    BaseEqualityTester,
    EqualityTester,
    objects_are_equal,
)

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=Hashable)


class Vocabulary(Generic[T]):
    r"""Implements a vocabulary built from a counter of tokens.

    Args:
        counter (``Counter``): Specifies the counter used to generate
            the vocabulary. The order of the items in the counter is
            used to define the index-to-token and token-to-index
            mappings.
    """

    def __init__(self, counter: Counter) -> None:
        self._counter = counter
        self._index_to_token = tuple(self._counter.keys())
        self._token_to_index = {token: i for i, token in enumerate(self._index_to_token)}

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__qualname__}(\n"
            f"  counter={self.counter},\n"
            f"  index_to_token={self.get_index_to_token()},\n"
            f"  token_to_index={self.get_token_to_index()},\n"
            f")"
        )

    def __str__(self) -> str:
        return f"{self.__class__.__qualname__}(vocab_size={self.get_vocab_size():,})"

    def __len__(self) -> int:
        return len(self._counter)

    @property
    def counter(self) -> Counter:
        r"""``Counter``: The counter of the vocabulary."""
        return self._counter

    def equal(self, other: Any) -> bool:
        r"""Indicates if two vocabularies are equal or not.

        Args:
            other: Specifies the value to compare.

        Returns:
            bool: ``True`` if the vocabularies are equal,
                ``False`` otherwise.
        """
        if not isinstance(other, Vocabulary):
            return False
        return (
            objects_are_equal(self.counter, other.counter)
            and objects_are_equal(self.get_index_to_token(), other.get_index_to_token())
            and objects_are_equal(self.get_token_to_index(), other.get_token_to_index())
        )

    def get_index(self, token: T) -> int:
        r"""Gets the index for a given index.

        Args:
            token: Specifies a token.

        Returns:
            int: The token index.

        Example usage:

        .. code-block:: pycon

            >>> from collections import Counter
            >>> from aroma.utils.vocab import Vocabulary
            >>> vocab = Vocabulary(Counter({"b": 3, "a": 1, "c": 2}))
            >>> vocab.get_index("a")
            1
            >>> vocab.get_index("b")
            0
            >>> vocab.get_index("c")
            2
        """
        return self._token_to_index[token]

    def get_index_to_token(self) -> tuple[T, ...]:
        r"""Gets the index to token mapping.

        Returns:
            dict: The index to token mapping.

        Example usage:

        .. code-block:: pycon

            >>> from collections import Counter
            >>> from aroma.utils.vocab import Vocabulary
            >>> vocab = Vocabulary(Counter({"b": 3, "a": 1, "c": 2}))
            >>> vocab.get_index_to_token()
            ('b', 'a', 'c')
        """
        return self._index_to_token

    def get_token(self, index: int) -> T:
        r"""Gets the token for a given index.

        Args:
            index: Specifies a index.

        Returns:
            The token associated to the index.

        Example usage:

        .. code-block:: pycon

            >>> from collections import Counter
            >>> from aroma.utils.vocab import Vocabulary
            >>> vocab = Vocabulary(Counter({"b": 3, "a": 1, "c": 2}))
            >>> vocab.get_token(0)
            'b'
            >>> vocab.get_token(1)
            'a'
            >>> vocab.get_token(2)
            'c'
        """
        return self._index_to_token[index]

    def get_token_to_index(self) -> dict[T, int]:
        r"""Gets the token to index mapping.

        Returns:
            dict: The token to index mapping.

        Example usage:

        .. code-block:: pycon

            >>> from collections import Counter
            >>> from aroma.utils.vocab import Vocabulary
            >>> vocab = Vocabulary(Counter({"b": 3, "a": 1, "c": 2}))
            >>> vocab.get_token_to_index()
            {'b': 0, 'a': 1, 'c': 2}
        """
        return self._token_to_index

    def get_vocab_size(self) -> int:
        r"""Gets the vocabulary size.

        Returns:
            int: The vocabulary size.

        Example usage:

        .. code-block:: pycon

            >>> from collections import Counter
            >>> from aroma.utils.vocab import Vocabulary
            >>> vocab = Vocabulary(Counter({"b": 3, "a": 1, "c": 2}))
            >>> vocab.get_vocab_size()
            3
        """
        return len(self)

    def load_state_dict(self, state_dict: dict) -> None:
        r"""Loads a state dict to the current vocabulary.

        Args:
            state_dict (dict): Specifies the state dict to load.

        Example usage:

        .. code-block:: pycon

            >>> from collections import Counter
            >>> from aroma.utils.vocab import Vocabulary
            >>> vocab = Vocabulary(Counter({}))
            >>> vocab.state_dict()
            {'counter': Counter(), 'index_to_token': (), 'token_to_index': {}}
            >>> vocab.load_state_dict(
            ...     {
            ...         "index_to_token": ("b", "a", "c"),
            ...         "token_to_index": {"b": 0, "a": 1, "c": 2},
            ...         "counter": Counter({"b": 3, "a": 1, "c": 2}),
            ...     }
            ... )
            >>> vocab.state_dict()
            {'counter': Counter({'b': 3, 'a': 1, 'c': 2}),
             'index_to_token': ('b', 'a', 'c'),
             'token_to_index': {'b': 0, 'a': 1, 'c': 2}}
        )
        """
        self._counter = state_dict["counter"]
        self._index_to_token = state_dict["index_to_token"]
        self._token_to_index = state_dict["token_to_index"]

    def state_dict(self) -> dict:
        r"""Gets the state dict of the vocabulary.

        Returns:
            dict: The state dict which contains 3 keys: ``"counter"``,
                ``"index_to_token"``, and ``"token_to_index"``.

        Example usage:

        .. code-block:: pycon

            >>> from collections import Counter
            >>> from aroma.utils.vocab import Vocabulary
            >>> vocab = Vocabulary(Counter({"b": 3, "a": 1, "c": 2}))
            >>> vocab.state_dict()
            {'counter': Counter({'b': 3, 'a': 1, 'c': 2}),
             'index_to_token': ('b', 'a', 'c'),
             'token_to_index': {'b': 0, 'a': 1, 'c': 2}}
        """
        return {
            "counter": self._counter,
            "index_to_token": self._index_to_token,
            "token_to_index": self._token_to_index,
        }

    def add(self, other: Vocabulary) -> Vocabulary:
        r"""Creates a new vocabulary where elements from ``other`` are added to
        ``self``.

        Args:
            other (``Vocabulary``): Specifies the vocabulary to add.

        Returns:
            ``Vocabulary``: The new vocabulary.

        Example usage:

        .. code-block:: pycon

            >>> from collections import Counter
            >>> from aroma.utils.vocab import Vocabulary
            >>> vocab1 = Vocabulary(Counter({"b": 3, "a": 1, "c": 2}))
            >>> vocab2 = Vocabulary(Counter({"b": 3, "d": 7}))
            >>> vocab = vocab1.add(vocab2)
            >>> vocab.counter
            Counter({'b': 6, 'a': 1, 'c': 2, 'd': 7})
            >>> vocab.get_index_to_token()
            ('b', 'a', 'c', 'd')
        """
        return Vocabulary(self.counter + other.counter)

    def sub(self, other: Vocabulary) -> Vocabulary:
        r"""Creates a new vocabulary where elements from ``other`` are removed
        from ``self``.

        Args:
            other (``Vocabulary``): Specifies the vocabulary to
                subtract.

        Returns:
            ``Vocabulary``: The new vocabulary.

        Example usage:

        .. code-block:: pycon

            >>> from collections import Counter
            >>> from aroma.utils.vocab import Vocabulary
            >>> vocab1 = Vocabulary(Counter({"b": 3, "a": 1, "c": 2}))
            >>> vocab2 = Vocabulary(Counter({"b": 3, "d": 7}))
            >>> vocab = vocab1.sub(vocab2)
            >>> vocab.counter
            Counter({'a': 1, 'c': 2})
            >>> vocab.get_index_to_token()
            ('a', 'c')
        """
        return Vocabulary(self.counter - other.counter)

    def sort_by_count(self, descending: bool = True) -> Vocabulary:
        r"""Creates a new vocabulary where the counter is sorted by
        count.

        If multiple tokens have the same count, they are sorted by
        token values.

        Args:
            descending (bool, optional): If ``True``, the items are
                sorted in descending order by token.
                Default: ``False``

        Returns:
            ``Vocabulary``: The new vocabulary where the counter is
                sorted by count.

        Example usage:

        .. code-block:: pycon

            >>> from collections import Counter
            >>> from aroma.utils.vocab import Vocabulary
            >>> vocab = Vocabulary(Counter({"b": 3, "a": 1, "c": 2})).sort_by_count()
            >>> vocab.counter
            Counter({'b': 3, 'c': 2, 'a': 1})
            >>> vocab.get_index_to_token()
            ('b', 'c', 'a')
        """
        return Vocabulary(
            Counter(
                dict(
                    sorted(
                        self.counter.items(),
                        key=lambda item: (item[1], item[0]),
                        reverse=descending,
                    )
                )
            )
        )

    def sort_by_token(self, descending: bool = False) -> Vocabulary:
        r"""Creates a new vocabulary where the counter is sorted by
        token.

        Args:
            descending (bool, optional): If ``True``, the items are
                sorted in descending order by token.
                Default: ``False``

        Returns:
            ``Vocabulary``: The new vocabulary where the counter is
                sorted by token.

        Example usage:

        .. code-block:: pycon

            >>> from collections import Counter
            >>> from aroma.utils.vocab import Vocabulary
            >>> vocab = Vocabulary(Counter({"b": 3, "a": 1, "c": 2})).sort_by_token()
            >>> vocab.counter
            Counter({'a': 1, 'b': 3, 'c': 2})
            >>> vocab.get_index_to_token()
            ('a', 'b', 'c')
        """
        return Vocabulary(Counter(dict(sorted(self.counter.items(), reverse=descending))))

    def most_common(self, max_num_tokens: int) -> Vocabulary:
        r"""Gets a new vocabulary with the ``max_num_tokens`` most common tokens
        of the current vocabulary.

        Args:
            max_num_tokens (int): Specifies the maximum number of
                tokens.

        Returns:
            ``Vocabulary``: The new vocabulary with the most common
                tokens. The counter is sorted by decreasing order of
                count.

        Example usage:

        .. code-block:: pycon

            >>> from collections import Counter
            >>> from aroma.utils.vocab import Vocabulary
            >>> vocab = Vocabulary(Counter({"b": 3, "a": 1, "c": 2})).most_common(2)
            >>> vocab.counter
            Counter({'b': 3, 'c': 2})
            >>> vocab.get_index_to_token()
            ('b', 'c')
        """
        return Vocabulary(Counter(dict(self.counter.most_common(max_num_tokens))))


class VocabularyEqualityOperator(BaseEqualityOperator[Vocabulary]):
    r"""Implements an equality operator for ``Vocabulary`` objects."""

    def __eq__(self, other: Any) -> bool:
        return isinstance(other, self.__class__)

    def clone(self) -> VocabularyEqualityOperator:
        return self.__class__()

    def equal(
        self,
        tester: BaseEqualityTester,
        object1: Vocabulary,
        object2: Any,
        show_difference: bool = False,
    ) -> bool:
        if not isinstance(object2, Vocabulary):
            if show_difference:
                logger.info(f"object2 is not a `Vocabulary` object: {type(object2)}")
            return False
        object_equal = object1.equal(object2)
        if show_difference and not object_equal:
            logger.info(
                f"`Vocabulary` objects are different\nobject1=\n{object1}\nobject2=\n{object2}"
            )
        return object_equal


if not EqualityTester.has_operator(Vocabulary):
    EqualityTester.add_operator(Vocabulary, VocabularyEqualityOperator())  # pragma: no cover
