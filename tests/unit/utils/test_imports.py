from unittest.mock import patch

from pytest import raises

from aroma.utils.imports import check_polars, is_polars_available

##################
#     polars     #
##################


def test_check_polars_with_package() -> None:
    with patch("aroma.utils.imports.is_polars_available", lambda *args: True):
        check_polars()


def test_check_polars_without_package() -> None:
    with patch("aroma.utils.imports.is_polars_available", lambda *args: False):
        with raises(RuntimeError, match="`polars` package is required but not installed."):
            check_polars()


def test_is_polars_available() -> None:
    assert isinstance(is_polars_available(), bool)
