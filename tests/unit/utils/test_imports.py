from __future__ import annotations

from unittest.mock import patch

from pytest import raises

from aroma.utils.imports import (
    check_gdown,
    check_polars,
    is_gdown_available,
    is_polars_available,
)

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


#################
#     gdown     #
#################


def test_check_gdown_with_package() -> None:
    with patch("aroma.utils.imports.is_gdown_available", lambda *args: True):
        check_gdown()


def test_check_gdown_without_package() -> None:
    with patch("aroma.utils.imports.is_gdown_available", lambda *args: False):
        with raises(RuntimeError, match="`gdown` package is required but not installed."):
            check_gdown()


def test_is_gdown_available() -> None:
    assert isinstance(is_gdown_available(), bool)
