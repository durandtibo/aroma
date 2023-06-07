from __future__ import annotations

__all__ = ["check_polars", "is_polars_available"]

from importlib.util import find_spec

##################
#     polars     #
##################


def check_polars() -> None:
    r"""Checks if the ``polars`` package is installed.

    Raises
    ------
        RuntimeError if the ``polars`` package is not installed.
    """
    if not is_polars_available():
        raise RuntimeError(
            "`polars` package is required but not installed. "
            "You can install `polars` package with the command:\n\n"
            "pip install polars\n"
        )


def is_polars_available() -> bool:
    r"""Indicates if the ``polars`` package is installed or not.

    https://www.pola.rs/
    """
    return find_spec("polars") is not None


#################
#     gdown
##################


def check_gdown() -> None:
    r"""Checks if the ``gdown`` package is installed.

    Raises
    ------
        RuntimeError if the ``gdown`` package is not installed.
    """
    if not is_gdown_available():
        raise RuntimeError(
            "`gdown` package is required but not installed. "
            "You can install `gdown` package with the command:\n\n"
            "pip install gdown\n"
        )


def is_gdown_available() -> bool:
    r"""Indicates if the ``gdown`` package is installed or not.

    https://github.com/wkentaro/gdown
    """
    return find_spec("gdown") is not None
