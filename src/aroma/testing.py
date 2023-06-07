from __future__ import annotations

__all__ = ["gdown_available"]

from pytest import mark

from aroma.utils.imports import is_gdown_available

gdown_available = mark.skipif(
    not is_gdown_available(),
    reason=("`gdown` is not available. Please install `gdown` if you want to run this test"),
)
