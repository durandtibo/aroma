__all__ = ["download_drive_file"]

from pathlib import Path

from aroma.utils.imports import check_gdown, is_gdown_available

if is_gdown_available():
    import gdown
else:
    gdown = None  # pragma: no cover


def download_drive_file(url: str, path: Path, *args, **kwargs) -> None:
    r"""Download a file from Google Drive.

    Args:
        url (str): Specifies the Google Drive URL.
        path (``pathlib.Path``): Specifies the path where to store the downloaded file.
        *args: See the documentation of ``gdown.download``
        **kwargs: See the documentation of ``gdown.download``

    Example usage:

    .. code-block:: python

        >>> from pathlib import Path
        >>> from aroma.utils.download import download_drive_file
        >>> download_drive_file(
        ...     "https://drive.google.com/open?id=1R3z_CkO1uIOhu4y2Nh0pCHjQQ2l-Ab9E",
        ...     Path('/path/to/data.tar.gz'),
        ...     quiet=False,
        ...     fuzzy=True,
        ... )
    """
    check_gdown()
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.is_file():
        # Save to tmp, then commit by moving the file in case the job gets
        # interrupted while writing the file
        tmp_path = path.with_name(f"{path.name}.tmp")
        gdown.download(url, tmp_path.as_posix(), *args, **kwargs)
        tmp_path.rename(path)
