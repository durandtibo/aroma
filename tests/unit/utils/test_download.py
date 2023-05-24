from pathlib import Path
from unittest.mock import patch

from gravitorch.utils.io import load_text, save_text

from aroma.testing import gdown_available
from aroma.utils.download import download_drive_file

#########################################
#     Tests for download_drive_file     #
#########################################


@gdown_available
def test_download_drive_file(tmp_path: Path) -> None:
    url = "https://drive.google.com/open?id=123456789ABCDEFGHIJKLMN"
    path = tmp_path.joinpath("data.txt")
    save_text("abc", tmp_path.joinpath("data.txt.tmp"))
    with patch("aroma.utils.download.gdown") as gdown_mock:
        download_drive_file(url, path)
        gdown_mock.download.assert_called_once_with(
            url, tmp_path.joinpath("data.txt.tmp").as_posix()
        )
    assert load_text(path) == "abc"


@gdown_available
def test_download_drive_file_already_exist(tmp_path: Path) -> None:
    url = "https://drive.google.com/open?id=123456789ABCDEFGHIJKLMN"
    path = tmp_path.joinpath("data.txt")
    save_text("abc", path)
    with patch("aroma.utils.download.gdown") as gdown_mock:
        download_drive_file(url, path)
        gdown_mock.download.assert_not_called()
    assert load_text(path) == "abc"
