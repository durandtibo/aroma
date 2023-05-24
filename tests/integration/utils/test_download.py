from pathlib import Path

from aroma.testing import gdown_available
from aroma.utils.download import download_drive_file

#########################################
#     Tests for download_drive_file     #
#########################################


@gdown_available
def test_download_drive_file(tmp_path: Path) -> None:
    url = "https://docs.google.com/document/d/1PK1HGa3HViKSJhAhvQgZNEYB72J0DhcXPNKuSpI4N80"
    path = tmp_path.joinpath("data.txt")
    assert not path.is_file()
    download_drive_file(url, path)
    assert path.is_file()
