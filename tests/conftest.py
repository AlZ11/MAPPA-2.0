import gzip
from pathlib import Path

import pytest

from mappa.store import Store

FIXTURES = Path(__file__).parent / "fixtures"


def fixture_html(name: str) -> str:
    return gzip.decompress((FIXTURES / name).read_bytes()).decode("utf-8")


@pytest.fixture
def store(tmp_path):
    return Store(tmp_path / "data")
