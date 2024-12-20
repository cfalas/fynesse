from . import datasets
from . import db

from pathlib import Path


def test_boundaries(tmp_path: str):
    filename = Path(tmp_path) / "boundaries.zip"
    conn = db.MockConnection()
    datasets.boundaries(
        "4d4e021d-fe98-4a0e-88e2-3ead84538537",
        "test_table",
        filename,
        conn,
    )
    assert filename.exists()
