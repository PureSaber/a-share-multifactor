"""The ordinary wheel must carry the same explicit catalog as research fixtures."""

import subprocess
import sys
from pathlib import Path
from zipfile import ZipFile

from a_share_multifactor import run_contract

ROOT = Path(__file__).resolve().parents[1]
RESOURCE = "a_share_multifactor/data/a_share_instrument_catalog_v1.csv"


def test_default_catalog_is_packaged():
    catalog = run_contract._CATALOG_PATH
    assert catalog.parent.parent == Path(run_contract.__file__).resolve().parent
    assert (
        catalog.read_bytes()
        == (ROOT / "data/fixtures/a_share_instrument_catalog_v1.csv").read_bytes()
    )
    assert {"000001", "510300"}.issubset(set(run_contract.load_fixture_catalog()["symbol"]))


def test_wheel_catalog(tmp_path):
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "wheel",
            "--no-index",
            "--no-deps",
            "--no-build-isolation",
            "--wheel-dir",
            str(tmp_path),
            str(ROOT),
        ],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
    )
    (wheel,) = tmp_path.glob("a_share_multifactor-*.whl")
    with ZipFile(wheel) as archive:
        assert (
            archive.read(RESOURCE)
            == (ROOT / "data/fixtures/a_share_instrument_catalog_v1.csv").read_bytes()
        )
