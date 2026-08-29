from __future__ import annotations

import hashlib
import zipfile
from pathlib import Path

import pytest

from tools.repack_akshare_wheel import (
    DERIVED_FILENAME,
    DERIVED_METADATA,
    DERIVED_RECORD,
    OFFICIAL_DIST_INFO,
    OFFICIAL_METADATA,
    WheelCertificationError,
    repack_wheel,
    verify_derived_wheel,
)


def _write_official_wheel(path: Path, *, include_all_providers: bool = True) -> str:
    requirements = [
        b'Requires-Dist: mini-racer>=0.12.4; platform_system != "Linux"\n',
        b'Requires-Dist: py-mini-racer>=0.6.0; platform_system == "Linux"\n',
        b'Requires-Dist: akracer>=0.0.13; platform_system == "Linux"\n',
    ]
    if not include_all_providers:
        requirements.pop()
    metadata = b"".join(
        [
            b"Metadata-Version: 2.4\n",
            b"Name: akshare\n",
            b"Version: 1.18.88\n",
            *requirements,
        ]
    )
    members = {
        "akshare/__init__.py": b"from ._version import __version__\n",
        "akshare/_version.py": b'__version__ = "1.18.88"\n',
        "akshare/data/payload.bin": b"\x00\r\n\xff\n",
        f"{OFFICIAL_DIST_INFO}/licenses/LICENSE": b"official-license\r\n",
        OFFICIAL_METADATA: metadata,
        f"{OFFICIAL_DIST_INFO}/WHEEL": b"Wheel-Version: 1.0\nTag: py3-none-any\n",
        f"{OFFICIAL_DIST_INFO}/RECORD": b"source-record-is-regenerated\n",
    }
    with zipfile.ZipFile(path, "w") as archive:
        for name, payload in reversed(list(members.items())):
            archive.writestr(name, payload)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_repack_is_deterministic_and_preserves_non_metadata_payloads(tmp_path: Path) -> None:
    source = tmp_path / "official.whl"
    source_hash = _write_official_wheel(source)
    first = tmp_path / "first" / DERIVED_FILENAME
    second = tmp_path / "second" / DERIVED_FILENAME

    first_hash = repack_wheel(source, first, expected_source_sha256=source_hash)
    second_hash = repack_wheel(source, second, expected_source_sha256=source_hash)

    assert first_hash == second_hash
    assert first.read_bytes() == second.read_bytes()
    report = verify_derived_wheel(
        source,
        first,
        expected_source_sha256=source_hash,
        expected_derived_sha256=first_hash,
    )
    assert report["runtime_payload_diff_count"] == 0
    assert report["license_payload_identical"] is True
    assert report["source_notice_entries"] == []
    with zipfile.ZipFile(source) as upstream, zipfile.ZipFile(first) as derived:
        for name in ("akshare/_version.py", "akshare/data/payload.bin"):
            assert derived.read(name) == upstream.read(name)
        assert b"Version: 1.18.88.post1" in derived.read(DERIVED_METADATA)
        assert DERIVED_RECORD in derived.namelist()


def test_repack_rejects_wrong_source_hash(tmp_path: Path) -> None:
    source = tmp_path / "official.whl"
    _write_official_wheel(source)

    with pytest.raises(WheelCertificationError, match="official wheel SHA-256 mismatch"):
        repack_wheel(
            source,
            tmp_path / DERIVED_FILENAME,
            expected_source_sha256="0" * 64,
        )


def test_repack_rejects_incomplete_provider_metadata(tmp_path: Path) -> None:
    source = tmp_path / "official.whl"
    source_hash = _write_official_wheel(source, include_all_providers=False)

    with pytest.raises(WheelCertificationError, match="upstream provider requirements"):
        repack_wheel(
            source,
            tmp_path / DERIVED_FILENAME,
            expected_source_sha256=source_hash,
        )
