"""Build and certify the metadata-only AKShare wheel used by this project."""

from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import io
import json
import os
import tempfile
import urllib.request
import zipfile
from pathlib import Path, PurePosixPath

OFFICIAL_VERSION = "1.18.88"
DERIVED_VERSION = "1.18.88.post1"
OFFICIAL_FILENAME = "akshare-1.18.88-py3-none-any.whl"
DERIVED_FILENAME = "akshare-1.18.88.post1-py3-none-any.whl"
OFFICIAL_URL = (
    "https://files.pythonhosted.org/packages/8e/24/"
    "f4a3d9d58993a67bf3ead06e44ad9dcd062eaa1e2b719be4be8e7e2646cf/"
    f"{OFFICIAL_FILENAME}"
)
OFFICIAL_SHA256 = "ba0b06ea2d341122e2ef8ed5e4982ff5925f01111e63d7940c8a01aa684578c0"
DERIVED_SHA256 = "de6d7f77008299b5c96e13559542e153ac58d0780523c0d5b45694ce97e2099f"

OFFICIAL_DIST_INFO = "akshare-1.18.88.dist-info"
DERIVED_DIST_INFO = "akshare-1.18.88.post1.dist-info"
OFFICIAL_METADATA = f"{OFFICIAL_DIST_INFO}/METADATA"
OFFICIAL_RECORD = f"{OFFICIAL_DIST_INFO}/RECORD"
DERIVED_METADATA = f"{DERIVED_DIST_INFO}/METADATA"
DERIVED_RECORD = f"{DERIVED_DIST_INFO}/RECORD"

UPSTREAM_PROVIDER_REQUIREMENTS = (
    b'Requires-Dist: mini-racer>=0.12.4; platform_system != "Linux"',
    b'Requires-Dist: py-mini-racer>=0.6.0; platform_system == "Linux"',
    b'Requires-Dist: akracer>=0.0.13; platform_system == "Linux"',
)
DERIVED_PROVIDER_REQUIREMENT = b"Requires-Dist: mini-racer==0.14.1"

# ZIP_STORED avoids platform/zlib-specific compressed byte streams. These
# attributes are deliberately independent of the source checkout and clock.
FIXED_ZIP_TIMESTAMP = (2020, 1, 1, 0, 0, 0)
REGULAR_FILE_MODE = 0o100644


class WheelCertificationError(ValueError):
    """Raised when a wheel violates the fixed derivation contract."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _assert_source_hash(source: Path, expected_sha256: str) -> None:
    actual = sha256_file(source)
    if actual != expected_sha256:
        raise WheelCertificationError(
            f"official wheel SHA-256 mismatch: expected {expected_sha256}, got {actual}"
        )


def _validate_member_name(name: str) -> None:
    path = PurePosixPath(name)
    if not name or name.endswith("/") or "\\" in name or path.is_absolute() or ".." in path.parts:
        raise WheelCertificationError(f"unsafe or unsupported wheel member: {name!r}")


def _read_members(source: Path) -> dict[str, bytes]:
    with zipfile.ZipFile(source) as archive:
        members: dict[str, bytes] = {}
        for info in archive.infolist():
            _validate_member_name(info.filename)
            if info.filename in members:
                raise WheelCertificationError(f"duplicate wheel member: {info.filename}")
            members[info.filename] = archive.read(info)
    return members


def _rewrite_metadata(payload: bytes) -> bytes:
    output: list[bytes] = []
    version_count = 0
    provider_counts = {requirement: 0 for requirement in UPSTREAM_PROVIDER_REQUIREMENTS}
    provider_inserted = False

    for line in payload.splitlines(keepends=True):
        content = line.rstrip(b"\r\n")
        ending = line[len(content) :] or b"\n"
        if content == b"Version: 1.18.88":
            output.append(b"Version: 1.18.88.post1" + ending)
            version_count += 1
            continue
        if content in provider_counts:
            provider_counts[content] += 1
            if not provider_inserted:
                output.append(DERIVED_PROVIDER_REQUIREMENT + ending)
                provider_inserted = True
            continue
        output.append(line)

    if version_count != 1:
        raise WheelCertificationError(f"expected one upstream Version field, found {version_count}")
    if any(count != 1 for count in provider_counts.values()):
        raise WheelCertificationError(
            f"unexpected upstream provider requirements: {provider_counts}"
        )
    if not provider_inserted:
        raise WheelCertificationError("derived MiniRacer requirement was not inserted")
    return b"".join(output)


def _rename_member(name: str) -> str:
    if name == OFFICIAL_DIST_INFO:
        return DERIVED_DIST_INFO
    prefix = f"{OFFICIAL_DIST_INFO}/"
    if name.startswith(prefix):
        return f"{DERIVED_DIST_INFO}/{name[len(prefix) :]}"
    return name


def _record_hash(payload: bytes) -> str:
    digest = base64.urlsafe_b64encode(hashlib.sha256(payload).digest()).rstrip(b"=")
    return f"sha256={digest.decode('ascii')}"


def _make_record(members_without_record: dict[str, bytes]) -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.writer(stream, lineterminator="\n")
    for name in sorted(members_without_record):
        payload = members_without_record[name]
        writer.writerow((name, _record_hash(payload), str(len(payload))))
    writer.writerow((DERIVED_RECORD, "", ""))
    return stream.getvalue().encode("utf-8")


def _derive_members(source_members: dict[str, bytes]) -> dict[str, bytes]:
    if OFFICIAL_METADATA not in source_members or OFFICIAL_RECORD not in source_members:
        raise WheelCertificationError("official wheel is missing METADATA or RECORD")
    derived: dict[str, bytes] = {}
    for source_name, payload in source_members.items():
        if source_name == OFFICIAL_RECORD:
            continue
        name = _rename_member(source_name)
        if name in derived:
            raise WheelCertificationError(f"derived member collision: {name}")
        derived[name] = _rewrite_metadata(payload) if source_name == OFFICIAL_METADATA else payload
    derived[DERIVED_RECORD] = _make_record(derived)
    return derived


def _write_deterministic_wheel(members: dict[str, bytes], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.name != DERIVED_FILENAME:
        raise WheelCertificationError(
            f"derived wheel must be named {DERIVED_FILENAME}, got {output.name}"
        )
    handle, temporary_name = tempfile.mkstemp(
        prefix=f".{output.name}.", suffix=".tmp", dir=output.parent
    )
    os.close(handle)
    temporary = Path(temporary_name)
    try:
        with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_STORED) as archive:
            for name in sorted(members):
                info = zipfile.ZipInfo(name, date_time=FIXED_ZIP_TIMESTAMP)
                info.compress_type = zipfile.ZIP_STORED
                info.create_system = 3
                info.external_attr = REGULAR_FILE_MODE << 16
                info.internal_attr = 0
                archive.writestr(info, members[name])
        os.replace(temporary, output)
    finally:
        if temporary.exists():
            temporary.unlink()


def repack_wheel(
    source: Path,
    output: Path,
    *,
    expected_source_sha256: str = OFFICIAL_SHA256,
) -> str:
    """Create the deterministic metadata-only derivative and return its SHA-256."""

    source = source.resolve()
    output = output.resolve()
    if source == output:
        raise WheelCertificationError("source and output paths must differ")
    _assert_source_hash(source, expected_source_sha256)
    _write_deterministic_wheel(_derive_members(_read_members(source)), output)
    return sha256_file(output)


def verify_derived_wheel(
    source: Path,
    derived: Path,
    *,
    expected_source_sha256: str = OFFICIAL_SHA256,
    expected_derived_sha256: str | None = None,
) -> dict[str, object]:
    """Verify hashes, entry topology, metadata edits, RECORD, and payload identity."""

    _assert_source_hash(source, expected_source_sha256)
    actual_derived_sha256 = sha256_file(derived)
    if expected_derived_sha256 and actual_derived_sha256 != expected_derived_sha256:
        raise WheelCertificationError(
            "derived wheel SHA-256 mismatch: "
            f"expected {expected_derived_sha256}, got {actual_derived_sha256}"
        )

    source_members = _read_members(source)
    derived_members = _read_members(derived)
    expected_members = _derive_members(source_members)
    if set(derived_members) != set(expected_members):
        missing = sorted(set(expected_members) - set(derived_members))
        extra = sorted(set(derived_members) - set(expected_members))
        raise WheelCertificationError(
            f"derived wheel member mismatch: missing={missing}, extra={extra}"
        )

    unexpected_payload_differences = [
        name for name in sorted(expected_members) if derived_members[name] != expected_members[name]
    ]
    if unexpected_payload_differences:
        raise WheelCertificationError(
            f"unexpected derived payload differences: {unexpected_payload_differences}"
        )

    runtime_payload_differences = []
    dist_info_payload_differences = []
    for source_name, source_payload in source_members.items():
        if source_name == OFFICIAL_RECORD:
            continue
        derived_name = _rename_member(source_name)
        if derived_members[derived_name] == source_payload:
            continue
        if source_name.startswith(f"{OFFICIAL_DIST_INFO}/"):
            dist_info_payload_differences.append(derived_name)
        else:
            runtime_payload_differences.append(derived_name)

    if runtime_payload_differences:
        raise WheelCertificationError(f"runtime payload changed: {runtime_payload_differences}")
    if dist_info_payload_differences != [DERIVED_METADATA]:
        raise WheelCertificationError(
            "only METADATA may differ before RECORD regeneration, got "
            f"{dist_info_payload_differences}"
        )

    metadata = derived_members[DERIVED_METADATA]
    if b"Version: 1.18.88.post1" not in metadata:
        raise WheelCertificationError("derived version is missing from METADATA")
    if metadata.count(DERIVED_PROVIDER_REQUIREMENT) != 1:
        raise WheelCertificationError("derived MiniRacer requirement is not unique")
    if any(requirement in metadata for requirement in UPSTREAM_PROVIDER_REQUIREMENTS):
        raise WheelCertificationError("an upstream overlapping MiniRacer provider remains")

    source_notices = sorted(
        name
        for name in source_members
        if PurePosixPath(name).name.upper() in {"NOTICE", "NOTICE.TXT", "NOTICE.MD"}
    )
    derived_notices = sorted(
        name
        for name in derived_members
        if PurePosixPath(name).name.upper() in {"NOTICE", "NOTICE.TXT", "NOTICE.MD"}
    )
    license_name = f"{OFFICIAL_DIST_INFO}/licenses/LICENSE"
    derived_license_name = f"{DERIVED_DIST_INFO}/licenses/LICENSE"
    if source_members.get(license_name) != derived_members.get(derived_license_name):
        raise WheelCertificationError("LICENSE payload is not byte-identical to the official wheel")

    return {
        "official_sha256": sha256_file(source),
        "derived_sha256": actual_derived_sha256,
        "entry_count": len(derived_members),
        "runtime_payload_diff_count": len(runtime_payload_differences),
        "dist_info_payload_diff_count_before_record": len(dist_info_payload_differences),
        "dist_info_payload_differences_before_record": dist_info_payload_differences,
        "license_payload_identical": True,
        "source_notice_entries": source_notices,
        "derived_notice_entries": derived_notices,
    }


def fetch_official_wheel(output: Path) -> str:
    """Download the exact PyPI wheel and reject any content hash mismatch."""

    output.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(
        OFFICIAL_URL, headers={"User-Agent": "PureSaber-wheel-audit/1"}
    )
    handle, temporary_name = tempfile.mkstemp(prefix=f".{output.name}.", dir=output.parent)
    os.close(handle)
    temporary = Path(temporary_name)
    try:
        with urllib.request.urlopen(request, timeout=120) as response, temporary.open("wb") as sink:
            while block := response.read(1024 * 1024):
                sink.write(block)
        _assert_source_hash(temporary, OFFICIAL_SHA256)
        os.replace(temporary, output)
    finally:
        if temporary.exists():
            temporary.unlink()
    return sha256_file(output)


def certify_committed_wheel(work_dir: Path, committed_wheel: Path) -> dict[str, object]:
    """Download once, rebuild twice, and certify the committed wheel."""

    if not DERIVED_SHA256:
        raise WheelCertificationError("DERIVED_SHA256 has not been frozen in the tool")
    work_dir.mkdir(parents=True, exist_ok=True)
    source = work_dir / OFFICIAL_FILENAME
    rebuild_one_dir = work_dir / "rebuild-one"
    rebuild_two_dir = work_dir / "rebuild-two"
    rebuild_one = rebuild_one_dir / DERIVED_FILENAME
    rebuild_two = rebuild_two_dir / DERIVED_FILENAME
    fetch_official_wheel(source)
    first_hash = repack_wheel(source, rebuild_one)
    second_hash = repack_wheel(source, rebuild_two)
    committed_report = verify_derived_wheel(
        source,
        committed_wheel,
        expected_derived_sha256=DERIVED_SHA256,
    )
    first_report = verify_derived_wheel(
        source,
        rebuild_one,
        expected_derived_sha256=DERIVED_SHA256,
    )
    second_report = verify_derived_wheel(
        source,
        rebuild_two,
        expected_derived_sha256=DERIVED_SHA256,
    )
    if first_hash != second_hash or first_hash != committed_report["derived_sha256"]:
        raise WheelCertificationError(
            "two rebuilds and the committed wheel are not byte-identical: "
            f"{first_hash}, {second_hash}, {committed_report['derived_sha256']}"
        )
    return {
        "official_url": OFFICIAL_URL,
        "official_sha256": OFFICIAL_SHA256,
        "committed_sha256": committed_report["derived_sha256"],
        "rebuild_sha256s": [first_hash, second_hash],
        "payload": first_report,
        "second_rebuild_payload": second_report,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    fetch = subparsers.add_parser("fetch", help="download and hash-check the official wheel")
    fetch.add_argument("--output", type=Path, required=True)

    build = subparsers.add_parser("build", help="create the deterministic derived wheel")
    build.add_argument("--source", type=Path, required=True)
    build.add_argument("--output", type=Path, required=True)

    verify = subparsers.add_parser("verify", help="verify one derived wheel")
    verify.add_argument("--source", type=Path, required=True)
    verify.add_argument("--derived", type=Path, required=True)
    verify.add_argument("--expected-derived-sha256")

    certify = subparsers.add_parser(
        "certify", help="download, rebuild twice, and compare with the committed wheel"
    )
    certify.add_argument("--work-dir", type=Path, required=True)
    certify.add_argument("--committed-wheel", type=Path, required=True)
    return parser


def main() -> int:
    args = _parser().parse_args()
    if args.command == "fetch":
        result: object = {"official_sha256": fetch_official_wheel(args.output)}
    elif args.command == "build":
        result = {"derived_sha256": repack_wheel(args.source, args.output)}
    elif args.command == "verify":
        result = verify_derived_wheel(
            args.source,
            args.derived,
            expected_derived_sha256=args.expected_derived_sha256,
        )
    else:
        result = certify_committed_wheel(args.work_dir, args.committed_wheel)
    print(json.dumps(result, ensure_ascii=True, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
