"""Auditable disclosure-to-history imports; source declarations are not certification."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from decimal import Decimal, InvalidOperation
from pathlib import Path
from tempfile import TemporaryDirectory
from urllib.parse import urlsplit

import pandas as pd
from quant_data_kit.research_coverage import import_history, load_history, validate_history

SCHEMA = "a-share.disclosures/v1"
UNITS = {
    "CNY": ("CNY", Decimal(1)),
    "CNY_1e4": ("CNY", Decimal(10000)),
    "CNY_1e8": ("CNY", Decimal(100000000)),
    "shares": ("shares", Decimal(1)),
    "ratio": ("ratio", Decimal(1)),
    "percent": ("ratio", Decimal("0.01")),
    "CNY/share": ("CNY/share", Decimal(1)),
}
RESERVED = {"date", "symbol", "open", "high", "low", "close", "volume", "amount", "available_at"}


def _hash(raw):
    return hashlib.sha256(raw).hexdigest()


def _text(value):
    if not isinstance(value, str) or not value.strip():
        raise ValueError("Disclosure strings must be nonempty")
    return value


def _keys(item, expected):
    if not isinstance(item, dict) or set(item) != set(expected.split()):
        raise ValueError(f"Disclosure fields must be exactly: {expected}")


def _time(value):
    stamp = pd.Timestamp(_text(value))
    if pd.isna(stamp) or stamp.tzinfo is None:
        raise ValueError("Disclosure timestamps require explicit time zones")
    return stamp.tz_convert("UTC")


def _local(root, name):
    name = _text(name)
    path = Path(name)
    if path.is_absolute() or ".." in path.parts or ":" in name or "\\" in name:
        raise ValueError("Disclosure material must be a relative confined path")
    resolved = (root / path).resolve()
    if not resolved.is_relative_to(root.resolve()) or resolved == root.resolve():
        raise ValueError("Disclosure material escapes input directory")
    return resolved


def _parse(source, policy):
    if policy not in {"captured", "source-declared"}:
        raise ValueError("Availability policy must be captured or source-declared")
    raw = source.read_bytes()
    config = json.loads(raw)
    _keys(config, "schema provider license_note documents records")
    if config["schema"] != SCHEMA:
        raise ValueError("Unsupported disclosure schema")
    for key in ("provider", "license_note"):
        _text(config[key])
    if not isinstance(config["documents"], list) or not isinstance(config["records"], list):
        raise TypeError("Disclosure documents and records must be arrays")
    documents, materials, locations = {}, {}, set()
    for item in config["documents"]:
        _keys(item, "document_id source_uri file sha256 captured_at publication")
        identity = _text(item["document_id"])
        if identity in documents:
            raise ValueError("Duplicate disclosure document identity")
        uri = urlsplit(_text(item["source_uri"]))
        if uri.scheme != "https" or not uri.netloc or uri.username or uri.password:
            raise ValueError("Disclosure source_uri must be an HTTPS source without credentials")
        path = _local(source.parent, item["file"])
        if path == source.resolve() or item["file"].casefold() in locations:
            raise ValueError("Duplicate or conflicting disclosure material path")
        locations.add(item["file"].casefold())
        payload = path.read_bytes()
        if _hash(payload) != item["sha256"]:
            raise ValueError("Disclosure material hash mismatch")
        captured = _time(item["captured_at"])
        publication = item["publication"]
        _keys(publication, "precision value evidence_document_id")
        precision = publication["precision"]
        published = None
        if precision == "exact":
            published = _time(publication["value"])
            _text(publication["evidence_document_id"])
            if published > captured:
                raise ValueError("Publication is later than material capture")
        elif precision == "date":
            value = publication["value"]
            if not isinstance(value, str) or not re.fullmatch(r"\d{4}-\d{2}-\d{2}", value):
                raise ValueError("Date-only publication requires YYYY-MM-DD")
            pd.Timestamp(value)
            if publication["evidence_document_id"] is not None:
                _text(publication["evidence_document_id"])
        elif (
            precision != "unknown"
            or publication["value"] is not None
            or publication["evidence_document_id"] is not None
        ):
            raise ValueError("Unknown publication must not contain invented timing evidence")
        documents[identity] = (item, captured, published)
        materials[item["file"]] = payload
    for item, _, _ in documents.values():
        evidence = item["publication"]["evidence_document_id"]
        if evidence is not None and evidence not in documents:
            raise ValueError("Publication evidence document is missing")
    rows, lineage, identities, units, revisions = [], [], set(), {}, {}
    for item in config["records"]:
        _keys(item, "record_id symbol field value unit effective_at document_id supersedes locator")
        identity = _text(item["record_id"])
        if identity in identities:
            raise ValueError("Duplicate disclosure record identity")
        identities.add(identity)
        if not re.fullmatch(r"[0-9]{6}", _text(item["symbol"])):
            raise ValueError("A-share symbol must contain six digits")
        field = _text(item["field"])
        if not re.fullmatch(r"[a-z][a-z0-9_]*", field) or field in RESERVED:
            raise ValueError("Invalid or reserved disclosure field")
        _text(item["locator"])
        if item["unit"] not in UNITS or item["document_id"] not in documents:
            raise ValueError("Unknown disclosure unit or source document")
        unit, scale = UNITS[item["unit"]]
        if units.setdefault(field, unit) != unit:
            raise ValueError("Inconsistent units for one disclosure field")
        try:
            value = Decimal(_text(item["value"])) * scale
        except InvalidOperation as exc:
            raise ValueError("Invalid disclosure number") from exc
        if not value.is_finite():
            raise ValueError("Disclosure values must be finite")
        effective = _time(item["effective_at"])
        document, captured, published = documents[item["document_id"]]
        if policy == "source-declared":
            if published is None:
                raise ValueError("Historical admission requires explicit publication evidence")
            available = published
        else:
            available = captured
        rows.append(
            {
                "domain": "fundamentals",
                "symbol": item["symbol"],
                "field": field,
                "effective_at": effective,
                "available_at": available,
                "value": str(value),
            }
        )
        lineage.append(
            {
                **item,
                "normalized_unit": unit,
                "normalized_value": str(value),
                "available_at": available.isoformat(),
                "captured_at": captured.isoformat(),
                "source_uri": document["source_uri"],
                "material_sha256": document["sha256"],
            }
        )
        key = (item["symbol"], field, effective)
        revisions.setdefault(key, []).append((available, identity, item["supersedes"]))
    for chain in revisions.values():
        previous = None
        previous_time = None
        for time, identity, parent in sorted(chain):
            if parent != previous or (previous_time is not None and time <= previous_time):
                raise ValueError(
                    "Disclosure revisions require an ordered, unambiguous supersedes chain"
                )
            previous, previous_time = identity, time
    frame = validate_history(pd.DataFrame(rows))
    return raw, config, materials, frame, sorted(lineage, key=lambda item: item["record_id"])


def import_disclosures(source, output, *, policy="captured"):
    source, output = Path(source), Path(output)
    if output.exists():
        raise FileExistsError(output)
    raw, config, materials, frame, lineage = _parse(source, policy)
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".disclosures-", dir=output.parent) as temporary:
        work = Path(temporary)
        table = work / "normalized.csv"
        frame.to_csv(table, index=False)
        snapshot = work / "snapshot"
        manifest = import_history(
            table,
            snapshot,
            provider=config["provider"],
            source_uri="disclosure:input.json",
            license_note=config["license_note"],
        )
        evidence = snapshot / "disclosure"
        evidence.mkdir()
        (evidence / "input.json").write_bytes(raw)
        for name, payload in materials.items():
            target = _local(evidence, name)
            if target == evidence / "input.json":
                raise ValueError("Material path conflicts with disclosure input")
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(payload)
        manifest["disclosure"] = {
            "schema": SCHEMA,
            "policy": policy,
            "input_sha256": _hash(raw),
            "field_units": {x["field"]: x["normalized_unit"] for x in lineage},
            "lineage": lineage,
            "historical_authenticity_certified": False,
        }
        (snapshot / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        load_research_history(snapshot)
        snapshot.rename(output)
    return manifest


def load_research_history(root):
    root = Path(root)
    manifest, frame = load_history(root)
    if "disclosure" not in manifest:
        return manifest, frame  # Existing generic source-declaration imports remain supported.
    evidence = manifest["disclosure"]
    _keys(
        evidence, "schema policy input_sha256 field_units lineage historical_authenticity_certified"
    )
    if evidence["schema"] != SCHEMA or evidence["historical_authenticity_certified"] is not False:
        raise ValueError("Disclosure manifest cannot certify historical authenticity")
    raw, _, _, expected, lineage = _parse(root / "disclosure" / "input.json", evidence["policy"])
    if (
        _hash(raw) != evidence["input_sha256"]
        or lineage != evidence["lineage"]
        or evidence["field_units"] != {x["field"]: x["normalized_unit"] for x in lineage}
    ):
        raise ValueError("Disclosure lineage or input hash mismatch")
    try:
        pd.testing.assert_frame_equal(frame, expected)
    except AssertionError as exc:
        raise ValueError("Disclosure history differs from its source evidence") from exc
    return manifest, frame


def main():
    parser = argparse.ArgumentParser(description="Import reviewed A-share disclosure evidence")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument(
        "--availability", choices=["captured", "source-declared"], default="captured"
    )
    args = parser.parse_args()
    result = import_disclosures(args.input, args.out, policy=args.availability)
    print(
        json.dumps(
            {
                "rows": result["rows"],
                "symbols": result["symbols"],
                "availability": args.availability,
                "historical_authenticity_certified": False,
            }
        )
    )


if __name__ == "__main__":
    main()
