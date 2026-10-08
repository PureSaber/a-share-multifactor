import hashlib
import json
import sys
from copy import deepcopy

import pandas as pd
import pytest
from quant_data_kit.research_coverage import asof_history, attach_history, import_history

from a_share_multifactor.disclosures import import_disclosures, load_research_history, main


@pytest.fixture
def package(tmp_path):
    root = tmp_path / "source"
    root.mkdir()
    documents, records = [], []
    for i, (date, value) in enumerate([("2025-08-13", "10"), ("2025-09-01", "12")]):
        name = f"document-{i}.txt"
        payload = f"Synthetic disclosure revision {i}".encode()
        (root / name).write_bytes(payload)
        documents.append(
            {
                "document_id": f"doc-{i}",
                "source_uri": "https://example.test/synthetic",
                "file": name,
                "sha256": hashlib.sha256(payload).hexdigest(),
                "captured_at": f"{date}T10:00:00+08:00",
                "publication": {
                    "precision": "exact",
                    "value": f"{date}T09:00:00+08:00",
                    "evidence_document_id": f"doc-{i}",
                },
            }
        )
        records.append(
            {
                "record_id": f"revision-{i}",
                "symbol": "600519",
                "field": "net_profit",
                "value": value,
                "unit": "CNY_1e4",
                "effective_at": "2025-06-30T00:00:00+08:00",
                "document_id": f"doc-{i}",
                "supersedes": f"revision-{i - 1}" if i else None,
                "locator": "synthetic statement page 1",
            }
        )
    content = {
        "schema": "a-share.disclosures/v1",
        "provider": "synthetic-test",
        "license_note": "generated fixture",
        "documents": documents,
        "records": records,
    }
    return root / "input.json", content, tmp_path / "snapshot"


def write(package):
    source, content, output = package
    source.write_text(json.dumps(content), encoding="utf-8")
    return source, output


def test_capture_default_and_revision_visibility(package):
    source, output = write(package)
    manifest = import_disclosures(source, output)
    _, frame = load_research_history(output)
    assert manifest["disclosure"]["field_units"] == {"net_profit": "CNY"}
    assert manifest["disclosure"]["historical_authenticity_certified"] is False
    assert asof_history(
        frame, as_of="2025-08-13T01:30:00Z", domain="fundamentals", field="net_profit"
    ).empty
    assert asof_history(
        frame, as_of="2025-08-13T02:00:00Z", domain="fundamentals", field="net_profit"
    ).value.tolist() == ["100000"]
    assert asof_history(
        frame, as_of="2025-09-01T02:00:00Z", domain="fundamentals", field="net_profit"
    ).value.tolist() == ["120000"]
    prices = pd.DataFrame(
        {"symbol": ["600519"] * 2, "date": pd.to_datetime(["2025-08-13", "2025-09-01"])}
    )
    attached = attach_history(prices, frame, {"net_profit": "fundamentals"})
    assert attached.net_profit.tolist() == [100000, 120000]
    with pytest.raises(FileExistsError):
        import_disclosures(source, output)


def test_explicit_source_declaration_does_not_claim_authentication(package):
    source, output = write(package)
    manifest = import_disclosures(source, output, policy="source-declared")
    _, frame = load_research_history(output)
    assert str(frame.iloc[0].available_at) == "2025-08-13 01:00:00+00:00"
    assert manifest["disclosure"]["historical_authenticity_certified"] is False


@pytest.mark.parametrize("precision,value", [("unknown", None), ("date", "2025-08-13")])
def test_inexact_publication_is_capture_only(package, precision, value):
    package[1]["documents"][0]["publication"] = {
        "precision": precision,
        "value": value,
        "evidence_document_id": None,
    }
    source, output = write(package)
    with pytest.raises(ValueError, match="Historical admission"):
        import_disclosures(source, output, policy="source-declared")
    assert not output.exists()
    import_disclosures(source, output)
    assert load_research_history(output)[1].available_at.iloc[0] == pd.Timestamp(
        "2025-08-13T02:00Z"
    )


@pytest.mark.parametrize(
    "change",
    [
        "hash",
        "path",
        "duplicate",
        "nan",
        "unit",
        "units_conflict",
        "chain",
        "fork",
        "missing_doc",
        "missing_evidence",
        "future_publication",
        "naive",
        "reserved",
        "extra",
    ],
)
def test_bad_evidence_never_publishes_snapshot(package, change):
    _, content, output = package
    document, record = content["documents"][0], content["records"][0]
    if change == "hash":
        document["sha256"] = "0" * 64
    elif change == "path":
        document["file"] = "../outside.txt"
    elif change == "duplicate":
        content["documents"].append(deepcopy(document))
    elif change == "nan":
        record["value"] = "NaN"
    elif change == "unit":
        record["unit"] = "inferred"
    elif change == "units_conflict":
        content["records"][1]["unit"] = "shares"
    elif change == "chain":
        content["records"][1]["supersedes"] = None
    elif change == "fork":
        content["records"].append({**record, "record_id": "fork"})
    elif change == "missing_doc":
        record["document_id"] = "missing"
    elif change == "missing_evidence":
        document["publication"]["evidence_document_id"] = "missing"
    elif change == "future_publication":
        document["publication"]["value"] = "2027-01-01T00:00Z"
    elif change == "naive":
        record["effective_at"] = "2025-06-30"
    elif change == "reserved":
        record["field"] = "close"
    else:
        record["extra"] = "ignored?"
    with pytest.raises(ValueError):
        import_disclosures(*write(package))
    assert not output.exists()


@pytest.mark.parametrize("change", ["material", "lineage", "history", "unit", "certification"])
def test_consumer_rechecks_sources_and_semantics(package, change):
    import_disclosures(*write(package))
    output = package[2]
    path = output / "manifest.json"
    manifest = json.loads(path.read_text())
    if change == "material":
        (output / "disclosure" / "document-0.txt").write_text("tampered")
    elif change == "lineage":
        manifest["disclosure"]["lineage"][0]["normalized_value"] = "999"
    elif change == "history":
        table = output / "history.parquet"
        frame = pd.read_parquet(table)
        frame.loc[0, "value"] = "999"
        frame.to_parquet(table, index=False)
        manifest["history"]["sha256"] = hashlib.sha256(table.read_bytes()).hexdigest()
    elif change == "unit":
        manifest["disclosure"]["field_units"]["net_profit"] = "ratio"
    else:
        manifest["disclosure"]["historical_authenticity_certified"] = True
    path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError):
        load_research_history(output)


def test_cli_and_existing_generic_import(package, monkeypatch, capsys, tmp_path):
    source, output = write(package)
    monkeypatch.setattr(
        sys, "argv", ["asm-disclosures", "--input", str(source), "--out", str(output)]
    )
    main()
    assert json.loads(capsys.readouterr().out)["rows"] == 2
    original = tmp_path / "old.csv"
    load_research_history(output)[1].to_csv(original, index=False)
    old = tmp_path / "old-history"
    import_history(original, old, provider="test", source_uri="test", license_note="test")
    assert len(load_research_history(old)[1]) == 2
