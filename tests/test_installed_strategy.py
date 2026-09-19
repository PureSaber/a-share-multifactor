import json
from types import SimpleNamespace

import pytest

from a_share_multifactor import run_contract


def test_installed_strategy_requires_an_immutable_vcs_revision(tmp_path, monkeypatch):
    revision = "a" * 40
    monkeypatch.setattr(
        run_contract.importlib.metadata,
        "distribution",
        lambda _: SimpleNamespace(
            read_text=lambda _: json.dumps({"vcs_info": {"commit_id": revision}})
        ),
    )
    assert run_contract._code_version(tmp_path) == revision
    monkeypatch.setattr(
        run_contract.importlib.metadata,
        "distribution",
        lambda _: SimpleNamespace(read_text=lambda _: "{}"),
    )
    with pytest.raises(RuntimeError, match="immutable VCS"):
        run_contract._code_version(tmp_path)
