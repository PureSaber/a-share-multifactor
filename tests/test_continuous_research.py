import json
from copy import deepcopy
from pathlib import Path

import pandas as pd
import pytest
from quant_lab.research import candidates
from test_research_workbench import recipe as recipe_fixture

from a_share_multifactor.research_workbench import EquityResearchExecutor

recipe = recipe_fixture


@pytest.mark.parametrize(
    "starts,reason", [(("2020-01-02", "2020-01-02"), "unique"), (("2020-01-03",), "cover")]
)
def test_certified_replay_rejects_invalid_boundaries_before_replay(starts, reason):
    from test_certified_execution import _certified_panel

    from a_share_multifactor.config import AppConfig
    from a_share_multifactor.run_contract import _replay

    with pytest.raises(ValueError, match=reason):
        _replay(_certified_panel().head(18), AppConfig(), "bad-boundaries", segment_starts=starts)


def selections(recipe, candidate):
    calendar = pd.read_parquet(Path(recipe["inputs"]["bundle"]) / "calendar.parquet")
    days = pd.DatetimeIndex(pd.to_datetime(calendar.date))
    days = days[(days >= recipe["interval"]["start"]) & (days <= recipe["interval"]["end"])]
    chunks = (days[:11], days[11:23], days[23:])
    return [
        {
            "train": {"start": "2024-01-01", "end": str((part[0] - pd.Timedelta(days=1)).date())},
            "test": {"start": str(part[0].date()), "end": str(part[-1].date())},
            "candidate": deepcopy(candidate),
        }
        for part in chunks
    ]


@pytest.mark.parametrize("allocation", [None, {"mode": "equal"}])
def test_identical_parameters_segmented_and_one_pass_match_exactly(recipe, tmp_path, allocation):
    if allocation:
        recipe["allocation"] = allocation
    candidate = candidates(recipe)[0]
    executor = EquityResearchExecutor()
    one, many = tmp_path / "one", tmp_path / "many"
    one.mkdir()
    many.mkdir()
    plain = executor(recipe, candidate, one)
    continuous = executor.continuous(recipe, selections(recipe, candidate), many)
    assert continuous["metrics"] == plain["metrics"]
    assert (one / "returns.csv").read_bytes() == (many / "returns.csv").read_bytes()
    for artifact in ("orders", "fills", "ledger", "account_snapshots"):
        left = one / f"standard/v2/{artifact}.parquet"
        right = many / f"standard/v2/{artifact}.parquet"
        if left.exists():
            pd.testing.assert_frame_equal(pd.read_parquet(left), pd.read_parquet(right))
    assert json.loads((one / "execution_diagnostics.json").read_text()) == json.loads(
        (many / "execution_diagnostics.json").read_text()
    )


def test_continuous_selection_rejects_leakage_gaps_and_fee_contract_changes(recipe, tmp_path):
    candidate = candidates(recipe)[0]
    for mutation in ("leak", "gap", "fees"):
        choices = selections(recipe, candidate)
        if mutation == "leak":
            choices[0]["train"]["end"] = choices[0]["test"]["start"]
        elif mutation == "gap":
            choices.pop(1)
        else:
            choices[1]["candidate"]["cost_multiplier"] = 2
        out = tmp_path / mutation
        out.mkdir()
        with pytest.raises(ValueError):
            EquityResearchExecutor().continuous(recipe, choices, out)
