from datetime import datetime, timezone

import pytest
import test_decision_workflow as decision_tests
import yaml
from test_decision_workflow import _inputs, _run

from a_share_multifactor.holdout_review import evaluate

setup_decision = decision_tests.setup_decision


def test_forward_holdout_cannot_be_evaluated_early_and_seals_complete_window(
    setup_decision, tmp_path
):
    config, output = setup_decision
    dates, _ = _inputs(tmp_path / "first")
    settings = yaml.safe_load(config.read_text())
    settings["study"] = {
        "id": "future-test",
        "hypothesis": "fixed test",
        "holdout_start": str(dates[80].date()),
        "holdout_end": str(dates[84].date()),
    }
    config.write_text(yaml.safe_dump(settings))
    first = _run(config, output, tmp_path / "first", dates)
    with pytest.raises(ValueError, match="not ended"):
        evaluate(
            output / "experiments.db",
            "future-test",
            first,
            now=dates[79].tz_localize("UTC").to_pydatetime(),
        )
    _inputs(tmp_path / "second", last=84)
    with pytest.raises(ValueError, match="does not reach"):
        evaluate(
            output / "experiments.db",
            "future-test",
            first,
            now=datetime(2027, 1, 1, tzinfo=timezone.utc),
        )
    last = _run(config, output, tmp_path / "second", dates, last=84)
    result = evaluate(
        output / "experiments.db",
        "future-test",
        last,
        now=datetime(2027, 1, 1, tzinfo=timezone.utc),
    )
    assert result["observations"] == 5
    assert "strategy_net" in result
