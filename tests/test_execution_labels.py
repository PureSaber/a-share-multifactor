import pandas as pd
import pytest

from a_share_multifactor.config import AppConfig
from a_share_multifactor.ic_analysis import analyze_factors, analyze_ic_decay
from a_share_multifactor.label_timing import mature_labels
from a_share_multifactor.preprocess import add_forward_return, prepare_factor_panel


def bars():
    return pd.DataFrame(
        {
            "symbol": ["A"] * 4,
            "date": pd.date_range("2020-01-01", periods=4),
            "open": [100.0, 200.0, 180.0, 190.0],
            "close": [100.0, 120.0, 140.0, 160.0],
            "market_cap": [10.0] * 4,
        }
    )


def test_next_open_excludes_signal_night_and_matures_at_exit():
    panel = add_forward_return(bars().sample(frac=1, random_state=1), 1, basis="next_open")
    first = panel.iloc[0]
    assert first["forward_return_1d"] == pytest.approx(-0.1)
    assert first["forward_return_1d_label_end_at"] == pd.Timestamp("2020-01-03")
    assert not mature_labels(panel, "forward_return_1d", "2020-01-03").iloc[0]
    assert mature_labels(panel, "forward_return_1d", "2020-01-04").iloc[0]
    assert panel["forward_return_1d"].iloc[-2:].isna().all()


def test_missing_next_session_is_not_silently_skipped():
    other = bars().assign(symbol="B")
    panel = pd.concat([bars().drop(index=1), other], ignore_index=True)
    result = add_forward_return(panel, 1, basis="next_open")
    assert pd.isna(result.query("symbol == 'A'").iloc[0]["forward_return_1d"])
    assert result.query("symbol == 'B'").iloc[0]["forward_return_1d"] == pytest.approx(-0.1)


def test_production_default_and_decay_share_execution_basis():
    cfg = AppConfig(factors=["market_cap"], holding_period="fixed", forward_return_days=1)
    panel = prepare_factor_panel(cfg, bars())
    assert panel.iloc[0][cfg.forward_return_col] == pytest.approx(-0.1)
    decay = analyze_ic_decay(bars(), ["market_cap"], [1], basis="next_open")
    assert set(decay["return_basis"]) == {"next_open"}


def test_next_open_requires_prices_and_rejects_bad_horizon():
    with pytest.raises(ValueError, match="open"):
        add_forward_return(bars().drop(columns="open"), 1, basis="next_open")
    with pytest.raises(ValueError, match="positive"):
        analyze_ic_decay(bars(), ["market_cap"], [0])


def test_late_or_unknown_prices_cannot_mature_early():
    frame = bars()
    frame["available_at"] = frame["date"]
    frame.loc[1, "available_at"] = pd.Timestamp("2020-01-10")
    result = add_forward_return(frame, 1, basis="next_open")
    assert not mature_labels(result, "forward_return_1d", "2020-01-05").iloc[0]
    frame.loc[1, "available_at"] = pd.NaT
    result = add_forward_return(frame, 1, basis="next_open")
    assert pd.isna(result.iloc[0]["forward_return_1d_label_available_at"])
    result = result.drop(columns="forward_return_1d_label_available_at")
    with pytest.raises(ValueError, match="explicit availability"):
        mature_labels(result, "forward_return_1d", "2020-01-05")


def test_ic_summary_discloses_basis_and_observed_cross_section():
    panel = add_forward_return(bars(), 1, basis="next_open")
    row = analyze_factors(panel, ["market_cap"], "forward_return_1d").iloc[0]
    assert row["return_basis"] == "next_open"
    assert row["min_cross_section"] == row["median_cross_section"] == 1
    assert row["dates_without_labels"] == 2
    assert row["n_obs"] == 0  # One-name correlations are undefined, not evidence.
