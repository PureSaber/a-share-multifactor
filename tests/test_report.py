from pathlib import Path

import pandas as pd

from a_share_multifactor.quantile_backtest import BacktestResult
from a_share_multifactor.report import write_html_report


def test_write_html_report(tmp_path: Path) -> None:
    results = BacktestResult(
        quantile_returns=pd.DataFrame({"Q1": [0.01]}, index=pd.to_datetime(["2020-01-31"])),
        cumulative_returns=pd.DataFrame({"Q1": [1.01]}, index=pd.to_datetime(["2020-01-31"])),
        long_short=pd.Series([0.01], index=pd.to_datetime(["2020-01-31"])),
        stats=pd.DataFrame({"portfolio": ["Q1"], "mean_return": [0.01]}),
    )
    ic_report = pd.DataFrame({"factor": ["f1"], "mean_ic": [0.05]})
    output = tmp_path / "report.html"
    write_html_report(results, ic_report, output)
    assert output.exists()
    assert "IC Summary" in output.read_text(encoding="utf-8")
