"""Compare synthesis methods and plot capital curves."""

from __future__ import annotations

import argparse
import base64
import logging
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from a_share_multifactor.config import AppConfig, load_config
from a_share_multifactor.data_loader import build_dataset, load_benchmark_returns
from a_share_multifactor.ic_analysis import analyze_factors
from a_share_multifactor.preprocess import prepare_factor_panel
from a_share_multifactor.quantile_backtest import run_long_only_backtest, run_quantile_backtest
from a_share_multifactor.synthesis import synthesize
from a_share_multifactor.trade_ledger import (
    build_trade_ledger,
    capital_curve_from_returns,
    period_start_capitals,
)

logger = logging.getLogger(__name__)

SYNTHESIS_METHODS = [
    "equal_weight",
    "ic_weight",
    "rolling_ic_weight",
    "ridge",
    "ols",
]

METHOD_LABELS = {
    "equal_weight": "等权 Equal Weight",
    "ic_weight": "IC加权 IC Weight",
    "rolling_ic_weight": "滚动IC Rolling IC",
    "ridge": "Ridge回归",
    "ols": "OLS回归",
}


def _load_symbol_names(symbols: list[str]) -> dict[str, str]:
    try:
        import akshare as ak

        idx = ak.index_stock_cons(symbol="000300")
        idx["symbol"] = idx["品种代码"].astype(str).str.zfill(6)
        mapping = idx.drop_duplicates("symbol").set_index("symbol")["品种名称"].to_dict()
        return {symbol: str(mapping.get(symbol, symbol)) for symbol in symbols}
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to load stock names: %s", exc)
        return {symbol: symbol for symbol in symbols}


def _stats_from_returns(
    period_returns: pd.Series,
    initial_capital: float,
    periods_per_year: int = 12,
) -> dict[str, float]:
    """Compute summary stats for a return series."""
    clean = period_returns.dropna()
    if clean.empty:
        return {
            "final_capital": initial_capital,
            "total_return_pct": 0.0,
            "ann_return": float("nan"),
            "ann_vol": float("nan"),
            "sharpe": float("nan"),
            "rebalance_periods": 0,
        }
    capital = capital_curve_from_returns(clean, initial_capital)
    mean_ret = float(clean.mean())
    vol = float(clean.std(ddof=0))
    ann_return = (1 + mean_ret) ** periods_per_year - 1
    ann_vol = vol * np.sqrt(periods_per_year) if vol > 0 else float("nan")
    sharpe = ann_return / ann_vol if ann_vol and not np.isnan(ann_vol) else float("nan")
    return {
        "final_capital": float(capital.iloc[-1]),
        "total_return_pct": float((capital.iloc[-1] / initial_capital - 1) * 100),
        "ann_return": ann_return,
        "ann_vol": ann_vol,
        "sharpe": sharpe,
        "rebalance_periods": len(clean),
    }


def run_synthesis_comparison(
    config: AppConfig,
    data_dir: Path | None = None,
    initial_capital: float = 1_000_000,
    eval_start_date: str | None = None,
    long_only: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, pd.DataFrame], pd.DataFrame]:
    """Run all synthesis methods and return capital curves, stats, ledgers, summary."""
    trial_config = config
    if config.costs.retail_mode:
        trial_config = replace(
            config,
            costs=replace(config.costs, initial_capital=initial_capital),
        )

    panel = prepare_factor_panel(trial_config, build_dataset(trial_config, data_dir=data_dir))
    factor_cols = [col for col in config.factors if col in panel.columns]

    eval_start = pd.Timestamp(eval_start_date) if eval_start_date else None
    ic_panel = panel
    if eval_start is not None:
        ic_panel = panel[panel["date"] >= eval_start]
    ic_report = analyze_factors(ic_panel, factor_cols, config.forward_return_col)

    benchmark = load_benchmark_returns(trial_config, data_dir=data_dir)
    name_map = _load_symbol_names(sorted(panel["symbol"].astype(str).unique()))

    capital_frames: list[pd.DataFrame] = []
    stats_rows: list[dict[str, object]] = []
    ledgers: dict[str, pd.DataFrame] = {}
    periods_per_year = 252 if trial_config.costs.trade_freq == "daily" else 12

    for method in SYNTHESIS_METHODS:
        logger.info("Running synthesis method: %s", method)
        trial = replace(trial_config, synthesis=replace(trial_config.synthesis, method=method))
        scored = synthesize(panel, trial, ic_summary=ic_report, min_score_date=eval_start)
        if long_only:
            results = run_long_only_backtest(
                scored,
                trial,
                benchmark_returns=benchmark,
                trade_start_date=eval_start,
            )
        else:
            results = run_quantile_backtest(scored, trial, benchmark_returns=benchmark)

        if results.long_short.empty:
            logger.warning("No portfolio returns for method=%s", method)
            continue

        portfolio_returns = results.long_short
        if eval_start is not None:
            portfolio_returns = portfolio_returns.loc[portfolio_returns.index >= eval_start]

        if portfolio_returns.empty:
            logger.warning("No returns on/after eval_start for method=%s", method)
            continue

        capital = capital_curve_from_returns(portfolio_returns, initial_capital)
        starts = period_start_capitals(portfolio_returns, initial_capital)
        ledger = build_trade_ledger(
            scored,
            trial,
            starts,
            name_map=name_map,
            long_only=long_only,
        )
        if eval_start is not None and not ledger.empty:
            ledger = ledger[ledger["open_date"] >= eval_start].reset_index(drop=True)
        ledgers[method] = ledger

        frame = capital.reset_index()
        frame.columns = ["date", method]
        capital_frames.append(frame.set_index("date"))

        metrics = _stats_from_returns(
            portfolio_returns,
            initial_capital,
            periods_per_year=periods_per_year,
        )
        stats_rows.append(
            {
                "method": method,
                "label": METHOD_LABELS.get(method, method),
                **metrics,
                "trade_pairs": len(ledger),
            }
        )

    if not capital_frames:
        empty = pd.DataFrame()
        return empty, empty, ledgers, pd.DataFrame(stats_rows)

    capital_curves = pd.concat(capital_frames, axis=1).sort_index()
    summary = pd.DataFrame(stats_rows).sort_values("final_capital", ascending=False)
    return capital_curves, ic_report, ledgers, summary


def plot_capital_comparison(
    capital_curves: pd.DataFrame,
    ledgers: dict[str, pd.DataFrame],
    initial_capital: float,
    output_path: Path,
    title_suffix: str = "",
    long_only: bool = False,
) -> None:
    """Save capital curve chart with rebalance trade markers."""
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt

    plt.rcParams["font.sans-serif"] = [
        "Microsoft YaHei",
        "SimHei",
        "Arial Unicode MS",
        "DejaVu Sans",
    ]
    plt.rcParams["axes.unicode_minus"] = False

    colors = {
        "equal_weight": "#1f77b4",
        "ic_weight": "#ff7f0e",
        "rolling_ic_weight": "#2ca02c",
        "ridge": "#d62728",
        "ols": "#9467bd",
    }

    fig, ax = plt.subplots(figsize=(14, 7))
    ax.axhline(initial_capital, color="#999", linestyle="--", linewidth=1, label="初始资金")

    for method in capital_curves.columns:
        series = capital_curves[method].dropna()
        label = METHOD_LABELS.get(method, method)
        ax.plot(
            series.index,
            series.values,
            label=label,
            color=colors.get(method, None),
            linewidth=2.2,
        )

        ledger = ledgers.get(method)
        if ledger is None or ledger.empty:
            continue

        open_dates = ledger["open_date"].drop_duplicates().sort_values()
        marker_capital = series.reindex(open_dates, method="ffill")
        ax.scatter(
            marker_capital.index,
            marker_capital.values,
            color=colors.get(method, "#333"),
            s=18,
            alpha=0.45,
            zorder=3,
        )

    if long_only:
        title = f"五种因子合成模型 — Q5 纯多头资金曲线（初始 {initial_capital:,.0f} 元）"
    else:
        title = "五种因子合成模型 — 多空组合资金曲线（初始 100 万）"
    if title_suffix:
        title += f"\n{title_suffix}"
    ax.set_title(title, fontsize=14)
    ax.set_xlabel("日期")
    ax.set_ylabel("资金（元）")
    if initial_capital >= 10000:
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _p: f"{x / 10000:.1f}万"))
    else:
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _p: f"{x:,.0f}"))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.legend(loc="upper left")
    ax.grid(alpha=0.3)
    fig.autofmt_xdate()
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_trade_events(
    capital_curves: pd.DataFrame,
    ledger: pd.DataFrame,
    method: str,
    initial_capital: float,
    output_path: Path,
    max_labels: int = 12,
    long_only: bool = False,
) -> None:
    """Plot one method with annotated sample buy/sell events."""
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt

    plt.rcParams["font.sans-serif"] = [
        "Microsoft YaHei",
        "SimHei",
        "Arial Unicode MS",
        "DejaVu Sans",
    ]
    plt.rcParams["axes.unicode_minus"] = False

    if method not in capital_curves.columns or ledger.empty:
        return

    series = capital_curves[method].dropna()
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.plot(series.index, series.values, color="#1f77b4", linewidth=2.2, label="资金曲线")
    ax.axhline(initial_capital, color="#999", linestyle="--", linewidth=1)

    longs = ledger[ledger["side"] == "long"]
    shorts = ledger[ledger["side"] == "short"]
    open_dates = ledger["open_date"].drop_duplicates().sort_values()
    marker_capital = series.reindex(open_dates, method="ffill")

    ax.scatter(
        marker_capital.index,
        marker_capital.values,
        c="#2ca02c",
        s=30,
        label="调仓开仓",
        zorder=4,
    )

    sample = longs.sort_values("open_date").head(max_labels // 2)
    for _, row in sample.iterrows():
        open_date = row["open_date"]
        cap = float(series.reindex([open_date], method="ffill").iloc[0])
        ax.annotate(
            f"买 {row['name']}\nQ{row['quantile']}",
            xy=(open_date, cap),
            xytext=(8, 12),
            textcoords="offset points",
            fontsize=7,
            color="#1a7f37",
            arrowprops={"arrowstyle": "->", "color": "#1a7f37", "lw": 0.8},
        )

    sample_short = shorts.sort_values("open_date").head(max_labels // 2)
    for _, row in sample_short.iterrows():
        open_date = row["open_date"]
        cap = float(series.reindex([open_date], method="ffill").iloc[0])
        ax.annotate(
            f"空 {row['name']}\nQ{row['quantile']}",
            xy=(open_date, cap),
            xytext=(8, -18),
            textcoords="offset points",
            fontsize=7,
            color="#b42318",
            arrowprops={"arrowstyle": "->", "color": "#b42318", "lw": 0.8},
        )

    title = METHOD_LABELS.get(method, method)
    if long_only:
        ax.set_title(f"{title} — 调仓买卖点示例（Q5 纯多头）", fontsize=13)
    else:
        ax.set_title(f"{title} — 调仓买卖点示例（Q5 做多 / Q1 做空）", fontsize=13)
    ax.set_xlabel("日期")
    ax.set_ylabel("资金（元）")
    if initial_capital >= 10000:
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _p: f"{x / 10000:.1f}万"))
    else:
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _p: f"{x:,.0f}"))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.legend(loc="upper left")
    ax.grid(alpha=0.3)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def write_comparison_html(
    output_path: Path,
    summary: pd.DataFrame,
    capital_chart: Path,
    trade_chart: Path | None,
    initial_capital: float,
    eval_period: str = "",
    long_only: bool = False,
    retail_mode: bool = False,
    max_holdings: int = 0,
    partial_rebalance: bool = True,
    trade_freq: str = "monthly",
    min_holding_days: int = 0,
) -> None:
    """Write HTML report embedding charts and summary table."""
    capital_b64 = base64.b64encode(capital_chart.read_bytes()).decode("ascii")
    trade_section = ""
    if trade_chart is not None and trade_chart.exists():
        trade_b64 = base64.b64encode(trade_chart.read_bytes()).decode("ascii")
        trade_section = (
            f"<h2>调仓买卖点示例（等权模型）</h2>"
            f"<img src='data:image/png;base64,{trade_b64}' style='max-width:100%;'/>"
        )

    summary_html = summary.to_html(index=False, float_format=lambda x: f"{x:,.2f}")
    strategy = "Q5 纯多头（全仓等权）" if long_only else "Q5 做多 − Q1 做空"
    rebalance_label = "日度调仓" if trade_freq == "daily" else "月度调仓"
    daily_rules = ""
    if retail_mode and trade_freq == "daily" and min_holding_days > 0:
        daily_rules = (
            f"、最少持有<strong>{min_holding_days}</strong>交易日"
            "、大涨提前止盈（单日≥8% / 累计≥25% / 连涨3日≥3%）"
        )
    retail_note = (
        " | 散户模拟：<strong>100股一手、佣金最低5元、卖出印花税0.05%</strong>"
        f"{f'、最多持仓<strong>{max_holdings}</strong>只' if max_holdings else ''}"
        f"{'、增量调仓' if partial_rebalance else ''}"
        f"{daily_rules}"
        if retail_mode
        else ""
    )
    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>合成模型对比</title>
<style>
body {{ font-family: "Microsoft YaHei", sans-serif; margin: 2rem; }}
table {{ border-collapse: collapse; margin: 1rem 0; }}
td, th {{ border: 1px solid #ddd; padding: 8px; }}
h1, h2 {{ color: #222; }}
</style></head><body>
<h1>五种因子合成模型对比</h1>
<p>初始资金：<strong>{initial_capital:,.0f}</strong> 元 |
策略：<strong>{strategy}</strong>（{rebalance_label}，含交易成本）
{f" | 回测区间：<strong>{eval_period}</strong>" if eval_period else ""}{retail_note}</p>
<h2>绩效汇总</h2>
{summary_html}
<h2>资金曲线对比</h2>
<img src="data:image/png;base64,{capital_b64}" style="max-width:100%;"/>
{trade_section}
<h2>交易明细</h2>
<p>每个模型的完整开平仓记录见同目录下 <code>trade_ledger_*.csv</code>，
包含股票名称、分层、买卖方向、成交量、持仓天数与收益率。</p>
</body></html>"""
    output_path.write_text(html, encoding="utf-8")


def write_outputs(
    output_dir: Path,
    capital_curves: pd.DataFrame,
    summary: pd.DataFrame,
    ledgers: dict[str, pd.DataFrame],
    initial_capital: float,
    eval_period: str = "",
    long_only: bool = False,
    retail_mode: bool = False,
    max_holdings: int = 0,
    partial_rebalance: bool = True,
    trade_freq: str = "monthly",
    min_holding_days: int = 0,
) -> None:
    """Persist CSV, charts, and HTML report."""
    output_dir.mkdir(parents=True, exist_ok=True)
    capital_curves.reset_index().to_csv(output_dir / "capital_curves.csv", index=False)
    summary.to_csv(output_dir / "synthesis_comparison_summary.csv", index=False)

    for method, ledger in ledgers.items():
        ledger.to_csv(output_dir / f"trade_ledger_{method}.csv", index=False)

    capital_chart = output_dir / "capital_comparison.png"
    plot_capital_comparison(
        capital_curves,
        ledgers,
        initial_capital,
        capital_chart,
        title_suffix=eval_period,
        long_only=long_only,
    )

    trade_chart = output_dir / "trade_events_equal_weight.png"
    if "equal_weight" in ledgers:
        plot_trade_events(
            capital_curves,
            ledgers["equal_weight"],
            "equal_weight",
            initial_capital,
            trade_chart,
            long_only=long_only,
        )

    write_comparison_html(
        output_dir / "synthesis_comparison.html",
        summary,
        capital_chart,
        trade_chart if trade_chart.exists() else None,
        initial_capital,
        eval_period=eval_period,
        long_only=long_only,
        retail_mode=retail_mode,
        max_holdings=max_holdings,
        partial_rebalance=partial_rebalance,
        trade_freq=trade_freq,
        min_holding_days=min_holding_days,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare synthesis methods with capital curves")
    parser.add_argument("--config", type=Path, default=Path("configs/run_report.yaml"))
    parser.add_argument("--data-dir", type=Path, default=Path("./data"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/synthesis_compare"))
    parser.add_argument("--initial-capital", type=float, default=1_000_000)
    parser.add_argument(
        "--eval-start-date",
        type=str,
        default=None,
        help="Only accumulate capital from this date (YYYY-MM-DD); still uses earlier data for rolling models",
    )
    parser.add_argument(
        "--long-only",
        action="store_true",
        help="Long-only Q5 portfolio (no short leg)",
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(message)s",
    )

    config = load_config(args.config)
    logger.info("Comparing synthesis methods on %s symbols", config.factors)

    capital_curves, ic_report, ledgers, summary = run_synthesis_comparison(
        config,
        data_dir=args.data_dir,
        initial_capital=args.initial_capital,
        eval_start_date=args.eval_start_date,
        long_only=args.long_only,
    )

    if capital_curves.empty:
        raise RuntimeError("No capital curves produced — check data and config")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    ic_report.to_csv(args.output_dir / "ic_summary.csv", index=False)

    eval_period = ""
    if args.eval_start_date:
        end = config.end_date
        eval_period = f"{args.eval_start_date} ~ {end}"

    write_outputs(
        args.output_dir,
        capital_curves,
        summary,
        ledgers,
        args.initial_capital,
        eval_period=eval_period,
        long_only=args.long_only,
        retail_mode=config.costs.retail_mode,
        max_holdings=config.costs.max_holdings,
        partial_rebalance=config.costs.partial_rebalance,
        trade_freq=config.costs.trade_freq,
        min_holding_days=config.costs.min_holding_days,
    )
    logger.info("Outputs written to %s", args.output_dir)
    print("\nSynthesis Comparison Summary:")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
