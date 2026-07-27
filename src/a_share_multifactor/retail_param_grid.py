"""Grid search retail OLS parameters and plot return surfaces."""

from __future__ import annotations

import argparse
import itertools
import logging
from dataclasses import replace
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from a_share_multifactor.calendar import trade_schedule_dates
from a_share_multifactor.config import AppConfig, load_config
from a_share_multifactor.data_loader import build_dataset
from a_share_multifactor.preprocess import prepare_factor_panel
from a_share_multifactor.synthesis import synthesize
from a_share_multifactor.synthesis_compare import capital_curve_from_returns
from a_share_multifactor.trading_costs import simulate_daily_retail_portfolio

logger = logging.getLogger(__name__)

EARLY_EXIT_PRESETS: dict[str, dict[str, float | int | bool]] = {
    "off": {"early_exit_enabled": False},
    "standard": {
        "early_exit_enabled": True,
        "early_exit_single_day_return": 0.08,
        "early_exit_cumulative_return": 0.25,
        "early_exit_consecutive_days": 3,
        "early_exit_consecutive_daily": 0.03,
    },
    "loose": {
        "early_exit_enabled": True,
        "early_exit_single_day_return": 0.12,
        "early_exit_cumulative_return": 0.35,
        "early_exit_consecutive_days": 3,
        "early_exit_consecutive_daily": 0.04,
    },
}

MIN_HOLDING_DAYS_GRID = [10, 20, 40]
TRADE_FREQ_GRID = ["daily", "weekly", "monthly"]
RANK_THRESHOLD_GRID = [0, 3, 5]


def _combo_label(
    early_exit: str,
    min_holding_days: int,
    trade_freq: str,
    rank_change_threshold: int,
) -> str:
    return (
        f"exit={early_exit}|hold={min_holding_days}|freq={trade_freq}|rank={rank_change_threshold}"
    )


def build_param_grid() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for early_exit, min_hold, trade_freq, rank_th in itertools.product(
        EARLY_EXIT_PRESETS,
        MIN_HOLDING_DAYS_GRID,
        TRADE_FREQ_GRID,
        RANK_THRESHOLD_GRID,
    ):
        rows.append(
            {
                "early_exit": early_exit,
                "min_holding_days": min_hold,
                "trade_freq": trade_freq,
                "rank_change_threshold": rank_th,
                "label": _combo_label(early_exit, min_hold, trade_freq, rank_th),
            }
        )
    return rows


def _load_or_build_scored_panel(
    panel: pd.DataFrame,
    config: AppConfig,
    trade_freq: str,
    cache_dir: Path,
    min_score_date: pd.Timestamp | None,
) -> pd.DataFrame:
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"ols_scored_{trade_freq}.parquet"
    if cache_path.exists():
        logger.info("Loading cached OLS scores: %s", cache_path)
        return pd.read_parquet(cache_path)

    logger.info("Synthesizing OLS scores for trade_freq=%s", trade_freq)
    trial = replace(
        config,
        synthesis=replace(config.synthesis, method="ols"),
        costs=replace(config.costs, trade_freq=trade_freq),
    )
    scored = synthesize(panel, trial, min_score_date=min_score_date)
    scored.to_parquet(cache_path, index=False)
    return scored


def _stats_from_returns(
    returns: pd.Series,
    initial_capital: float,
    trade_freq: str,
) -> dict[str, float | int]:
    clean = returns.dropna()
    if clean.empty:
        return {
            "final_capital": initial_capital,
            "total_return_pct": 0.0,
            "ann_return": float("nan"),
            "sharpe": float("nan"),
            "trade_periods": 0,
        }

    periods_per_year = {"daily": 252, "weekly": 52, "monthly": 12}[trade_freq]
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
        "sharpe": sharpe,
        "trade_periods": int(len(clean)),
    }


def run_retail_param_grid(
    config: AppConfig,
    data_dir: Path | None = None,
    eval_start_date: str = "2025-01-01",
    initial_capital: float = 10_000.0,
    cache_dir: Path | None = None,
) -> pd.DataFrame:
    """Run all parameter combinations on cached OLS scored panels."""
    eval_start = pd.Timestamp(eval_start_date)
    panel = prepare_factor_panel(config, build_dataset(config, data_dir=data_dir))
    cache_root = cache_dir or Path(config.outputs_dir) / "retail_param_grid_cache"
    scored_by_freq = {
        trade_freq: _load_or_build_scored_panel(panel, config, trade_freq, cache_root, eval_start)
        for trade_freq in TRADE_FREQ_GRID
    }

    rows: list[dict[str, object]] = []
    grid = build_param_grid()
    for idx, combo in enumerate(grid, start=1):
        trade_freq = str(combo["trade_freq"])
        early_exit = str(combo["early_exit"])
        min_hold = int(combo["min_holding_days"])
        rank_th = int(combo["rank_change_threshold"])
        label = str(combo["label"])

        costs_kwargs: dict[str, object] = {
            "min_holding_days": min_hold,
            "trade_freq": trade_freq,
            "rank_change_threshold": rank_th,
            "initial_capital": initial_capital,
        }
        costs_kwargs.update(EARLY_EXIT_PRESETS[early_exit])
        trial_costs = replace(config.costs, **costs_kwargs)
        trial = replace(config, costs=trial_costs)

        scored = scored_by_freq[trade_freq]
        trade_idx = trade_schedule_dates(
            scored["date"],
            trial.rebalance_freq,
            trial.costs.retail_mode,
            trade_freq,
        )
        trade_idx = trade_idx[trade_idx >= eval_start]

        returns = simulate_daily_retail_portfolio(
            scored,
            trial,
            "composite_score",
            trial.quantiles,
            trade_idx,
        )
        metrics = _stats_from_returns(returns, initial_capital, trade_freq)
        rows.append(
            {
                "combo_id": idx,
                "label": label,
                "early_exit": early_exit,
                "min_holding_days": min_hold,
                "trade_freq": trade_freq,
                "rank_change_threshold": rank_th,
                **metrics,
            }
        )
        logger.info(
            "[%s/%s] %s -> %.1f%%",
            idx,
            len(grid),
            label,
            metrics["total_return_pct"],
        )

    summary = pd.DataFrame(rows).sort_values("total_return_pct", ascending=False)
    return summary


def plot_param_grid_results(summary: pd.DataFrame, output_dir: Path) -> None:
    """Save heatmaps and bar charts for parameter grid results."""
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams["font.sans-serif"] = [
        "Microsoft YaHei",
        "SimHei",
        "Arial Unicode MS",
        "DejaVu Sans",
    ]
    plt.rcParams["axes.unicode_minus"] = False

    summary = summary.copy()
    summary.to_csv(output_dir / "param_grid_results.csv", index=False)

    # Top combinations bar chart
    top_n = min(20, len(summary))
    top = summary.head(top_n)
    fig, ax = plt.subplots(figsize=(14, 8))
    colors = plt.cm.viridis(np.linspace(0.2, 0.9, top_n))
    ax.barh(top["label"][::-1], top["total_return_pct"][::-1], color=colors[::-1])
    ax.set_xlabel("总收益率 (%)")
    ax.set_title(f"OLS 散户参数网格 Top {top_n}（2025 至今）")
    ax.axvline(0, color="#999", linewidth=1)
    fig.tight_layout()
    fig.savefig(output_dir / "param_grid_top20.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Heatmaps: min_holding vs rank_threshold, faceted by trade_freq x early_exit
    exit_modes = list(EARLY_EXIT_PRESETS)
    freqs = TRADE_FREQ_GRID
    fig, axes = plt.subplots(
        len(freqs),
        len(exit_modes),
        figsize=(5 * len(exit_modes), 4.5 * len(freqs)),
        sharex=True,
        sharey=True,
    )
    if len(freqs) == 1 and len(exit_modes) == 1:
        axes = np.array([[axes]])

    vmin = summary["total_return_pct"].min()
    vmax = summary["total_return_pct"].max()

    for i, trade_freq in enumerate(freqs):
        for j, early_exit in enumerate(exit_modes):
            ax = axes[i, j]
            subset = summary[
                (summary["trade_freq"] == trade_freq) & (summary["early_exit"] == early_exit)
            ]
            pivot = subset.pivot_table(
                index="min_holding_days",
                columns="rank_change_threshold",
                values="total_return_pct",
                aggfunc="mean",
            )
            pivot = pivot.reindex(index=MIN_HOLDING_DAYS_GRID, columns=RANK_THRESHOLD_GRID)
            im = ax.imshow(pivot.values, cmap="RdYlGn", vmin=vmin, vmax=vmax, aspect="auto")
            ax.set_xticks(range(len(RANK_THRESHOLD_GRID)))
            ax.set_xticklabels([str(v) for v in RANK_THRESHOLD_GRID])
            ax.set_yticks(range(len(MIN_HOLDING_DAYS_GRID)))
            ax.set_yticklabels([str(v) for v in MIN_HOLDING_DAYS_GRID])
            ax.set_xlabel("排名缓冲阈值")
            ax.set_ylabel("最少持有(交易日)")
            ax.set_title(f"{trade_freq} / {early_exit}")
            for y in range(pivot.shape[0]):
                for x in range(pivot.shape[1]):
                    val = pivot.values[y, x]
                    if not np.isnan(val):
                        ax.text(
                            x,
                            y,
                            f"{val:.0f}%",
                            ha="center",
                            va="center",
                            fontsize=9,
                            color="black" if abs(val) < (vmax - vmin) * 0.35 else "white",
                        )

    fig.colorbar(im, ax=axes.ravel().tolist(), label="总收益率 (%)", shrink=0.85)
    fig.suptitle("OLS 散户参数网格：收益率热力图", fontsize=14, y=1.01)
    fig.tight_layout()
    fig.savefig(output_dir / "param_grid_heatmaps.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Grouped comparison by single dimension (marginal means)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    dims = [
        ("early_exit", "止盈模式"),
        ("min_holding_days", "最少持有(日)"),
        ("trade_freq", "调仓频率"),
        ("rank_change_threshold", "排名缓冲"),
    ]
    for ax, (col, title) in zip(axes.ravel(), dims):
        grouped = summary.groupby(col)["total_return_pct"].mean().sort_values(ascending=False)
        ax.bar(grouped.index.astype(str), grouped.values, color="#4C78A8")
        ax.set_title(f"平均收益 by {title}")
        ax.set_ylabel("总收益率 (%)")
        ax.axhline(0, color="#999", linewidth=1)
        ax.tick_params(axis="x", rotation=20)
    fig.suptitle("单参数边际平均收益（OLS 网格）", fontsize=14)
    fig.tight_layout()
    fig.savefig(output_dir / "param_grid_marginal.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Grid search retail OLS trading parameters")
    parser.add_argument("--config", type=Path, default=Path("configs/run_retail_daily_10k.yaml"))
    parser.add_argument("--data-dir", type=Path, default=Path("./data"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/retail_param_grid"))
    parser.add_argument("--eval-start-date", type=str, default="2025-01-01")
    parser.add_argument("--initial-capital", type=float, default=10_000.0)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(message)s",
    )

    config = load_config(args.config)
    config = replace(config, costs=replace(config.costs, retail_mode=True))

    summary = run_retail_param_grid(
        config,
        data_dir=args.data_dir,
        eval_start_date=args.eval_start_date,
        initial_capital=args.initial_capital,
        cache_dir=args.output_dir / "cache",
    )
    plot_param_grid_results(summary, args.output_dir)
    logger.info("Saved results to %s", args.output_dir)
    print("\nTop 10 combinations:")
    print(
        summary[
            [
                "label",
                "total_return_pct",
                "ann_return",
                "sharpe",
                "trade_periods",
            ]
        ]
        .head(10)
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
