"""Export ranked return vs max drawdown table for param grid."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

root = Path("outputs/retail_param_grid")
df = pd.read_csv(root / "param_grid_results_with_mdd.csv")
full = pd.read_csv(root / "param_grid_results.csv")
df = df.merge(
    full[
        ["label", "early_exit", "min_holding_days", "trade_freq", "rank_change_threshold", "sharpe"]
    ],
    on="label",
    how="left",
)
df["calmar"] = df["total_return_pct"] / df["max_drawdown_pct"].abs()
df = df.sort_values("total_return_pct", ascending=False).reset_index(drop=True)
df["rank"] = df.index + 1

export = df[
    [
        "rank",
        "early_exit",
        "min_holding_days",
        "trade_freq",
        "rank_change_threshold",
        "total_return_pct",
        "max_drawdown_pct",
        "sharpe",
        "calmar",
    ]
].copy()
export.columns = [
    "排名",
    "止盈",
    "持有天数",
    "调仓频率",
    "排名缓冲",
    "总收益%",
    "最大回撤%",
    "Sharpe",
    "收益/回撤",
]
export["总收益%"] = export["总收益%"].round(1)
export["最大回撤%"] = export["最大回撤%"].round(1)
export["Sharpe"] = export["Sharpe"].round(2)
export["收益/回撤"] = export["收益/回撤"].round(2)
export.to_csv(root / "param_grid_return_mdd_ranked.csv", index=False, encoding="utf-8-sig")

fig, ax = plt.subplots(figsize=(12, 8))
colors = {"off": "#2ca02c", "standard": "#1f77b4", "loose": "#ff7f0e"}
for mode, group in df.groupby("early_exit"):
    ax.scatter(
        group["max_drawdown_pct"],
        group["total_return_pct"],
        label=mode,
        c=colors.get(mode, "gray"),
        alpha=0.8,
        s=60,
    )
for _, row in df.head(5).iterrows():
    ax.annotate(
        f"hold={int(row['min_holding_days'])}/{row['trade_freq']}",
        (row["max_drawdown_pct"], row["total_return_pct"]),
        fontsize=8,
        xytext=(5, 5),
        textcoords="offset points",
    )
ax.axhline(0, color="#999", lw=0.8)
ax.axvline(0, color="#999", lw=0.8)
ax.set_xlabel("最大回撤 (%)")
ax.set_ylabel("总收益 (%)")
ax.set_title("OLS 参数网格 81 组：收益 vs 最大回撤 (2025至今)")
ax.legend(title="止盈模式")
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(root / "param_grid_return_vs_mdd.png", dpi=150)
print(export.to_string(index=False))
