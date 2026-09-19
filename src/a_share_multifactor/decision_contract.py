"""Versioned decision card: one payload for humans and downstream applications."""

from __future__ import annotations

import html
import json
import math
from pathlib import Path

import pandas as pd

SCHEMA_VERSION = "quant.decision/v1"


def clean_json(value):
    if isinstance(value, dict):
        return {str(k): clean_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean_json(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def validate_decision(card: dict) -> None:
    required = {
        "schema_version",
        "run_id",
        "status",
        "as_of",
        "generated_at",
        "valid_until",
        "scope",
        "data_quality",
        "validation",
        "current_positions",
        "targets",
        "proposed_trades",
        "estimated_cost",
        "risk",
        "evidence",
        "reasons",
    }
    if missing := required - card.keys():
        raise ValueError(f"Missing decision fields: {sorted(missing)}")
    if card["schema_version"] != SCHEMA_VERSION or card["status"] not in {
        "blocked",
        "observe",
        "paper_ready",
    }:
        raise ValueError("Invalid decision schema/status")
    if card["scope"] != "paper_simulation_only":
        raise ValueError("This profile only authorizes paper simulation")
    if card["status"] != "paper_ready" and (card["targets"] or card["proposed_trades"]):
        raise ValueError("Blocked/observe decisions cannot contain actionable targets")
    if card["status"] == "paper_ready":
        if not card["data_quality"].get("passed") or not card["validation"].get("passed"):
            raise ValueError("Paper targets require data and research validation")
        if not card["evidence"].get("standard_manifest"):
            raise ValueError("Paper targets require a reconciled standard ledger")
    json.dumps(clean_json(card), allow_nan=False)


def write_decision(card: dict, output: Path) -> None:
    card = clean_json(card)
    validate_decision(card)
    output.mkdir(parents=True, exist_ok=True)
    (output / "decision.json").write_text(
        json.dumps(card, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8"
    )
    labels = {
        "blocked": "数据或执行检查未通过",
        "observe": "仅观察，暂不生成调仓建议",
        "paper_ready": "可进行模拟调仓，尚未验证真实投资效果",
    }

    def table(key):
        frame = pd.DataFrame(card[key])
        return frame.to_html(index=False, escape=True, border=0) if len(frame) else "<p>无</p>"

    def detail(value):
        return html.escape(json.dumps(value, ensure_ascii=False, indent=2))

    content = f"""<!doctype html><html lang="zh-CN"><meta charset="utf-8">
<title>投研决策卡 · {html.escape(card["as_of"])}</title>
<style>body{{font:16px/1.65 system-ui;margin:36px auto;max-width:1050px;padding:0 20px;color:#172638}}
h1{{font-size:30px}}.status{{padding:18px;background:#edf3f7;border-left:5px solid #337587}}
table{{border-collapse:collapse;width:100%;font-size:14px}}td,th{{padding:9px;text-align:left;border-bottom:1px solid #dbe2e8}}
pre{{white-space:pre-wrap;background:#f6f8fa;padding:14px;font-size:13px}}small{{color:#556}}
</style><h1>A 股日／周频研究决策卡</h1>
<p class="status">{labels[card["status"]]}</p>
<p>行情截至：{html.escape(card["as_of"])}　生成时间：{html.escape(card["generated_at"])}<br>
有效期至：{html.escape(str(card["valid_until"]))}（下一交易日开盘前须重新检查）</p>
<p>账户为虚拟模拟账户。观察名单仅用于验证流程，不代表股票推荐。成交采用下一交易日开盘模拟，
使用未复权价格、配置成本和成交量上限；无实盘下单。</p>
<h2>状态原因</h2><pre>{detail(card["reasons"])}</pre>
<h2>当前模拟持仓</h2>{table("current_positions")}
<h2>目标持仓</h2>{table("targets")}<h2>拟模拟调仓</h2>{table("proposed_trades")}
<h2>预计成本（元）</h2><pre>{detail(card["estimated_cost"])}</pre>
<h2>风险与约束</h2><pre>{detail(card["risk"])}</pre>
<h2>研究验证与基准</h2><pre>{detail(card["validation"])}</pre>
<h2>数据质量</h2><pre>{detail(card["data_quality"])}</pre>
<details><summary>可重放证据与版本</summary><pre>{detail(card["evidence"])}</pre></details>
<small>所有页面内容由同一份 decision.json 生成；历史观察名单研究不能证明未来超额收益。</small></html>"""
    (output / "decision.html").write_text(content, encoding="utf-8")
