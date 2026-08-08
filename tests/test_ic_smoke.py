from pathlib import Path

import pandas as pd
import yaml

from a_share_multifactor.factors import compute_factors
from a_share_multifactor.ic_smoke import (
    build_smoke_panel,
    load_ic_smoke_config,
    run_ic_smoke,
)


def test_ic_smoke_config_exists() -> None:
    cfg_path = Path("configs/ic_smoke.yaml")
    assert cfg_path.exists()
    cfg = load_ic_smoke_config(cfg_path)
    assert "momentum_20d" in cfg["factors"]
    assert cfg["outputs_dir"] == "outputs/ic_smoke"


def test_quant_factors_columns_on_smoke_panel() -> None:
    cfg = load_ic_smoke_config("configs/ic_smoke.yaml")
    panel = build_smoke_panel(["A", "B"], periods=40)
    result = compute_factors(panel, factor_names=cfg["factors"])
    for col in cfg["factors"]:
        assert col in result.columns
        assert result[col].notna().any()


def test_run_ic_smoke_writes_outputs(tmp_path: Path) -> None:
    out = run_ic_smoke("configs/ic_smoke.yaml", output_dir=tmp_path)
    assert (out / "ic_summary.csv").exists()
    assert (out / "manifest.yaml").exists()
    summary = pd.read_csv(out / "ic_summary.csv")
    cfg = load_ic_smoke_config("configs/ic_smoke.yaml")
    assert set(summary["factor"]) == set(cfg["factors"])
    manifest = yaml.safe_load((out / "manifest.yaml").read_text(encoding="utf-8"))
    assert manifest["return_col"] == "forward_return_20d"
