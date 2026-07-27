from pathlib import Path

from a_share_multifactor.config import AppConfig, load_config


def test_load_config_defaults(tmp_path: Path) -> None:
    config_file = tmp_path / "config.yaml"
    config_file.write_text("universe: hs300\n", encoding="utf-8")

    config = load_config(config_file)

    assert config.universe == "hs300"
    assert config.start_date == "2020-01-01"
    assert config.quantiles == 5
    assert config.preprocess.winsorize == (0.01, 0.99)
    assert config.synthesis.method == "equal_weight"
    assert config.forward_return_days == 20
    assert config.holding_period == "rebalance"
    assert config.factor_directions["market_cap"] == -1
    assert config.filters.use_historical_universe is True


def test_load_config_full(tmp_path: Path) -> None:
    config_file = tmp_path / "config.yaml"
    config_file.write_text(
        """
universe: hs300
start_date: "2021-01-01"
end_date: "2023-12-31"
factors:
  - momentum_20d
quantiles: 10
forward_return_days: 5
preprocess:
  winsorize: [0.05, 0.95]
  standardize: zscore
synthesis:
  method: ic_weight
data:
  price: custom/prices.parquet
""",
        encoding="utf-8",
    )

    config = load_config(config_file)

    assert config.start_date == "2021-01-01"
    assert config.factors == ["momentum_20d"]
    assert config.quantiles == 10
    assert config.forward_return_days == 5
    assert config.preprocess.winsorize == (0.05, 0.95)
    assert config.synthesis.method == "ic_weight"
    assert config.data.price == "custom/prices.parquet"


def test_load_config_retail_costs(tmp_path: Path) -> None:
    config_file = tmp_path / "config.yaml"
    config_file.write_text(
        """
costs:
  retail_mode: true
  min_commission: 5.0
  stamp_tax: 0.0005
  lot_size: 100
  initial_capital: 10000
  max_holdings: 10
  partial_rebalance: true
""",
        encoding="utf-8",
    )
    config = load_config(config_file)
    assert config.costs.retail_mode is True
    assert config.costs.min_commission == 5.0
    assert config.costs.stamp_tax == 0.0005
    assert config.costs.lot_size == 100
    assert config.costs.initial_capital == 10_000
    assert config.costs.max_holdings == 10
    assert config.costs.partial_rebalance is True


def test_forward_return_col() -> None:
    config = AppConfig(forward_return_days=20)
    assert config.forward_return_col == "forward_return_20d"
