"""Configuration loading and typed app settings."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

DEFAULT_FACTOR_DIRECTIONS: dict[str, int] = {
    "market_cap": -1,
    "pe_ratio": -1,
    "pb_ratio": -1,
    "momentum_20d": 1,
    "reversal_5d": 1,
    "turnover_20d": -1,
    "volatility_20d": -1,
    "forecast_score": 1,
    "northbound_chg_5d": 1,
    "industry_rs_20d": 1,
}


@dataclass
class DataPaths:
    price: str = "cn_a/daily/prices.parquet"
    fundamentals: str = "cn_a/fundamentals.parquet"
    universe: str = "cn_a/universe/hs300_membership.parquet"
    benchmark: str = "cn_a/benchmark/hs300_index.parquet"
    earnings_forecast: str = "cn_a/alt/earnings_forecast.parquet"
    northbound: str = "cn_a/alt/northbound_holdings.parquet"
    industry_returns: str = "cn_a/alt/industry_returns.parquet"


@dataclass
class FilterConfig:
    use_historical_universe: bool = True
    exclude_st: bool = True
    min_list_days: int = 60
    pit_fundamentals: bool = True
    fundamental_lag_days: int = 0


@dataclass
class PreprocessConfig:
    winsorize: tuple[float, float] = (0.01, 0.99)
    standardize: str = "zscore"
    neutralize: bool = False
    neutralize_by: list[str] = field(default_factory=lambda: ["industry"])


@dataclass
class SynthesisConfig:
    method: str = "equal_weight"
    lookback_months: int = 12
    ridge_alpha: float = 1.0


@dataclass
class GridSearchConfig:
    quantiles: list[int] = field(default_factory=lambda: [3, 5, 10])
    winsorize_upper: list[float] = field(default_factory=lambda: [0.95, 0.99])


@dataclass
class CostsConfig:
    commission: float = 0.0003
    slippage: float = 0.001
    retail_mode: bool = False
    min_commission: float = 5.0
    stamp_tax: float = 0.0005
    lot_size: int = 100
    initial_capital: float = 100_000.0
    max_holdings: int = 0
    partial_rebalance: bool = True
    trade_freq: str = "monthly"
    min_holding_days: int = 20
    early_exit_enabled: bool = True
    early_exit_single_day_return: float = 0.08
    early_exit_consecutive_days: int = 3
    early_exit_consecutive_daily: float = 0.03
    early_exit_cumulative_return: float = 0.25
    rank_change_threshold: int = 0


@dataclass
class FetchConfig:
    max_workers: int = 4
    sleep_seconds: float = 0.2
    max_retries: int = 3


@dataclass
class AppConfig:
    universe: str = "hs300"
    start_date: str = "2020-01-01"
    end_date: str = "2024-12-31"
    factors: list[str] = field(
        default_factory=lambda: [
            "market_cap",
            "pe_ratio",
            "pb_ratio",
            "momentum_20d",
            "volatility_20d",
        ]
    )
    factor_directions: dict[str, int] = field(
        default_factory=lambda: DEFAULT_FACTOR_DIRECTIONS.copy()
    )
    quantiles: int = 5
    rebalance_freq: str = "monthly"
    holding_period: str = "rebalance"
    forward_return_days: int = 20
    outputs_dir: str = "outputs"
    data: DataPaths = field(default_factory=DataPaths)
    filters: FilterConfig = field(default_factory=FilterConfig)
    preprocess: PreprocessConfig = field(default_factory=PreprocessConfig)
    synthesis: SynthesisConfig = field(default_factory=SynthesisConfig)
    costs: CostsConfig = field(default_factory=CostsConfig)
    fetch: FetchConfig = field(default_factory=FetchConfig)
    grid_search: GridSearchConfig = field(default_factory=GridSearchConfig)
    ic_decay_horizons: list[int] = field(default_factory=lambda: [1, 5, 10, 20])

    @property
    def forward_return_col(self) -> str:
        return f"forward_return_{self.forward_return_days}d"

    @property
    def period_return_col(self) -> str:
        return "period_return"

    def factor_direction(self, factor: str) -> int:
        return self.factor_directions.get(factor, 1)

    def price_path(self, data_dir: Path | None = None) -> Path:
        root = data_dir or Path("./data")
        return root / self.data.price

    def fundamentals_path(self, data_dir: Path | None = None) -> Path:
        root = data_dir or Path("./data")
        return root / self.data.fundamentals

    def universe_path(self, data_dir: Path | None = None) -> Path:
        root = data_dir or Path("./data")
        return root / self.data.universe

    def benchmark_path(self, data_dir: Path | None = None) -> Path:
        root = data_dir or Path("./data")
        return root / self.data.benchmark


def _parse_factor_directions(raw: dict[str, Any] | None) -> dict[str, int]:
    directions = DEFAULT_FACTOR_DIRECTIONS.copy()
    if not raw:
        return directions
    for factor, value in raw.items():
        if isinstance(value, dict):
            directions[factor] = int(value.get("direction", 1))
        else:
            directions[factor] = int(value)
    return directions


def _dict_to_config(raw: dict[str, Any]) -> AppConfig:
    defaults = AppConfig()
    data_raw = raw.get("data", {})
    preprocess_raw = raw.get("preprocess", {})
    synthesis_raw = raw.get("synthesis", {})
    filters_raw = raw.get("filters", {})
    costs_raw = raw.get("costs", {})
    fetch_raw = raw.get("fetch", {})
    grid_raw = raw.get("grid_search", {})
    ic_decay_horizons = raw.get("ic_decay_horizons", defaults.ic_decay_horizons)

    winsorize = preprocess_raw.get("winsorize", list(defaults.preprocess.winsorize))
    if isinstance(winsorize, list):
        winsorize = tuple(winsorize)

    neutralize_by = preprocess_raw.get("neutralize_by", defaults.preprocess.neutralize_by)
    if isinstance(neutralize_by, str):
        neutralize_by = [neutralize_by]

    return AppConfig(
        universe=raw.get("universe", defaults.universe),
        start_date=str(raw.get("start_date", defaults.start_date)),
        end_date=str(raw.get("end_date", defaults.end_date)),
        factors=list(raw.get("factors", defaults.factors)),
        factor_directions=_parse_factor_directions(raw.get("factor_directions")),
        quantiles=int(raw.get("quantiles", defaults.quantiles)),
        rebalance_freq=raw.get("rebalance_freq", defaults.rebalance_freq),
        holding_period=raw.get("holding_period", defaults.holding_period),
        forward_return_days=int(raw.get("forward_return_days", defaults.forward_return_days)),
        outputs_dir=raw.get("outputs_dir", defaults.outputs_dir),
        data=DataPaths(
            price=data_raw.get("price", defaults.data.price),
            fundamentals=data_raw.get("fundamentals", defaults.data.fundamentals),
            universe=data_raw.get("universe", defaults.data.universe),
            benchmark=data_raw.get("benchmark", defaults.data.benchmark),
            earnings_forecast=data_raw.get(
                "earnings_forecast", defaults.data.earnings_forecast
            ),
            northbound=data_raw.get("northbound", defaults.data.northbound),
            industry_returns=data_raw.get(
                "industry_returns", defaults.data.industry_returns
            ),
        ),
        filters=FilterConfig(
            use_historical_universe=bool(
                filters_raw.get("use_historical_universe", defaults.filters.use_historical_universe)
            ),
            exclude_st=bool(filters_raw.get("exclude_st", defaults.filters.exclude_st)),
            min_list_days=int(filters_raw.get("min_list_days", defaults.filters.min_list_days)),
            pit_fundamentals=bool(
                filters_raw.get("pit_fundamentals", defaults.filters.pit_fundamentals)
            ),
            fundamental_lag_days=int(
                filters_raw.get("fundamental_lag_days", defaults.filters.fundamental_lag_days)
            ),
        ),
        preprocess=PreprocessConfig(
            winsorize=winsorize,
            standardize=preprocess_raw.get("standardize", defaults.preprocess.standardize),
            neutralize=bool(preprocess_raw.get("neutralize", defaults.preprocess.neutralize)),
            neutralize_by=list(neutralize_by),
        ),
        synthesis=SynthesisConfig(
            method=synthesis_raw.get("method", defaults.synthesis.method),
            lookback_months=int(
                synthesis_raw.get("lookback_months", defaults.synthesis.lookback_months)
            ),
            ridge_alpha=float(synthesis_raw.get("ridge_alpha", defaults.synthesis.ridge_alpha)),
        ),
        costs=CostsConfig(
            commission=float(costs_raw.get("commission", defaults.costs.commission)),
            slippage=float(costs_raw.get("slippage", defaults.costs.slippage)),
            retail_mode=bool(costs_raw.get("retail_mode", defaults.costs.retail_mode)),
            min_commission=float(costs_raw.get("min_commission", defaults.costs.min_commission)),
            stamp_tax=float(costs_raw.get("stamp_tax", defaults.costs.stamp_tax)),
            lot_size=int(costs_raw.get("lot_size", defaults.costs.lot_size)),
            initial_capital=float(costs_raw.get("initial_capital", defaults.costs.initial_capital)),
            max_holdings=int(costs_raw.get("max_holdings", defaults.costs.max_holdings)),
            partial_rebalance=bool(
                costs_raw.get("partial_rebalance", defaults.costs.partial_rebalance)
            ),
            trade_freq=str(costs_raw.get("trade_freq", defaults.costs.trade_freq)),
            min_holding_days=int(
                costs_raw.get("min_holding_days", defaults.costs.min_holding_days)
            ),
            early_exit_enabled=bool(
                costs_raw.get("early_exit_enabled", defaults.costs.early_exit_enabled)
            ),
            early_exit_single_day_return=float(
                costs_raw.get(
                    "early_exit_single_day_return",
                    defaults.costs.early_exit_single_day_return,
                )
            ),
            early_exit_consecutive_days=int(
                costs_raw.get(
                    "early_exit_consecutive_days",
                    defaults.costs.early_exit_consecutive_days,
                )
            ),
            early_exit_consecutive_daily=float(
                costs_raw.get(
                    "early_exit_consecutive_daily",
                    defaults.costs.early_exit_consecutive_daily,
                )
            ),
            early_exit_cumulative_return=float(
                costs_raw.get(
                    "early_exit_cumulative_return",
                    defaults.costs.early_exit_cumulative_return,
                )
            ),
            rank_change_threshold=int(
                costs_raw.get("rank_change_threshold", defaults.costs.rank_change_threshold)
            ),
        ),
        fetch=FetchConfig(
            max_workers=int(fetch_raw.get("max_workers", defaults.fetch.max_workers)),
            sleep_seconds=float(fetch_raw.get("sleep_seconds", defaults.fetch.sleep_seconds)),
            max_retries=int(fetch_raw.get("max_retries", defaults.fetch.max_retries)),
        ),
        grid_search=GridSearchConfig(
            quantiles=list(grid_raw.get("quantiles", defaults.grid_search.quantiles)),
            winsorize_upper=list(
                grid_raw.get("winsorize_upper", defaults.grid_search.winsorize_upper)
            ),
        ),
        ic_decay_horizons=list(ic_decay_horizons),
    )


def load_config(config_path: Path) -> AppConfig:
    """Load YAML config and return typed AppConfig with defaults applied."""
    with config_path.open(encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    return _dict_to_config(raw)
