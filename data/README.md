# Data Directory

Local Parquet cache for A-share market data. This directory is gitignored; populate it via:

```bash
python -m a_share_multifactor.fetch_data --config configs/default.yaml
```

## Layout

```
data/
├── cn_a/
│   ├── daily/
│   │   └── prices.parquet
│   ├── fundamentals.parquet
│   ├── universe/
│   │   └── hs300_membership.parquet
│   └── benchmark/
│       └── hs300_index.parquet
```

## Parquet Schemas

### `cn_a/daily/prices.parquet`

| Column   | Type   | Description    |
|----------|--------|----------------|
| symbol   | string | Stock code     |
| date     | date   | Trading date   |
| open     | float  | Open price     |
| high     | float  | High price     |
| low      | float  | Low price      |
| close    | float  | Close price    |
| volume   | float  | Trading volume |
| name     | string | Optional       |
| industry | string | Optional       |

### `cn_a/fundamentals.parquet`

| Column      | Type   | Description                         |
|-------------|--------|-------------------------------------|
| symbol      | string | Stock code                          |
| date        | date   | Observation date from data source   |
| report_date | date   | Report/as-of date used for PIT merge|
| market_cap  | float  | Market capitalization               |
| pe_ratio    | float  | Price-to-earnings ratio             |
| pb_ratio    | float  | Price-to-book ratio                 |

### Point-in-Time merge

When `filters.pit_fundamentals: true` (default), prices and fundamentals are merged with
`pd.merge_asof` per symbol: each trading day only sees the latest fundamental record whose
`report_date` (plus optional `fundamental_lag_days`) is on or before that day.

Set `pit_fundamentals: false` to use same-day exact join (may overstate availability of
fundamental data on sparse report dates).

### `cn_a/universe/hs300_membership.parquet`

| Column      | Type   | Description              |
|-------------|--------|--------------------------|
| symbol      | string | Stock code               |
| date        | date   | Trading date             |
| in_universe | int    | 1 if HS300 member        |

### `cn_a/benchmark/hs300_index.parquet`

| Column            | Type  | Description        |
|-------------------|-------|--------------------|
| date              | date  | Trading date       |
| benchmark_return  | float | Daily index return |
