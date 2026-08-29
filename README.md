# A-Share Multifactor

本版本`0.4.1`完成M6策略层依赖治理。认证路径消费`standard/v2@2.0.0`，策略只通过
`Strategy.on_event`产生订单意图，成交、费用、滑点、持仓和NAV继续由统一执行与账本提供；
本次不改变策略逻辑和历史`standard/v1`语义。

A 股多因子选股研究项目：从 AKShare 拉取行情，计算因子，做 IC 分析、分层回测与散户约束下的组合模拟。

**研究用途**：结果仅供学习研究，不构成投资建议。

## 技术栈

Python · Pandas · Scikit-learn · [**quant-data-kit**](https://github.com/PureSaber/quant-data-kit) · AKShare · PyArrow · Matplotlib

## 四类因子（基本面 / 技术面 / 情感面 / 宏观面）

| 类别 | 因子 | 说明 |
|------|------|------|
| 基本面 | `pe_ratio`, `pb_ratio`, `market_cap`, `forecast_score` | 估值 + 业绩预告得分 |
| 技术面 | `momentum_20d`, `volatility_20d` | 动量、波动 |
| 情感面 | `northbound_chg_5d` | 北向持股 5 日变化（T+1 披露滞后） |
| 宏观面 | `industry_rs_20d` | 行业 20 日相对 HS300 强弱 |

```bash
# 安装审计过的运行时、开发和editable构建依赖；Notebook工具单独安装`.[notebook]`
python -m pip install --no-deps --requirement requirements.lock
python -m pip check
python -m pip install --no-deps --no-build-isolation --editable .
python -m pip check

# 本地开发已通过上面的no-deps、no-build-isolation方式安装editable项目。

# 拉取含另类数据
python -m a_share_multifactor.fetch_data --fetch-alt --symbols-limit 10 --verbose

# 四类因子 IC + 回测
python -m a_share_multifactor.backtest --config configs/run_four_factors.yaml --verbose
# 输出: outputs/four_factors/ic_summary.csv, report.html
```

离线 smoke test（无网络时）：

```bash
python scripts/seed_alt_smoke_data.py
python -m a_share_multifactor.backtest --config configs/run_four_factors.yaml --symbols-limit 5
```

## 功能概览

| 模块 | 说明 |
|------|------|
| 因子 | 市值、PE、PB、20 日动量、20 日波动率 |
| 合成 | 等权、IC 加权、滚动 IC、Ridge、OLS |
| 回测 | 分层回测、HS300 基准超额、换手成本 |
| 数据治理 | PIT可获得时间、历史股票池、不可变快照、质量摘要 |
| 验证 | expanding walk-forward、泄漏审计、FDR校正、折间稳定性 |
| 运行契约 | 标准returns/positions/orders/costs/exposures及哈希清单 |
| 散户模式 | 1 万元本金、100 股一手、最低佣金、印花税 |
| 调仓频率 | 日 / 周 / 月；最少持有天数；可选提前止盈 |
| 工具 | 多模型对比、81 组参数网格、指定日期买入清单 |

## 目录结构

```
a-share-multifactor/
├── configs/                  # YAML 配置
├── src/a_share_multifactor/  # 核心库
├── scripts/                  # 辅助脚本
├── notebooks/                # Jupyter 研究笔记
├── tests/                    # 单元测试
├── data/                     # Parquet 缓存（gitignore，见 data/README.md）
└── outputs/                  # 回测输出（gitignore）
```

## 环境要求

- Python **3.10+**
- 可访问 AKShare 数据源的网络环境
- 全量拉取 HS300 约 **15–20 分钟**（视网络与限流而定）

## 快速开始（复现）

```bash
# 1. 克隆并安装
git clone https://github.com/PureSaber/a-share-multifactor.git
cd a-share-multifactor
python -m venv .venv

# Windows
.venv\Scripts\activate
# macOS / Linux
# source .venv/bin/activate

pip install -e ".[dev]"

# 2. （可选）复制环境变量模板
copy .env.example .env   # Windows
# cp .env.example .env   # macOS / Linux

# 3. 拉取并缓存数据
python -m a_share_multifactor.fetch_data --config configs/default.yaml
# 或: asm-fetch --config configs/default.yaml

# 调试时可限制股票数量
python -m a_share_multifactor.fetch_data --symbols-limit 10 --verbose

# 4. 运行 IC 分析 + 分层回测
python -m a_share_multifactor.backtest --config configs/default.yaml --verbose
# 或: asm-backtest --config configs/default.yaml --verbose

# 5. 查看 outputs/latest/ 下的 CSV 与 report.html

# 6. 运行测试
pytest -q
```

## CLI 命令

安装后可使用以下入口（等价于 `python -m ...`）：

| 命令 | 说明 |
|------|------|
| `asm-fetch` | 拉取 AKShare 数据到 Parquet |
| `asm-backtest` | IC 分析 + 分层回测 |
| `asm-grid-search` | 因子参数扫描 |
| `asm-compare` | 多合成方法对比（含散户模式） |
| `asm-retail-grid` | 散户 OLS 参数网格（81 组） |

## 配置文件

| 文件 | 用途 |
|------|------|
| `configs/default.yaml` | 默认研究配置（demo 区间） |
| `configs/run_2025_now.yaml` | 2025 至今多模型对比 |
| `configs/run_retail_10k.yaml` | 1 万散户、**月度**调仓 |
| `configs/run_retail_daily_10k.yaml` | 1 万散户、**日/周频**调仓 |
| `configs/run_report.yaml` | 生成 HTML 报告 |

回测前请将各配置中的 `end_date` 改为你需要的截止日期，然后重新 `asm-fetch`。

### 散户 1 万 · 2025 至今对比

```bash
asm-fetch --config configs/run_retail_10k.yaml
asm-compare --config configs/run_retail_10k.yaml
```

输出：`outputs/long_only_10k_retail_2025_now/`（资金曲线、交易流水、HTML 报告）

### 散户参数网格（OLS × 止盈 × 持有期 × 频率）

```bash
asm-fetch --config configs/run_retail_daily_10k.yaml
asm-retail-grid --config configs/run_retail_daily_10k.yaml
```

输出：`outputs/retail_param_grid/`（CSV、热力图、Top 20 排名）

### 指定日期买入清单

```bash
python scripts/compute_buy_list.py --trade-date 2026-07-17 \
  --config configs/run_retail_daily_10k.yaml
```

## 主要配置项

| 字段 | 说明 |
|------|------|
| `factor_directions` | 因子方向（+1 / -1） |
| `holding_period` | `rebalance` 或固定天数 |
| `filters.use_historical_universe` | 是否按历史 HS300 成分过滤 |
| `filters.pit_fundamentals` | 时点基本面（merge_asof） |
| `synthesis.method` | equal_weight / ic_weight / rolling_ic_weight / ridge / ols |
| `costs.retail_mode` | 散户模式（整手、最低佣金等） |
| `costs.trade_freq` | daily / weekly / monthly |
| `costs.min_holding_days` | 最少持有交易日 |
| `fetch.max_workers` | 并行拉取线程数（限流时可设为 1） |
| `validation.*` | OOS训练/测试/步长/embargo与FDR阈值 |

每次认证回测在`outputs/<run_id>/standard/v2/`写入不可变`backtest-ledger`契约，完整包含
returns、positions、portfolio_snapshots、exposures、orders、order_events、fills、costs、
cash_ledger和margin，并通过`load_and_validate_standard_run`回读验证；`standard/`根目录仍
双写历史v1以保持读取兼容。认证订单只能经过
`Strategy.on_event -> DeterministicRunEngine -> 撮合/成交 -> RuleBookRiskGate -> ExactAccountLedger`。
`trading_costs.py`和`trade_ledger.py`保留为legacy/research-only，不得用于认证产物。
fixture目录是版本化、显式、PIT的测试目录，其有效期是认证fixture窗口，不代表标的上市历史。
认证写出在任何文件落盘前检查Git工作树并对dirty状态fail closed；scored_panel使用与行列顺序无关的canonical SHA-256进入dataset snapshots和lineage，fixture catalog同样以`sha256:`标识，instrument master版本绑定catalog哈希前12位。
冻结的QExec`v0.4.1`对非期货费用只提供统一Fee及原始`maker`/`taker`分类；认证路径原样保留该分类，不在产物层拆分commission或stamp_duty。若需要显式费用分类，必须由项目负责人串行修订quant-execution并重新发布冻结tag。

## M6依赖和契约治理

`pyproject.toml`和`requirements.lock`均使用已发布annotatedtag：QDK`v0.6.1`（peeledcommit
`edf1351690dc60691cc6330390adcdbf8bc79c5f`）、QFactors`v0.2.1`（`c06472b713f15b3cf8078690b33807eba6563a9c`）、
QExec`v0.4.1`（`29eccc0e392968b5f7c31976a329605aacce369a`）和QLab`v0.3.1`
（`27489d270e132adbec1bced93eb2ae84ad5e1a9b`）。禁止依赖浮动分支或未发布commit。

锁文件由Python3.10重建，覆盖runtime、dev和editable-build依赖；Jupyter等仅用于交互研究的
Notebook工具不进入CI的dev闭包，需要时单独安装`.[notebook]`。并在Python3.10、3.11、3.12
中严格按锁安装验证。重建命令为：

```bash
pip-compile --extra dev --build-deps-for editable --allow-unsafe --strip-extras \
  --resolver backtracking --index-url https://pypi.org/simple \
  --output-file requirements.lock pyproject.toml
```

迁移只新增不可变`standard/v2`产物，不改写历史v1；回滚使用Git revert同时恢复
`pyproject.toml`、`requirements.lock`、CI和本文档，再按旧锁重装。旧tag不移动、不覆盖。

## 开发规范

```bash
# 代码检查与格式化
ruff check src tests scripts
ruff format src tests scripts

# 测试
pytest -q

# 提交前钩子（推荐）
pre-commit install
pre-commit run --all-files
```

CI（GitHub Actions）在 push / PR 时自动运行Ruff静态检查和带branch coverage门禁的Pytest（Python 3.10 / 3.11 / 3.12）。

## 常见问题

- **AKShare 限流**：调低 `fetch.max_workers` 或增大 `sleep_seconds`
- **数据目录为空**：必须先运行 `asm-fetch`，详见 [data/README.md](data/README.md)
- **HS300 历史成分**：基于 AKShare 成分调整记录，可能存在数据源误差
- **PE/市值缺失**：合并时 left join，IC 计算自动 dropna
- **Windows 下 scipy/sklearn DLL 报错**：尝试 `pip install --force-reinstall scipy scikit-learn`

## 许可证

[MIT](LICENSE)

## 学习目标

- 因子投资核心概念（IC、IR、分层回测）
- 使用 Python 量化库进行 A 股因子研究的方法论
