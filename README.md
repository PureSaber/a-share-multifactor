# A-Share Multifactor

A 股多因子选股研究项目：从 AKShare 拉取行情，计算因子，做 IC 分析、分层回测与散户约束下的组合模拟。

**研究用途**：结果仅供学习研究，不构成投资建议。

## 技术栈

Python · Pandas · Scikit-learn · AKShare · PyArrow · Matplotlib

## 功能概览

| 模块 | 说明 |
|------|------|
| 因子 | 市值、PE、PB、20 日动量、20 日波动率 |
| 合成 | 等权、IC 加权、滚动 IC、Ridge、OLS |
| 回测 | 分层回测、HS300 基准超额、换手成本 |
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

CI（GitHub Actions）在 push / PR 时自动运行 Ruff + Pytest（Python 3.10 / 3.11）。

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
