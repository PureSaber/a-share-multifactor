# 日／周频真实数据研究与模拟决策

这个开发版本使用包含修复的 `quant-data-kit` 和 `quant-lab` 不可变提交。
尚未替代冻结发布标签，不宣称 L2 数据 GA 或投资有效性认证。

## 安装与运行

在当前仓库和新的虚拟环境中安装锁定依赖，无需相邻仓库：

```powershell
python -m pip install --no-deps -r requirements.lock
python -m pip install --no-deps --no-build-isolation -e .
python -m pip check
python -m a_share_multifactor.decision_workflow --config configs/decision_watchlist.yaml
```

仅在跨仓库开发时才需要将三个仓库并列放置并执行 `tools/install_research_workspace.py`。
运行前提交本地源码修改；产物记录实际 Git commit（已安装依赖读取发行元数据），
脏工作树会阻断正式产物。`paper_state.json` 同时固定策略和内部依赖版本，
版本变更或旧账户未记录版本时拒绝续跑，需要新建实验输出目录。

默认观察名单是四只沪深主板股票，用于验证数据和工作流，不代表推荐持有。
账户为 10 万元虚拟账户，不读取真实持仓。初始模拟时点是第一次捕获的收盘，
首日没有历史成交、没有可宣称的 forward 收益。

之后在收盘后向**同一输出目录**再次运行，系统读取 `paper_state.json`，保留已经观察到的
信号和行情，并用新交易日数据推进同一 QExec 模拟账本。若中途漏跑，只盯市并处理此前已记录
的待模拟订单，不补造漏跑日的信号。配置改变需使用新的输出目录，形成独立实验。
同一账户只允许一个写入进程。

账户一旦触发最大回撤上限，本实验停止生成新订单；净值随后恢复也不会自动重启。
这一状态与决策卡使用的历史最大回撤口径一致。

```powershell
# 重放已经捕获的真实输入，不联网。过期决策只输出 observe。
python -m a_share_multifactor.decision_workflow --inputs <run>/inputs --as-of 2026-09-18 --output outputs/replay
```

## 同一份决策与账本

- `decision.json`：版本 `quant.decision/v1`；供下游使用。
- `decision.html`：由同一 JSON 生成，包含时点、虚拟持仓、拟调仓、预计成本、风险和证据。
- `standard/v2/`：QExec 的订单、成交、费用、现金账本、持仓和净值；含完整性校验。
- `inputs/manifest.json`：真实取数来源、抓取时间、原始价／复权价／日历／基准快照哈希。
- `scored_panel.parquet`、`validation_folds.csv`、`validation_fdr.csv`：当时的信号与研究诊断。
- `experiment.json`：预先给定的配置和哈希；不按本次收益选参。
- `latest.json`：最新产物指针；刷新失败也会指向 blocked 结果，避免继续显示旧买入卡。

三个状态：`blocked` 表示取数、完整性、公司行动或执行检查失败；`observe` 表示只能观察；
`paper_ready` 表示可进行下一交易日的模拟试验。后两者都不代表策略已经证明可投资。
`blocked` 和 `observe` 的目标持仓、拟调仓数组为空。

拟调仓来自本次执行器已接受的订单，不另外计算另一组数量。成交只能发生在信号后的下一根日线，
手续费进入同一精确账本：佣金比例、每订单最低佣金、卖出印花税、百分比不利滑点和成交量参与上限。
有卖出时先模拟卖出，后续收盘再按真实模拟持仓生成买入；拒单或部分成交不会被当作已完成目标。
日线无法证明盘口排队和实际成交概率；成本、规则是配置的模拟假设。

## 研究可信度与仍存在的数据边界

1. 前向收益标签保留结束及可见时点。OLS、Ridge、滚动 IC 和验证训练仅使用已成熟标签；
   配置名 `ic_weight` 改为因果滚动别名，不再使用整个评价区间的 IC 权重。
2. 复合年化增长率、算术平均年化收益、Sharpe、回撤分开计算。每日盯市的账户使用日频年化，
   调仓频率不改变收益采样频率。基准也在相同账户起始收盘置一。
3. QDK 成交量统一为股；基本面没有真实发布时间时返回未知，不伪造为数据日期。
   历史指数成分回溯撤销查询终点之后的调整，但上游事件覆盖完整性仍需独立证明。
4. 最新因子使用捕获时的复权价格比率，执行和估值使用未复权价。旧 forward 信号固定保存，
   后续复权版本不能改写旧信号。整个历史区间的因子诊断仍是当前数据版本下的回顾研究，
   不是已经验证历史可获得时间的回测。
5. 公司行动只有在公告、股权登记、除权与发放日证据齐全且能解释复权差异时进入账本。现金分红
   在除息日确认为不可用的应收资产，在真实付款日转成现金；延期送股仍因缺少股份应收账本而阻断。
6. 当前名单不是历史沪深 300，存在名单选择偏差。walk-forward/FDR 只描述指定因子的诊断，
   不作为自动批准真实投资的依据；最终未触碰留出区间和长期 forward 证据尚待积累。

旧的 `asm-backtest` 输出也以单次 QExec 回放为账户权威来源；多分组研究助手仍可调用。
不再同时生成与账户不同的 standard/v1 回放。旧 retail 提前退出、最短持有规则未在 QExec
中实现，调用时明确拒绝。未复权执行价格和严格 PIT 的前置检查可能阻断旧缓存／配置，需更新输入，
不能通过关闭检查恢复“认证”标签。

## 研究配方的组合与历史执行选项

`preflight_recipe(recipe,candidate=None)`只读取并校验输入，不运行回放、不生成成交。返回的
`asm.research-preflight/v1`报告包含`passed`、`issues`、因子`requirements`、逐标的覆盖率以及
归一化后的`allocation`和`execution`。`instrument_master`逐标的列出需要覆盖的已上市会话；
规则参数在开盘前尚不可用或已超出有效期时返回`INSTRUMENT_MASTER_PIT_COVERAGE`，不进入零成交
的伪成功回放。从未上市且没有交易的标的不要求历史规则参数。界面预检与正式执行共用同一准备路径。

显式`allocation`支持`equal`、`inverse_vol`和`cost_aware`。后者调用`quant-portfolio`既有
均值方差优化器。每次调仓在信号收盘后读取同一QExec账本的当前NAV、持仓和收盘价，再按整手、
现金缓冲、单票上限、佣金、最低佣金和滑点生成目标数量；未配置`allocation`的旧配方继续使用
原来的初始资金目标份数逻辑。

显式`execution`要求`listed`、`delisted`、`tradable`、`limit_up`、`limit_down`五个完整的
PIT状态字段。`dynamic`模式另要求一个`universe`字段。退出研究池只把正常目标权重降为零，
仍需按市场状态卖出；它不是退市证据。停牌冻结持仓，涨停禁买、跌停禁卖，A股T+1继续由QExec
规则和精确账本判断。每个会话使用IOC订单，拒单、部分成交后的到期和有界逐会话重试写入
`execution_diagnostics.json`。显式退市时如果
仍有持仓而输入没有当前实现支持的处置事实，回放明确失败，不生成虚假卖单或成交。日线模型只接受
开盘撮合前已知的状态；开盘后才出现且会改变当日状态的记录会因无法因果排序而失败。
