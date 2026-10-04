# A股研究缓存准备

本目录保存分层研究使用的本地Parquet缓存，实际行情文件不提交到Git。
`fixtures/`中的合成输入用于软件测试，不能替代真实历史数据验收。
以下路径相对于传入的`--data-dir`，默认是仓库根目录下的`data/`；文件名可在配置的`data`字段中修改。

## 先检查已有数据

在已安装兼容依赖的环境中，从仓库根目录执行：

```bash
python -m a_share_multifactor.preflight --config configs/default.yaml --data-dir data
```

使用四类因子时，将配置换成`configs/run_four_factors.yaml`。Studio使用其保存的配置，
应检查相同配置所声明的时间窗口、因子和路径，不能用另一个配置的成功预检代替。
预检只读取已有文件，检查历史成员、基本面可得时点、因子可计算性和全收益基准；
不下载、不创建快照、不计算回测、不写策略结果。

`--symbols-limit`在完整面板加载和历史成员覆盖检查之后筛选研究标的，
不能让缺少历史成员行情的缓存通过预检。完整历史成员输入要求与调试标的数量是两件事。
缺少文件时，应准备有来源和适用区间的输入；关闭PIT或历史成员检查不能补足这些证据。

## 文件与用途

|默认路径|用途与前置要求|
|---|---|
|`cn_a/daily/prices.parquet`|研究区间行情及所需预热历史，覆盖该区间历史股票池中的标的|
|`cn_a/fundamentals.parquet`|带真实可得时点的基本面记录；只读加载要求此文件存在|
|`cn_a/universe/hs300_membership.parquet`|逐日历史成分；默认`use_historical_universe: true`时必需|
|`cn_a/benchmark/hs300_index.parquet`|沪深300全收益指数H00300的日收益及口径标记|
|`cn_a/alt/earnings_forecast.parquet`|业绩预告及其生效日期，用于`forecast_score`|
|`cn_a/alt/northbound_holdings.parquet`|北向持仓历史，用于`northbound_chg_5d`|
|`cn_a/alt/industry_returns.parquet`|行业收益；结合行情的行业分类和基准计算`industry_rs_20d`|

后三类输入随所选因子准备。文件存在不代表因子可计算：数据区间、时点、过期规则和
预热长度仍会影响有效行；所选因子完全缺失时预检失败。已有四ETF研究快照不包含
这套A股缓存所需的股票基本面、历史成员及上述因子输入，不能直接替代。

## 行情字段

|字段|类型|含义|
|---|---|---|
|`symbol`|string|证券代码，保留前导零|
|`date`|date/datetime|交易日|
|`open`、`high`、`low`、`close`|number|行情价格，保留来源与价格口径|
|`volume`|number|成交量，记录来源单位|
|`name`或`is_st`|string或boolean|默认启用ST过滤，至少提供其中一种适用信息|
|`industry`|string|行业相对强弱或行业中性化所需分类|
|`list_date`|date/datetime|可选上市日期；缺失时现有过滤器按缓存观察条数计算上市时长|

请保留策略所需的预热历史。缓存观察条数不等于经独立核验的上市或交易日历，
当前名称和行业标签也不能自动证明历史状态。

## 基本面与可得时点

|字段|类型|含义|
|---|---|---|
|`symbol`|string|证券代码|
|`date`|date/datetime|来源记录的观察日期|
|`available_at`|datetime|该记录可以进入研究的有证据支持的可得时点|
|`report_date`|date/datetime|来源可提供的报告期信息，不作为本加载器的PIT连接键|
|`market_cap`、`pe_ratio`、`pb_ratio`|number|市值、市盈率、市净率等所选因子输入|

默认`pit_fundamentals: true`和`require_availability_timestamp: true`。
加载器按证券将行情的`date`与基本面的`available_at`做时点连接，只使用当时已可得的记录；
`fundamental_lag_days`加在`available_at`上，`fundamental_max_age_days`限制记录年龄。
缺失`available_at`列或默认PIT模式下包含未知可得时点会失败。

报告期、观察日期和发布时间具有不同含义。不能把`report_date`、今天下载到的历史数值，
或任意固定滞后直接当成历史已知事实。行情文件内嵌的基本面值也不会覆盖发布记录的PIT选择。
关闭PIT改为同日连接是另一种研究假设，不能用于声称已通过历史可得性验收。

## 历史股票池与基准

历史股票池至少含`symbol`、`date`和`in_universe`，其中`in_universe=1`表示当天属于股票池。
空历史成员表会失败；仅抓取当前成分股可能遗漏历史退出成员，也会触发覆盖检查。
标的覆盖检查本身不认证完整退市史或每个交易日的市场状态。

基准缓存的提供方标准输出为：

|字段|类型|含义|
|---|---|---|
|`date`|date/datetime|交易日|
|`benchmark_return`|number|日全收益率，使用比例值；日期须唯一、收益须有限且不低于−1|
|`benchmark_kind`|string|`total_return`；当前加载器要求所有行均有此标记|
|`benchmark_symbol`|string|提供方记录`H00300`，用于识别来源指数|

价格指数`sh000300`不包含分红，不能作为H00300的替代品。
预检还检查调仓持有期的基准对齐。身份标记和预检成功不替代来源核验，
也不证明独立交易日历、完整历史或策略可投资性；正式执行会重新读取并校验输入。

## 需要采集时

确认来源、目标股票池、时间窗口、字段和下载预算后，显式运行采集命令：

```bash
python -m a_share_multifactor.fetch_data --config configs/default.yaml --data-dir data
```

四类因子配置还使用扩展输入：

```bash
python -m a_share_multifactor.fetch_data --config configs/run_four_factors.yaml --data-dir data --fetch-alt
```

这些命令会联网并写缓存，末尾的数据集构建还可能生成快照；默认配置范围不是小样本。
`--symbols-limit`属于采集调试选项，局部采集成功不代表完整历史股票池预检通过。
来源可能限流、缺少历史披露时点或缺少扩展数据；命令退出成功也不能替代对应配置的预检。
保留来源、采集时刻、时区、单位、价格口径和哈希，完成准备后再次运行上述只读预检。
