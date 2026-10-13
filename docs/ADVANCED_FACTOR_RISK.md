# 高级因子风险接入

`risk_model.exposure_pipeline`从已经冻结导入的QDK历史中逐字段选择当时已生效、已可得的
描述子、市值与行业分类，保留原始版本。需要`inputs.history`及对应`required_history`映射；
不能用当日读取时间替换原始数据时间，也不能用最新行业覆盖历史。

下面是配方中的相关片段，字段名须与自己的已导入历史一致：

```yaml
required_history:
  log_size: fundamentals
  book_to_price: fundamentals
  market_cap: fundamentals
  sector: classification
risk_model:
  model_kind: fundamental_style
  lookback: 252
  min_periods: 120
  min_asset_observations: 120
  min_assets_per_period: 30
  exposure_pipeline:
    descriptor_fields:
      size: log_size
      value: book_to_price
    market_cap_field: market_cap
    industry_field: sector
    config:
      styles:
        size: {size: 1.0}
        value: {value: 1.0}
      industry_universe: [financials, industrials, technology]
      standardization_weighting: market_cap
      winsor_method: mad
      winsor_limit: 5.0
      missing_policy: error
      max_descriptor_age_days: 550
      max_classification_age_days: 550
      orthogonalize: {value: [size]}
  estimation:
    regression_weighting: sqrt_market_cap
    industry_constraint: true
    factor_half_life: 90
    specific_half_life: 60
    risk_horizon: 5
    newey_west_lags: 4
    specific_newey_west_lags: 4
    regime_window: 20
    regime_min_periods: 60
    regime_half_life: 10
    specific_target: industry
    specific_shrinkage_method: bayesian
    specific_prior_dof: 20
    specific_min_group_assets: 3
    specific_group_fallback: error
  factor_bounds:
    size: [-1.0, 1.0]
    industry:financials: [0.0, 0.4]
```

这些数值只是显式配置示例，并非推荐或经过真实市场校准的参数。实际行业全集应与估计
资产池匹配，每期全部声明行业均需有覆盖；薄组与共线性会明确失败。十类公开风格预设
及多描述子组合在quant-risk-monitor的`public_equity_style_config`/`ExposureBuildConfig`
中提供。旧`exposure_fields`与新`exposure_pipeline`不能同时声明。

构建后的风险矩阵直接传给现有优化器；因子与主动因子上下限、行业集中度和跟踪误差
继续由组合检查执行。报告保留年化风险率，并明确`risk_horizon`与`horizon_variance`。
H期方差等于年化矩阵方差乘H/A；这是线性收益近似，不代表精确复利账本。

估计资产池使用完整历史价格覆盖；动态上市/退市或缺失历史不会被静默填零。行业变化
必须来自原始历史，数据年龄约束逐字段检查。上游QDK历史导入本身要求已提供值有限，
故缺失描述子的研究性行业填补应在独立暴露研究入口使用，不能借该配方伪造未知历史。

校准在quant-risk-monitor的`calibrate-risk`执行。选择期挑候选后冻结，再评价独立留出期；
返回结果不会自动覆盖此配方。算法、独立验算、合成样例与界限见
[风险模型完整文档](https://github.com/PureSaber/quant-risk-monitor/blob/main/docs/ADVANCED_FACTOR_RISK.md)。
