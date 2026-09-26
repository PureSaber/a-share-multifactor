# A股与ETF研究适配器

`EquityResearchExecutor`把quant-lab闭合配方转换成独立QExec回放。运行入口见quant-workspace的`profiles/research-workbench`，或安装完整工作台后使用`python -m quant_pipeline.research_workbench recipe.yaml --output ../studies/my-study`。

支持17个共享因子的显式方向等权秩组合、固定观察池、日/周/月调仓、ETF均线过滤及不满足条件时持现金。输入必须是已有decision_workflow不可变raw/adjusted/calendar/benchmark包和显式证券主表；基本面因子需额外PIT历史，不能把现值当历史。预检失败时保留preflight.json。

每个候选仅使用QExec标准账本，信号在收盘后形成、由后续行情事件成交，保留参与率、费用、滑点、公司行动和风控行为。配置allocation后按当次账本NAV重新计算份额；equal、inverse_vol、cost_aware均有明确换手约束。未配置allocation的旧配方仍按初始资金计算，结果会标明该限制。显式execution配置可接入五类PIT交易状态及动态池；缺少真实状态时不得宣称完整动态PIT回测。

输出standard/v2、returns.csv、factors.json、preflight.json及机器可读研究结果。真实历史样例只有已保存的40个交易日，不能解释为多年稳定或样本外收益。合成ETF样例明确标识synthetic。

顺带修复行情缓存与基本面合并时产生_x/_y列的根因：发布数据拥有基本面字段；缓存现值不能绕过PIT披露时点。未知因子不再静默丢弃。

## 风险模型、敞口与执行

`risk_model`要求`allocation.mode: cost_aware`。逐日使用期初已公开的X与之后实现的收益做截面回归，估计年化F/D及Σ=XFXᵀ+D；Σ直接传入优化器。风险收益由原始成交价及除权日前已公告的现金/份额权益构造，未来复权版本不会改写此前风险收益。市场常数因子用于小ETF池时必须标识`statistical_proxy`，不代表MSCI Barra、ETF底层穿透或已验证的基本面描述子。

`factor_bounds`限制绝对因子敞口，`active_factor_bounds`结合显式`benchmark_weights`转换为绝对优化约束；`max_tracking_error`在整手份额生成后作为拒单门禁，暂不作为二次优化约束。若持仓数量/行业上限/换手/因子边界不能同时满足，执行失败并保留原因，不缩减约束或使用未收敛结果。

`risk.max_industry_weight`是每个行业的统一上限，`industry_field`必须映射到PIT classification。行业one-hot进入组合优化，整手目标和每日实际持仓分别检查。价格波动使已投资持仓出现critical后，账户锁存并取消挂单；`exposure_breach_action`默认halt，可显式设为liquidate通过reduce_only清仓，之后不自动重启。全现金的基准偏离不在入场前触发该锁存，但目标仍须通过主动约束。退出到现金不是“恢复跟踪误差合规”，报告持续记录实际偏离。订单级CrossAssetRiskPolicy同时检查实际gross与单品种集中度，所有卖出意图均为reduce_only，允许在保留数据检查的前提下逐步降低已超限风险。

`risk.drawdown_action`支持`halt`与`liquidate`，默认halt。二者均锁存并取消未完成挂单、释放预留；liquidate通过原QExec账本发出reduce_only退出意图。停牌、T+1、无成交量仍可能延迟清仓，不虚构成交。清仓不受普通调仓换手限制阻止，仍受交易规则和PIT验证约束。

`neutralization: [industry, market_cap]`会改变实际打分，要求相应classification/fundamentals历史；缺字段拒绝。训练阶段使用train_ic时，方向选择依据同一“中性化→截面秩→信号延迟”信号的成熟标签IC；延迟先于评估区间裁剪，因此预热记录不会丢失。

每次运行新增risk_models.json与execution_diagnostics.json中的risk_checks/runtime_risk_events。报告区分目标拒绝、真实持仓漂移、回撤锁存，模型/暴露/协方差均可追溯至输入包与代码SHA。真实历史walk-forward仍是回溯样本外对比，真实前向必须另行冻结并等待未来市场数据。
