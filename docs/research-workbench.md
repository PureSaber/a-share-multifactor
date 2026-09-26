# A股与ETF研究适配器

`EquityResearchExecutor`把quant-lab闭合配方转换成独立QExec回放。运行入口见quant-workspace的`profiles/research-workbench`，或安装完整工作台后使用`python -m quant_pipeline.research_workbench recipe.yaml --output ../studies/my-study`。

支持17个共享因子的显式方向等权秩组合、固定观察池、日/周/月调仓、ETF均线过滤及不满足条件时持现金。输入必须是已有decision_workflow不可变raw/adjusted/calendar/benchmark包和显式证券主表；基本面因子需额外PIT历史，不能把现值当历史。预检失败时保留preflight.json。

每个候选仅使用QExec标准账本，信号在收盘后形成、由后续行情事件成交，保留参与率、费用、滑点、公司行动和风控行为。份额按初始资金确定，暂未实现逐次净值再平衡。基准按主配方可投入资金等分观察池。变动成分/限制交易状态目前拒绝，避免假装具备停牌撮合能力。

输出standard/v2、returns.csv、factors.json、preflight.json及机器可读研究结果。真实历史样例只有已保存的40个交易日，不能解释为多年稳定或样本外收益。合成ETF样例明确标识synthetic。

顺带修复行情缓存与基本面合并时产生_x/_y列的根因：发布数据拥有基本面字段；缓存现值不能绕过PIT披露时点。未知因子不再静默丢弃。
