# 01 / 04 A 股适配

共享新因子可在 factor_names 显式选择 average_volume_20d、turnover_rate_20d_v2、amihud_illiq_20d_v2；后两者需要来源单位、原始流通股数和实际成交金额/收益价格。默认仍用旧因子集，不因新增 registry 项隐式要求新数据。

研究历史状态解析允许 unknown/缺行，传到 QExec 状态事件；不把缺失转换为 False 或可交易。listed 的 pandas 筛选使用显式 eq(True)/eq(False)，避免 nullable/object 布尔取反变成整数。严格 preflight 仍会阻断证据不足的完整研究认证；底层执行也阻断 unknown。

tests/test_qf_parity.py 验证全部共享因子与 quant-factors 数值一致；tests/test_research_workbench.py 验证缺失状态 fail-closed。已有日频限制仍在：开盘后首次披露的状态不能反推开盘成交。
