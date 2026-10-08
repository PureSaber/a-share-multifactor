# 披露材料与修订历史接入

`asm-disclosures`将经过核对的结构化摘录、原始文件摘要和修订链转换为研究工作台可读的 QDK 历史快照。输入必须包含原始材料，不能只提供一个未经核对的发布日期字符串。

```text
python -m a_share_multifactor.disclosures --input reviewed/input.json --out snapshots/disclosure-001
```

默认 `--availability captured`，记录只能从其实际采集时点起进入研究。显式选择 `--availability source-declared`时，所有使用的披露必须提供带时区的精确发布时间和可定位的发布时间材料；日期级/未知时间拒绝历史准入。此模式表示相信来源声明，不认证来源真伪，也不把精确字符串当成经独立验证的物理时钟。日期级公告不会被补成当日零点或开盘前。

输入例子（文件内容、哈希和日期均应替换为实际材料；不要将示意值用于真实认证）：

```json
{
  "schema": "a-share.disclosures/v1",
  "provider": "reviewed-official-disclosures",
  "license_note": "在已核对的许可范围内用于内部研究",
  "documents": [{
    "document_id": "report-1",
    "source_uri": "https://example.test/report.pdf",
    "file": "report.pdf",
    "sha256": "实际原文件的64位小写SHA256",
    "captured_at": "2026-10-08T15:00:00+08:00",
    "publication": {"precision": "unknown", "value": null, "evidence_document_id": null}
  }],
  "records": [{
    "record_id": "profit-v1", "symbol": "600519", "field": "net_profit",
    "value": "45402962298.10", "unit": "CNY",
    "effective_at": "2025-06-30T00:00:00+08:00",
    "document_id": "report-1", "supersedes": null, "locator": "主要会计数据表，归母净利润"
  }]
}
```

输入 schema 和字段集合严格校验；数值用十进制字符串，字段不能覆盖价格/成交量。支持 `CNY`、`CNY_1e4`、`CNY_1e8`、`shares`、`ratio`、`percent`、`CNY/share`；万元/亿元折为元，百分数折为比例，同一字段不能混用金额与比例。币种换算、季度累计值拆分、TTM、行业分类和公司行为不会自行推断。

`documents`可额外包含发布时间元数据文件；`publication.precision=exact`时，`value`必须带显式时区，`evidence_document_id`必须引用包中材料。`date`只记录 YYYY-MM-DD，`unknown`的 value 和 evidence_document_id 必须为 null。来源 URI 要求 HTTPS 且不含凭据；程序不会访问该 URI。材料相对输入目录取文件，路径逃逸、重复身份、哈希不符、未来发布时间都会拒绝。

对同一股票、字段、有效时点的后续修订，新增记录并通过 `supersedes`引用前一版本；准入时间必须严格递增，禁止并行分叉或同一时点含糊覆盖。不同报告期各有独立链。旧报告期的迟到修订不会挤掉更新报告期；按已知时点和有效时点的已有 QDK 规则选值。

导入输出为不可覆盖的新目录，含规范化历史、原始输入、引用材料及逐行 lineage。临时目录完成校验后才发布；失败不留下可消费的半成品。哈希用于检测损坏，不是数字签名。manifest 始终保留 `historical_authenticity_certified=false`。

将研究 recipe 的 `inputs.history`设为输出目录，并按原契约配置 `required_history`，例如 `{"net_profit":"fundamentals"}`。工作台会重新校验材料、修订链、单位及派生历史，重新签写错误 Parquet 哈希也不能掩盖与原始摘录的差异。通用 QDK 历史导入继续按其原有来源声明语义工作；它不会自动获得本模块的材料认证等级。

软件不提供商业历史库，也不自动解析任意 PDF 或认证公告时间。完整市场 PIT 仍需合法可用的历史材料、退市/成分及公司行为覆盖。真实材料应留在许可允许的位置，不提交到公开 Git；小样本验收结束后按数据留存要求删除临时副本。
