# 空压机检索基线实验

AI依据原文初标，待人工复核；非现场维修数据。split在检索前固定，未据结果调参。

Top K=5，向量阈值为空；固定现有索引与分块，未调参。

| 分组 | 方案 | Hit@5 | Recall@5 | MRR | nDCG@5 | 平均ms | 重排成功 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| development | vector | 0.950 | 0.950 | 0.817 | 0.851 | 3.7 | 0/20 |
| development | hybrid_rrf | 0.950 | 0.950 | 0.854 | 0.878 | 6.4 | 0/20 |
| development | hybrid_rerank | 1.000 | 1.000 | 0.960 | 0.969 | 934.1 | 20/20 |
| validation | vector | 0.800 | 0.800 | 0.675 | 0.706 | 4.3 | 0/10 |
| validation | hybrid_rrf | 0.900 | 0.900 | 0.750 | 0.789 | 7.1 | 0/10 |
| validation | hybrid_rerank | 1.000 | 1.000 | 1.000 | 1.000 | 1115.6 | 10/10 |

## 非满召回或首位未命中用例

- development/vector/ac-001: AC-200的额定排气量和压力是多少？；相关排名 [2]
- development/vector/ac-002: 这台示例机器采用什么冷却和润滑方式？；相关排名 [2]
- development/vector/ac-007: E102的含义及优先检查部件是什么？；相关排名 [2]
- development/vector/ac-008: E301与E302分别表示哪个过滤器堵塞？；相关排名 []
- development/vector/ac-011: 怀疑温度显示不准，如何核对传感器？；相关排名 [2]
- development/vector/ac-013: 机器一直加载，车间压力却升不上去，会有哪些原因？；相关排名 [3]
- development/hybrid_rrf/ac-007: E102的含义及优先检查部件是什么？；相关排名 [2]
- development/hybrid_rrf/ac-008: E301与E302分别表示哪个过滤器堵塞？；相关排名 []
- development/hybrid_rrf/ac-011: 怀疑温度显示不准，如何核对传感器？；相关排名 [3]
- development/hybrid_rrf/ac-013: 机器一直加载，车间压力却升不上去，会有哪些原因？；相关排名 [4]
- development/hybrid_rerank/ac-013: 机器一直加载，车间压力却升不上去，会有哪些原因？；相关排名 [5]
- validation/vector/ac-009: E401信号异常要先检查什么？；相关排名 []
- validation/vector/ac-015: 怎么确认加载时进气阀确实打开了？；相关排名 [2]
- validation/vector/ac-024: 演示手册建议空气滤芯多久检查一次？；相关排名 [4]
- validation/vector/ac-027: CASE-AC-001最后定位到的故障原因是什么？；相关排名 []
- validation/hybrid_rrf/ac-009: E401信号异常要先检查什么？；相关排名 []
- validation/hybrid_rrf/ac-015: 怎么确认加载时进气阀确实打开了？；相关排名 [2]
- validation/hybrid_rrf/ac-024: 演示手册建议空气滤芯多久检查一次？；相关排名 [2]
- validation/hybrid_rrf/ac-027: CASE-AC-001最后定位到的故障原因是什么？；相关排名 [2]

## 实验边界

- 单份演示资料、AI初标，需人工复核；不能声称工厂诊断准确率。
- 重排成功数不足时，该方案含回退结果，不能作为完整重排性能结论。
- 分组按问题固定，同一章节可能跨组；不是跨文档泛化测试。
- 只评估检索，不评估答案正确性、拒答能力或多轮诊断。
- 延迟来自串行单次执行，受缓存与网络影响；不代表并发性能。
- 文档中推荐问答切片保留为干扰项，不标为答案证据。