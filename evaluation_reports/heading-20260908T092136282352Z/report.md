# 父标题单变量实验

仅20条开发集；26条切片正文与标签不变，保留推荐问题章节。

| 语料 | 方案 | Recall@5 | MRR | nDCG@5 | 重排成功 |
| --- | --- | --- | --- | --- | --- |
| baseline | vector | 0.950 | 0.817 | 0.851 | 0/20 |
| baseline | hybrid_rrf | 0.950 | 0.854 | 0.878 | 0/20 |
| baseline | hybrid_rerank | 1.000 | 0.960 | 0.969 | 20/20 |
| heading | vector | 0.950 | 0.817 | 0.850 | 0/20 |
| heading | hybrid_rrf | 1.000 | 0.839 | 0.879 | 0/20 |
| heading | hybrid_rerank | 0.950 | 0.950 | 0.950 | 20/20 |

## 逐题排名变化（旧 → 新）

- vector/ac-001: [2] → [4]
- vector/ac-002: [2] → [1]
- vector/ac-011: [2] → [4]
- hybrid_rrf/ac-001: [1] → [2]
- hybrid_rrf/ac-008: [] → [5]
- hybrid_rrf/ac-011: [3] → [4]
- hybrid_rrf/ac-013: [4] → [3]
- hybrid_rerank/ac-013: [5] → []

AI初标、单文档小样本；不代表生产效果。未运行重排时不能推断重排后的排名。