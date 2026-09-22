# L1 基线对照（tiny）

> 脚本 `scripts/router/train_l1.py`；口径 `eval_harness.py`；三层合成 `l1_data.assemble_route`。

- 设备：cuda（NVIDIA GeForce RTX 4050 Laptop GPU）
- 参数量：338,975（fp32 ≈ 1.36 MB）
- 推理：单条 1.45 ms·次⁻¹ / 整批 26.4 ms·844条⁻¹（= 0.031 ms·条⁻¹）
- 训练合计 52.2s / 5 折
- 标签空间：31 类；L1 可训练样本 815 条
- 契约来源：snapshot

## 指标对比

| 指标 | 词法基线(groupCV) | L1-tiny 纯模型 | L1-tiny+规则 |
| --- | --- | --- | --- |
| 样本数 | 844 | 844 | 844 |
| **整条决策准确率** | 80.7% | 67.4% | 73.1% |
| 决策类型准确率 | 97.8% | 90.5% | 96.2% |
| ① 可执行决策准确率 | 79.8% | 69.7% | 72.5% |
| ② 语义闸门准确率 | 96.6% | 37.3% | 84.8% |
| 工具选择准确率 | 80.1% | 72.6% | 72.6% |
| · 类型对时工具对 | 79.8% | 69.7% | 72.5% |
| chat F1 | 82.9% | 71.0% | 74.6% |
| call F1 | 98.9% | 95.2% | 98.3% |
| ask F1 | 87.3% | 0.0% | 72.3% |
| escalate F1 | 98.2% | 0.0% | 98.2% |
| 无谓升级率 ↓ | 0.0% | 0.0% | 0.0% |
| 漏升级率 ↓ | 3.5% | 100.0% | 3.5% |
| 升级理由准确率 | 89.7% | 0.0% | 89.7% |

## 分组折明细

| fold | train_l1 | test | best_epoch | val_acc | train_s |
| --- | --- | --- | --- | --- | --- |
| 1 | 655 | 164 | 58 | 80.5% | 10.73 |
| 2 | 660 | 163 | 60 | 59.2% | 10.48 |
| 3 | 656 | 167 | 60 | 78.9% | 10.57 |
| 4 | 644 | 175 | 13 | 73.0% | 10.36 |
| 5 | 645 | 175 | 50 | 79.9% | 10.06 |

## L1 纯模型混淆（金标→预测）

| 金标 | 预测 | 次数 |
| --- | --- | --- |
| call::ml.predict | call::ml.evaluate | 6 |
| call::dataset.quality | call::eda.outlier | 6 |
| call::dataset.schema | call::ml.detect_task | 6 |
| call::data.transform | call::dataset.schema | 5 |
| call::eda.visualize | call::eda.distribution | 5 |
| call::dataset.inspect | call::dataset.preview | 4 |
| call::dataset.quality | call::data.clean | 4 |
| call::dataset.inspect | call::dataset.schema | 4 |
| call::data.clean | call::dataset.quality | 4 |
| call::workflow.inspect | call::workflow.run | 4 |
| call::ml.explain | call::ml.evaluate | 4 |
| call::data.filter | call::dataset.schema | 4 |
| call::dataset.preview | call::dataset.schema | 4 |
| call::dataset.quality | call::dataset.inspect | 4 |
| call::data.transform | call::dataset.list | 4 |

## 误差互补性（L1 vs 词法基线，整条 route）

| 组合 | 条数 |
| --- | --- |
| 两者都对 | 572 |
| 仅词法基线对 | 109 |
| 仅 L1 对 | 45 |
| 两者都错 | 118 |

L1 补上了词法基线 **27.6%** 的错误；最佳单模型 **80.7%**；完美选择器上限 **86.0%**。

> 「完美选择器上限」是**事后**统计（逐条取两者中较优），不是可达指标；它的意义只是判断「融合两个模型有没有空间」。对比对象是**已套 L0 规则的部署形态基线**。

## 口径说明

- **purpose**：回答「L1 神经模型是否超过词法基线」；两个口径并列：纯模型看语义能力，+规则看线上质量。
- **fairness**：同一 group_folds（同 seed/折数/group_key）；同一次运行内重算词法基线以消除 sklearn 版本差异。
- **no_test_peek**：词表/权重/最佳 epoch 均只用训练折；测试折仅前向一次。
- **escalate_excluded**：升级样本不参与 L1 训练（由 L0 规则兜住），但仍出现在测试折中按其规则结果计分。