# L1 基线对照（mid）

> 脚本 `scripts/router/train_l1.py`；口径 `eval_harness.py`；三层合成 `l1_data.assemble_route`。

- 设备：cuda（NVIDIA GeForce RTX 4050 Laptop GPU）
- 参数量：1,002,079（fp32 ≈ 4.01 MB）
- 推理：单条 1.74 ms·次⁻¹ / 整批 42.6 ms·844条⁻¹（= 0.050 ms·条⁻¹）
- 训练合计 181.55s / 5 折
- 标签空间：31 类；L1 可训练样本 815 条
- 契约来源：snapshot

## 指标对比

| 指标 | 词法基线(groupCV) | L1-mid 纯模型 | L1-mid+规则 |
| --- | --- | --- | --- |
| 样本数 | 844 | 844 | 844 |
| **整条决策准确率** | 80.7% | 66.5% | 73.0% |
| 决策类型准确率 | 97.8% | 90.2% | 95.5% |
| ① 可执行决策准确率 | 79.8% | 68.3% | 72.0% |
| ② 语义闸门准确率 | 96.6% | 42.4% | 89.8% |
| 工具选择准确率 | 80.1% | 72.1% | 72.1% |
| · 类型对时工具对 | 79.8% | 68.3% | 72.0% |
| chat F1 | 82.9% | 68.5% | 73.5% |
| call F1 | 98.9% | 95.1% | 97.7% |
| ask F1 | 87.3% | 0.0% | 72.9% |
| escalate F1 | 98.2% | 0.0% | 98.2% |
| 无谓升级率 ↓ | 0.0% | 0.0% | 0.0% |
| 漏升级率 ↓ | 3.5% | 100.0% | 3.5% |
| 升级理由准确率 | 89.7% | 0.0% | 89.7% |

## 分组折明细

| fold | train_l1 | test | best_epoch | val_acc | train_s |
| --- | --- | --- | --- | --- | --- |
| 1 | 655 | 164 | 75 | 70.7% | 33.61 |
| 2 | 660 | 163 | 136 | 71.4% | 37.54 |
| 3 | 656 | 167 | 143 | 77.5% | 42.2 |
| 4 | 644 | 175 | 29 | 77.0% | 35.97 |
| 5 | 645 | 175 | 142 | 78.5% | 32.23 |

## L1 纯模型混淆（金标→预测）

| 金标 | 预测 | 次数 |
| --- | --- | --- |
| call::eda.visualize | call::eda.distribution | 14 |
| call::dataset.inspect | call::dataset.preview | 11 |
| call::ml.explain | call::ml.evaluate | 6 |
| call::dataset.profile | call::dataset.preview | 6 |
| call::ml.predict | call::ml.evaluate | 6 |
| call::dataset.quality | call::eda.outlier | 6 |
| call::dataset.profile | call::dataset.schema | 6 |
| call::ml.detect_task | call::ml.train | 5 |
| call::dataset.list | chat | 4 |
| call::data.clean | call::ml.prepare | 4 |
| call::workflow.build_and_run | call::workflow.create | 4 |
| call::data.merge | call::eda.correlation | 4 |
| call::dataset.preview | call::dataset.inspect | 4 |
| call::dataset.schema | call::ml.detect_task | 4 |
| call::dataset.inspect | call::dataset.schema | 4 |

## 误差互补性（L1 vs 词法基线，整条 route）

| 组合 | 条数 |
| --- | --- |
| 两者都对 | 567 |
| 仅词法基线对 | 114 |
| 仅 L1 对 | 49 |
| 两者都错 | 114 |

L1 补上了词法基线 **30.1%** 的错误；最佳单模型 **80.7%**；完美选择器上限 **86.5%**。

> 「完美选择器上限」是**事后**统计（逐条取两者中较优），不是可达指标；它的意义只是判断「融合两个模型有没有空间」。对比对象是**已套 L0 规则的部署形态基线**。

## 口径说明

- **purpose**：回答「L1 神经模型是否超过词法基线」；两个口径并列：纯模型看语义能力，+规则看线上质量。
- **fairness**：同一 group_folds（同 seed/折数/group_key）；同一次运行内重算词法基线以消除 sklearn 版本差异。
- **no_test_peek**：词表/权重/最佳 epoch 均只用训练折；测试折仅前向一次。
- **escalate_excluded**：升级样本不参与 L1 训练（由 L0 规则兜住），但仍出现在测试折中按其规则结果计分。