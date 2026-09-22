# L1 基线对照（chroberta-L2H128）

> 脚本 `scripts/router/train_l1.py`；口径 `eval_harness.py`；三层合成 `l1_data.assemble_route`。

- 后端：pretrained（`D:/xiaoluodataweb/xiaoluoAIdata/xiaoluo-lab/backend/.hf-cache/local/uer__chinese_roberta_L-2_H-128`）
- 设备：cuda（NVIDIA GeForce RTX 4050 Laptop GPU）
- 参数量：3,187,487（fp32 ≈ 12.75 MB）
- 推理：单条 3.41 ms·次⁻¹ / 整批 65.8 ms·844条⁻¹（= 0.078 ms·条⁻¹）
- 训练合计 40.11s / 5 折
- 标签空间：31 类；L1 可训练样本 815 条
- 契约来源：snapshot

## 指标对比

| 指标 | 词法基线(groupCV) | L1-chroberta-L2H128 纯模型 | L1-chroberta-L2H128+规则 |
| --- | --- | --- | --- |
| 样本数 | 844 | 844 | 844 |
| **整条决策准确率** | 80.7% | 61.5% | 67.8% |
| 决策类型准确率 | 97.8% | 90.3% | 96.1% |
| ① 可执行决策准确率 | 79.8% | 62.8% | 66.2% |
| ② 语义闸门准确率 | 96.6% | 44.1% | 91.5% |
| 工具选择准确率 | 80.1% | 66.2% | 66.2% |
| · 类型对时工具对 | 79.8% | 62.8% | 66.2% |
| chat F1 | 82.9% | 72.2% | 75.4% |
| call F1 | 98.9% | 95.0% | 98.0% |
| ask F1 | 87.3% | 0.0% | 77.6% |
| escalate F1 | 98.2% | 0.0% | 98.2% |
| 无谓升级率 ↓ | 0.0% | 0.0% | 0.0% |
| 漏升级率 ↓ | 3.5% | 100.0% | 3.5% |
| 升级理由准确率 | 89.7% | 0.0% | 89.7% |

## 分组折明细

| fold | train_l1 | test | best_epoch | val_acc | train_s |
| --- | --- | --- | --- | --- | --- |
| 1 | 655 | 164 | 28 | 69.2% | 8.36 |
| 2 | 660 | 163 | 29 | 61.9% | 8.01 |
| 3 | 656 | 167 | 30 | 78.2% | 8.0 |
| 4 | 644 | 175 | 27 | 65.9% | 7.95 |
| 5 | 645 | 175 | 28 | 75.7% | 7.79 |

## L1 纯模型混淆（金标→预测）

| 金标 | 预测 | 次数 |
| --- | --- | --- |
| call::dataset.inspect | call::dataset.preview | 8 |
| call::data.clean | call::dataset.quality | 8 |
| call::dataset.quality | call::eda.outlier | 8 |
| call::data.aggregate | call::eda.describe | 8 |
| call::ml.evaluate | call::ml.explain | 8 |
| call::data.aggregate | call::data.merge | 8 |
| call::data.transform | call::dataset.schema | 6 |
| call::eda.outlier | call::dataset.quality | 6 |
| call::dataset.profile | call::dataset.inspect | 6 |
| call::ml.explain | call::ml.evaluate | 6 |
| call::dataset.preview | call::dataset.inspect | 6 |
| call::dataset.quality | call::data.clean | 5 |
| call::workflow.list | call::workflow.inspect | 5 |
| call::eda.distribution | call::dataset.profile | 5 |
| call::dataset.profile | call::eda.describe | 4 |

## 误差互补性（L1 vs 词法基线，整条 route）

| 组合 | 条数 |
| --- | --- |
| 两者都对 | 531 |
| 仅词法基线对 | 150 |
| 仅 L1 对 | 41 |
| 两者都错 | 122 |

L1 补上了词法基线 **25.1%** 的错误；最佳单模型 **80.7%**；完美选择器上限 **85.5%**。

> 「完美选择器上限」是**事后**统计（逐条取两者中较优），不是可达指标；它的意义只是判断「融合两个模型有没有空间」。对比对象是**已套 L0 规则的部署形态基线**。

## 口径说明

- **purpose**：回答「L1 神经模型是否超过词法基线」；两个口径并列：纯模型看语义能力，+规则看线上质量。
- **fairness**：同一 group_folds（同 seed/折数/group_key）；同一次运行内重算词法基线以消除 sklearn 版本差异。
- **no_test_peek**：词表/权重/最佳 epoch 均只用训练折；测试折仅前向一次。
- **escalate_excluded**：升级样本不参与 L1 训练（由 L0 规则兜住），但仍出现在测试折中按其规则结果计分。