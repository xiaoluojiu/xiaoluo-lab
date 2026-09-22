# L1 基线对照（chroberta-L4H256）

> 脚本 `scripts/router/train_l1.py`；口径 `eval_harness.py`；三层合成 `l1_data.assemble_route`。

- 后端：pretrained（`D:/xiaoluodataweb/xiaoluoAIdata/xiaoluo-lab/backend/.hf-cache/local/uer__chinese_roberta_L-4_H-256`）
- 设备：cuda（NVIDIA GeForce RTX 4050 Laptop GPU）
- 参数量：8,773,663（fp32 ≈ 35.09 MB）
- 推理：单条 4.19 ms·次⁻¹ / 整批 82.4 ms·844条⁻¹（= 0.098 ms·条⁻¹）
- 训练合计 196.14s / 5 折
- 标签空间：31 类；L1 可训练样本 815 条
- 契约来源：snapshot

## 指标对比

| 指标 | 词法基线(groupCV) | L1-chroberta-L4H256 纯模型 | L1-chroberta-L4H256+规则 |
| --- | --- | --- | --- |
| 样本数 | 844 | 844 | 844 |
| **整条决策准确率** | 80.7% | 73.0% | 79.5% |
| 决策类型准确率 | 97.8% | 91.1% | 97.9% |
| ① 可执行决策准确率 | 79.8% | 75.2% | 78.8% |
| ② 语义闸门准确率 | 96.6% | 44.1% | 91.5% |
| 工具选择准确率 | 80.1% | 79.0% | 79.0% |
| · 类型对时工具对 | 79.8% | 75.2% | 78.8% |
| chat F1 | 82.9% | 82.5% | 85.2% |
| call F1 | 98.9% | 95.4% | 99.0% |
| ask F1 | 87.3% | 0.0% | 85.7% |
| escalate F1 | 98.2% | 0.0% | 98.2% |
| 无谓升级率 ↓ | 0.0% | 0.0% | 0.0% |
| 漏升级率 ↓ | 3.5% | 100.0% | 3.5% |
| 升级理由准确率 | 89.7% | 0.0% | 89.7% |

## 分组折明细

| fold | train_l1 | test | best_epoch | val_acc | train_s |
| --- | --- | --- | --- | --- | --- |
| 1 | 655 | 164 | 81 | 75.9% | 39.48 |
| 2 | 660 | 163 | 95 | 81.0% | 39.53 |
| 3 | 656 | 167 | 81 | 85.9% | 40.04 |
| 4 | 644 | 175 | 66 | 86.5% | 39.23 |
| 5 | 645 | 175 | 66 | 84.7% | 37.86 |

## L1 纯模型混淆（金标→预测）

| 金标 | 预测 | 次数 |
| --- | --- | --- |
| call::data.aggregate | call::eda.describe | 12 |
| call::ml.evaluate | call::ml.predict | 7 |
| call::dataset.profile | call::eda.distribution | 6 |
| call::ml.detect_task | call::ml.train | 6 |
| call::data.filter | call::data.aggregate | 5 |
| call::data.filter | call::data.clean | 5 |
| call::dataset.inspect | call::dataset.schema | 4 |
| call::dataset.profile | call::eda.describe | 4 |
| call::dataset.list | chat | 4 |
| call::dataset.quality | call::eda.outlier | 4 |
| call::eda.describe | call::dataset.inspect | 4 |
| call::dataset.schema | call::dataset.profile | 4 |
| call::eda.distribution | call::dataset.profile | 4 |
| call::dataset.preview | call::dataset.inspect | 4 |
| call::eda.describe | call::data.aggregate | 4 |

## 误差互补性（L1 vs 词法基线，整条 route）

| 组合 | 条数 |
| --- | --- |
| 两者都对 | 601 |
| 仅词法基线对 | 80 |
| 仅 L1 对 | 70 |
| 两者都错 | 93 |

L1 补上了词法基线 **42.9%** 的错误；最佳单模型 **80.7%**；完美选择器上限 **89.0%**。

> 「完美选择器上限」是**事后**统计（逐条取两者中较优），不是可达指标；它的意义只是判断「融合两个模型有没有空间」。对比对象是**已套 L0 规则的部署形态基线**。

## 口径说明

- **purpose**：回答「L1 神经模型是否超过词法基线」；两个口径并列：纯模型看语义能力，+规则看线上质量。
- **fairness**：同一 group_folds（同 seed/折数/group_key）；同一次运行内重算词法基线以消除 sklearn 版本差异。
- **no_test_peek**：词表/权重/最佳 epoch 均只用训练折；测试折仅前向一次。
- **escalate_excluded**：升级样本不参与 L1 训练（由 L0 规则兜住），但仍出现在测试折中按其规则结果计分。