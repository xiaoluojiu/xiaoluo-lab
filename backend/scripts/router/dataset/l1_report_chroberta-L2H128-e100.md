# L1 基线对照（chroberta-L2H128-e100）

> 脚本 `scripts/router/train_l1.py`；口径 `eval_harness.py`；三层合成 `l1_data.assemble_route`。

- 后端：pretrained（`D:/xiaoluodataweb/xiaoluoAIdata/xiaoluo-lab/backend/.hf-cache/local/uer__chinese_roberta_L-2_H-128`）
- 设备：cuda（NVIDIA GeForce RTX 4050 Laptop GPU）
- 参数量：3,187,487（fp32 ≈ 12.75 MB）
- 推理：单条 3.61 ms·次⁻¹ / 整批 57.4 ms·844条⁻¹（= 0.068 ms·条⁻¹）
- 训练合计 134.76s / 5 折
- 标签空间：31 类；L1 可训练样本 815 条
- 契约来源：snapshot

## 指标对比

| 指标 | 词法基线(groupCV) | L1-chroberta-L2H128-e100 纯模型 | L1-chroberta-L2H128-e100+规则 |
| --- | --- | --- | --- |
| 样本数 | 844 | 844 | 844 |
| **整条决策准确率** | 80.7% | 69.4% | 75.9% |
| 决策类型准确率 | 97.8% | 91.0% | 97.3% |
| ① 可执行决策准确率 | 79.8% | 71.5% | 75.2% |
| ② 语义闸门准确率 | 96.6% | 42.4% | 89.8% |
| 工具选择准确率 | 80.1% | 75.3% | 75.3% |
| · 类型对时工具对 | 79.8% | 71.5% | 75.2% |
| chat F1 | 82.9% | 80.7% | 82.0% |
| call F1 | 98.9% | 95.3% | 98.7% |
| ask F1 | 87.3% | 0.0% | 81.5% |
| escalate F1 | 98.2% | 0.0% | 98.2% |
| 无谓升级率 ↓ | 0.0% | 0.0% | 0.0% |
| 漏升级率 ↓ | 3.5% | 100.0% | 3.5% |
| 升级理由准确率 | 89.7% | 0.0% | 89.7% |

## 分组折明细

| fold | train_l1 | test | best_epoch | val_acc | train_s |
| --- | --- | --- | --- | --- | --- |
| 1 | 655 | 164 | 96 | 79.7% | 27.58 |
| 2 | 660 | 163 | 96 | 74.8% | 27.26 |
| 3 | 656 | 167 | 97 | 90.1% | 27.64 |
| 4 | 644 | 175 | 100 | 76.2% | 26.53 |
| 5 | 645 | 175 | 98 | 83.3% | 25.75 |

## L1 纯模型混淆（金标→预测）

| 金标 | 预测 | 次数 |
| --- | --- | --- |
| call::data.aggregate | call::eda.describe | 8 |
| call::dataset.quality | call::eda.outlier | 6 |
| call::dataset.profile | call::eda.distribution | 6 |
| call::ml.evaluate | call::ml.explain | 6 |
| call::dataset.inspect | call::dataset.preview | 5 |
| call::dataset.quality | call::data.clean | 4 |
| call::dataset.inspect | call::dataset.schema | 4 |
| call::data.transform | call::dataset.schema | 4 |
| call::dataset.profile | call::eda.describe | 4 |
| call::workflow.create | call::workflow.build_and_run | 4 |
| call::dataset.profile | call::dataset.inspect | 4 |
| call::ml.explain | call::ml.evaluate | 4 |
| call::dataset.schema | call::dataset.list | 4 |
| call::eda.distribution | call::dataset.profile | 4 |
| call::eda.distribution | call::eda.outlier | 4 |

## 误差互补性（L1 vs 词法基线，整条 route）

| 组合 | 条数 |
| --- | --- |
| 两者都对 | 583 |
| 仅词法基线对 | 98 |
| 仅 L1 对 | 58 |
| 两者都错 | 105 |

L1 补上了词法基线 **35.6%** 的错误；最佳单模型 **80.7%**；完美选择器上限 **87.6%**。

> 「完美选择器上限」是**事后**统计（逐条取两者中较优），不是可达指标；它的意义只是判断「融合两个模型有没有空间」。对比对象是**已套 L0 规则的部署形态基线**。

## 口径说明

- **purpose**：回答「L1 神经模型是否超过词法基线」；两个口径并列：纯模型看语义能力，+规则看线上质量。
- **fairness**：同一 group_folds（同 seed/折数/group_key）；同一次运行内重算词法基线以消除 sklearn 版本差异。
- **no_test_peek**：词表/权重/最佳 epoch 均只用训练折；测试折仅前向一次。
- **escalate_excluded**：升级样本不参与 L1 训练（由 L0 规则兜住），但仍出现在测试折中按其规则结果计分。