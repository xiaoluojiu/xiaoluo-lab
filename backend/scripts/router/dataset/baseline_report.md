# 词法基线 Router 报告

> 生成脚本 `backend/scripts/router/baseline_lexical.py`；口径 `eval_harness.py`。

| 指标 | majority(holdout) | tfidf_svc(holdout) | hybrid(holdout) | tfidf_svc(groupCV) | hybrid(groupCV) | abstain-top1(groupCV) | abstain-margin(groupCV) | hier(groupCV) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 样本数 | 132 | 132 | 132 | 844 | 844 | 844 | 844 | 844 |
| **整条决策准确率** | 0.8% | 87.9% | 89.4% | 70.3% | 78.4% | 75.1% | 69.5% | 76.1% |
| 决策类型准确率 | 87.9% | 93.9% | 95.5% | 90.3% | 95.6% | 90.6% | 80.2% | 94.3% |
| ① 可执行决策准确率 | 0.8% | 89.3% | 91.8% | 71.0% | 79.7% | 76.6% | 71.0% | 80.4% |
| ② 语义闸门准确率 | 0.0% | 70.0% | 60.0% | 66.1% | 66.1% | 83.1% | 79.7% | 30.5% |
| 工具选择准确率 | 0.8% | 91.8% | 91.8% | 76.6% | 80.1% | 76.9% | 71.3% | 80.8% |
| · 类型对时工具对 | 0.8% | 89.3% | 91.8% | 71.0% | 79.7% | 76.6% | 71.0% | 80.4% |
| chat F1 | 0.0% | 90.9% | 90.9% | 76.3% | 80.6% | 84.1% | 79.4% | 53.7% |
| call F1 | 93.5% | 96.6% | 97.5% | 95.2% | 97.9% | 95.0% | 88.5% | 97.0% |
| ask F1 | 0.0% | 50.0% | 90.9% | 0.0% | 87.3% | 85.7% | 83.6% | 88.9% |
| escalate F1 | 0.0% | 57.1% | 33.3% | 51.3% | 51.3% | 38.1% | 20.7% | 38.9% |
| 无谓升级率 ↓ | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 6.9% | 17.7% | 0.0% |
| 漏升级率 ↓ | 100.0% | 60.0% | 80.0% | 65.5% | 65.5% | 31.0% | 31.0% | 75.9% |
| 升级理由准确率 | 0.0% | 40.0% | 20.0% | 24.1% | 24.1% | 13.8% | 10.3% | 0.0% |

## 模型规模 / 速度

| 项 | 值 |
| --- | --- |
| vectorizer | TfidfVectorizer(char_wb, 1-4gram, sublinear) |
| classifier | LinearSVC(C=1.0) |
| n_features | 8326 |
| model_bytes | 3455490 |
| train_seconds | 0.17 |
| infer_ms_per_sample | 0.015 |
| learned_classes_full | 36 |
| learned_classes_abstain | 32 |

## 弃权阈值扫描（操作点）

训练集中被移出学习的「无工具可匹配」升级样本：15 条 (ambiguous, out_of_scope)

### 弃权信号 = top1

逐折校准出的 θ：[-0.569, -0.67, -0.362, -0.611, -0.546]

| θ 分位 | θ | escalate 精确率 | escalate 召回率 | escalate F1 | 弃权率 |
| --- | --- | --- | --- | --- | --- |
| 0% | -0.787 | 0.0% | 0.0% | 0.0% | 0.0% |
| 5% | -0.575 | 23.8% | 34.5% | 28.2% | 5.0% |
| 10% | -0.498 | 14.3% | 41.4% | 21.2% | 10.0% |
| 15% | -0.422 | 11.9% | 51.7% | 19.4% | 14.9% |
| 20% | -0.345 | 11.9% | 69.0% | 20.3% | 19.9% |
| 25% | -0.28 | 10.9% | 79.3% | 19.2% | 25.0% |
| 30% | -0.2 | 9.5% | 82.8% | 17.0% | 30.0% |
| 40% | -0.062 | 8.0% | 93.1% | 14.8% | 39.9% |
| 50% | 0.104 | 6.9% | 100.0% | 12.9% | 50.0% |

### 弃权信号 = margin

逐折校准出的 θ：[0.047, 0.091, 0.441, 0.229, 0.048]

| θ 分位 | θ | escalate 精确率 | escalate 召回率 | escalate F1 | 弃权率 |
| --- | --- | --- | --- | --- | --- |
| 0% | 0.0 | 0.0% | 0.0% | 0.0% | 0.0% |
| 5% | 0.04 | 11.9% | 17.2% | 14.1% | 5.0% |
| 10% | 0.074 | 9.5% | 27.6% | 14.2% | 10.0% |
| 15% | 0.124 | 8.7% | 37.9% | 14.2% | 14.9% |
| 20% | 0.19 | 8.3% | 48.3% | 14.2% | 19.9% |
| 25% | 0.251 | 7.6% | 55.2% | 13.3% | 25.0% |
| 30% | 0.31 | 8.3% | 72.4% | 14.9% | 30.0% |
| 40% | 0.424 | 6.8% | 79.3% | 12.6% | 39.9% |
| 50% | 0.582 | 5.9% | 86.2% | 11.1% | 50.0% |

## 拒识门（hier 第一段，专门二分类）

| 指标 | 值 |
| --- | --- |
| 样本数 | 844 |
| 精确率 | 95.0% |
| 召回率 | 100.0% |
| F1 | 97.5% |
| tp / fp / fn | 785 / 41 / 0 |

门＝「平台有工具能接吗」的二分类；与弃权阈值是两种不同的拒识实现。

## 分组 5 折明细

| fold | train | test | test 组数 |
| --- | --- | --- | --- |
| 1 | 680 | 164 | 89 |
| 2 | 681 | 163 | 89 |
| 3 | 677 | 167 | 88 |
| 4 | 669 | 175 | 88 |
| 5 | 669 | 175 | 88 |

## 按类目明细（groupCV 三列）

### tfidf_svc(groupCV)

| category | n | route_acc |
| --- | --- | --- |
| chat | 30 | 96.7% |
| escalate | 29 | 24.1% |
| follow_up | 12 | 41.7% |
| implicit | 16 | 62.5% |
| routing | 719 | 75.4% |
| slot_missing | 38 | 0.0% |

主要类型混淆（金标→预测）：ask→call×33；escalate→call×15；call→ask×14；call→chat×10；ask→chat×5；escalate→chat×2

升级识别率（按原因，kind 级）：ambiguous 17%(n=6)；conflict 80%(n=5)；multi_step 40%(n=5)；out_of_scope 22%(n=9)；param_dependency 25%(n=4)

### hybrid(groupCV)

| category | n | route_acc |
| --- | --- | --- |
| chat | 30 | 96.7% |
| escalate | 29 | 24.1% |
| follow_up | 12 | 41.7% |
| implicit | 16 | 68.8% |
| routing | 719 | 80.7% |
| slot_missing | 38 | 78.9% |

主要类型混淆（金标→预测）：escalate→call×17；call→chat×9；ask→call×5；escalate→chat×2；ask→chat×2；chat→ask×1

升级识别率（按原因，kind 级）：ambiguous 17%(n=6)；conflict 80%(n=5)；multi_step 40%(n=5)；out_of_scope 22%(n=9)；param_dependency 25%(n=4)

### abstain-top1(groupCV)

| category | n | route_acc |
| --- | --- | --- |
| chat | 30 | 96.7% |
| escalate | 29 | 13.8% |
| follow_up | 12 | 25.0% |
| implicit | 16 | 62.5% |
| routing | 719 | 77.7% |
| slot_missing | 38 | 76.3% |

主要类型混淆（金标→预测）：call→escalate×54；escalate→call×7；call→chat×6；ask→call×4；escalate→chat×2；ask→chat×2

升级识别率（按原因，kind 级）：ambiguous 67%(n=6)；conflict 100%(n=5)；multi_step 60%(n=5)；out_of_scope 67%(n=9)；param_dependency 50%(n=4)

### abstain-margin(groupCV)

| category | n | route_acc |
| --- | --- | --- |
| chat | 30 | 90.0% |
| escalate | 29 | 10.3% |
| follow_up | 12 | 25.0% |
| implicit | 16 | 62.5% |
| routing | 719 | 71.9% |
| slot_missing | 38 | 71.1% |

主要类型混淆（金标→预测）：call→escalate×138；escalate→call×7；call→chat×7；ask→call×4；ask→escalate×4；escalate→chat×2

升级识别率（按原因，kind 级）：ambiguous 50%(n=6)；conflict 100%(n=5)；multi_step 80%(n=5)；out_of_scope 44%(n=9)；param_dependency 100%(n=4)

### hier(groupCV)

| category | n | route_acc |
| --- | --- | --- |
| chat | 30 | 36.7% |
| escalate | 29 | 0.0% |
| follow_up | 12 | 50.0% |
| implicit | 16 | 68.8% |
| routing | 719 | 81.1% |
| slot_missing | 38 | 81.6% |

主要类型混淆（金标→预测）：escalate→call×22；chat→call×18；ask→call×6；chat→ask×1；call→ask×1

升级识别率（按原因，kind 级）：ambiguous 17%(n=6)；conflict 80%(n=5)；multi_step 20%(n=5)；out_of_scope 11%(n=9)；param_dependency 0%(n=4)

## 注意事项

- **leakage_warning**：样本级 holdout 与训练集共享 meta.template（仅槽位取值不同）⇒ 表层模型虚高。**泛化真值请看 *(groupCV) 三列**。
- **ask_is_symbolic**：`ask`（反问用户）不该由模型学：它是 required_params 的确定性函数。证据：747 条 call 样本里有 231 条 bound_dataset_id 也是 None ⇒「未绑定」不是判别特征。
- **abstain_rationale**：`ambiguous` / `out_of_scope` 按定义是「没有工具匹配得上」⇒ 用**弃权**处理，而不是当分类标签学；θ 逐折在内层 group CV 的 OOF 分数上取 binary F1 最优点。
- **escalate_is_bottleneck**：扩数据后工具选择已到 80%，剩下瓶颈是语义闸门；误判方向集中为 escalate→call。
- **eval_split_note**：eval split 仅 74 条且类别极不均衡 ⇒ 不作主指标（报告里已省略）。
- **label_space**：route 串由 eval_harness.route_label 派生，工具名零硬编码 ⇒ 不可能输出不存在的工具。
- **hard_negatives_note**：dataset/hard_negatives.jsonl 是工具混淆对诊断，未参与训练/评估。