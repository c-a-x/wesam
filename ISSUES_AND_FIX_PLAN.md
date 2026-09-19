# 问题记录与修改计划（1pct ISIC WeSAM 低于 No-prompt）

本文记录当前 IFP-WeSAM 半监督流程中已确认的问题、支撑证据与拟定的修改方向。
所有结论均基于同一份公平协议：相同 train/val/test 划分、相同 `support_1pct.json` 索引、
相同随机种子（seed=1337），只改变被检验的那一个因素。

---

## 0. 背景：待解决的核心现象

| 方法 | ISIC 1pct test F1 | ISIC 1pct test IoU |
| --- | ---: | ---: |
| No prompt | **88.70** | **81.74** |
| WeSAM（IFP + 半监督） | 87.01 | 79.72 |
| WeSAM + dropout 0.5（当前最好） | 87.93 | — |

目标：让 WeSAM 在 1pct ISIC 上**稳定超过 88.70**，且该结论不能靠单次种子偶然得到。

---

## 问题 1：Teacher 与 Student 共用同一个 IFP 点（结构性耦合）

### 现象
Teacher 生成的伪标签并不独立于 Student 的提示通道。IFP 选点一旦偏移：
坏 prompt → Teacher 坏 mask → 被接受为 pseudo-label → Student 拟合 → 误差同向固化。

### 证据
- `adaptation.py:141` 处只生成一次 prompt：
  ```python
  prompts = prompt_generator(images_weak)
  ```
  该 `prompts` 同时喂给 **Anchor、Teacher（间接）、Student**。
- `labeled_prompt_mode='ifp'` 且 `labeled_gt_prompt_probability=0.0`
  导致 `_replace_labeled_prompts`（`adaptation.py:49`）永远命中 `continue`，
  **有 GT 的样本也不会替换成 GT 点**。
- 交叉对照实验（2×2 prompt transfer）说明耦合是决定性因素：

  | 训练权重 \ 评估 prompt | no prompt | IFP |
  | --- | ---: | ---: |
  | No-prompt 模型 θ∅ | **88.70 / 81.74** | 86.86 / 79.56 |
  | WeSAM 模型 θIFP | 79.57 / 71.00 | 86.88 / 79.52 |

  去掉 prompt 后 θIFP 掉 7.44，而 θ∅ 加 prompt 只掉 1.84 —— 能力高度绑在单一提示通道上。

### 拟定修改
- 让 Teacher 与 Student 走**不同的候选点**，而不是换掉 Teacher 的提示来源。
- 候选方案（按优先级）：
  1. Teacher 用 IFP top-1，Student 用 IFP top-2 / 抖动后的点，避免死记同一个点。
  2. Student 的 prompt 注入 dropout / jitter（当前唯一有正效果的方向）。
- **明确排除**：给 Teacher 换成无 prompt。该方案只在 ISIC 这类“大目标、居中、
  单显著物体”的数据集上成立（见问题 5），在 Polyp 等多目标数据集上不可用。

---

## 问题 2：`iterative_teacher_pseudo_labels` 在 `max_iters=1` 时是空转

### 现象
文档与直觉上认为 Teacher“自己重新选点”，但当前配置下它选出的点与 Student 完全相同。

### 证据
- `run_semisup_manifest.py:203` 设定：
  ```python
  prompt.iterative_pseudo.enabled, prompt.iterative_pseudo.max_iters = not fully_supervised, 1
  ```
- `iterative_teacher_pseudo_labels`（`ifp_alignment/prompt_generator.py:143`）逐轮逻辑为：
  1. `scores = self.score_maps(images)`
  2. 第 1 轮 `argmax(working_scores)` 取最高分 patch 作为正点
  3. `teacher_model.decode(...)` 出 mask
  4. 通过质量过滤后并入 `pseudo_masks`，并把该 mask 覆盖的 patch 设为 `-inf`
  5. 下一轮取剩余 patch 的 argmax
- 当 `max_iters=1` 时只走第 1 轮，其 `argmax` 与
  `_select_indices`（`max_positive_points=1, min_positive_points=1`）的 top-1
  **是同一个 patch**。

### 影响
- 多轮 mask 并集机制当前未生效。
- “Teacher 独立选点”这一解耦假设不成立（与问题 1 同一根因）。

### 拟定修改
- 若要保留多轮机制，需显式提高 `max_iters`（例如 3），并重新验证
  `teacher_iou_threshold`、`min_overlap_ratio`、`max_new_area_ratio` 的组合。
- 或明确文档化：当前 1-iter 配置下 Teacher 与 Student 同点，不要再假设二者已解耦。

---

## 问题 3：伪标签接受准则缺少对提示质量的约束

### 现象
`adaptation.py` 中对 Teacher 输出的接受条件过宽，坏提示产生的 mask 也可能被接受。

### 证据
- `adaptation.py:241`：
  ```python
  soft_mask = (soft_mask > 0.).float()
  ```
  仅做二值化，无额外质量过滤。
- 已有过滤位于 Teacher 侧参数：
  `teacher_iou_threshold=0.8`、`max_mask_area_ratio=0.6`、`max_iters=1`，
  但这些约束**不针对 IFP 点是否落在前景内**，无法阻止“点落入背景但 mask 自评为高质量”的情况。
- prompt 审计显示不同提示来源的分歧很大：
  IFP 前景命中率 0.991、DINO 0.998，但二者**相互一致率仅 0.329**
  （`output_current/prompt_policy_control/ISIC_1pct_prompt_audit.json`）。

### 拟定修改
- 增加与提示来源无关的质量/一致性过滤，例如：
  - IFP 点必须落在 no-prompt/Teacher mask 的前景区域内；
  - mask 面积比、与中心先验的偏差需在阈值内；
  - 视觉特征置信度低于阈值时直接丢弃。
- 所有比较实验必须统一 `prediction >= 0.5` 的二值化口径（此前存在 IFP-only
  用 0.5、CLIP/DINO 用 0.0 的不一致，已修正）。

---

## 问题 4：同提示一致性门槛（agreement gate）设计无效

### 现象
`min_student_teacher_iou` 门槛几乎不拒绝样本，加 gate 后指标没有改善。

### 证据
- gate 0.7 实验：F1 87.02，相对 baseline 仅 +0.01。
- 每个 batch 平均只拒绝约 0.10 个样本（拒绝比例约 9%）。
- 原因：Teacher 与 Student 共享同一个 IFP prompt，误差高度相关，
  **一致性高只说明一起错**，不代表伪标签正确。

### 拟定修改
- gate 必须建立在**不同提示来源**的预测之间，才有判别力。
- 单纯调高/调低 IoU 阈值不会改变结论，应先解决提示通道解耦（问题 1）。

---

## 问题 5：No-prompt 的强势依赖数据集几何先验，不可跨数据集推广

### 现象
No prompt 在 ISIC 1pct 上达到 88.70，但这来自数据集先验，而非通用分割能力。

### 证据
- ISIC 几何统计：train 前景面积 0.228±0.19、cx 0.482±0.058、cy 0.499±0.069，
  与居中圆盘的 Dice 达 0.708±0.211；test 面积 0.256±0.212、Dice 0.745±0.164。
- 无 prompt 时 prompt encoder 使用 `no_mask_embed`（`model.py` 中 `cfg.prompt=="none"`
  走 `concat_points=None`），decoder 依赖预训练形成的“显著、居中、大面积单物体”偏向。
- ISIC 恰好匹配该偏向，因此 no-prompt 强；Polyp 等多目标、小目标、位置不定场景不匹配。

### 拟定修改
- 不要把 no-prompt 作为 Teacher 的提示来源推广到其他数据集。
- 若要在 ISIC 上利用该先验，应作为**辅助分支或安全回退**，而非替换主提示通道。
- 跨数据集结论必须分别验证，不能由 ISIC 结果外推。

---

## 问题 6：单种子结论不具备统计显著性

### 现象
WeSAM 最好变体（dropout 0.5，87.93）与 no-prompt（88.70）差 0.77，
但该差距在单种子下无法判定显著。

### 证据
- 历史 3-seed 波动范围约 2 个百分点（85.25–87.31）。
- 0.77 的差距小于该波动，且 best checkpoint 集中在 epoch 1–2，
  早停策略本身会引入选择噪声。

### 拟定修改
- 所有关键比较至少跑 3 个种子（1337 / 2027 / 3407），报告 mean ± std。
- 统一早停到 epoch 1–2（各变体 best checkpoint 均落在该区间）。
- 最终结论只在独立 test 集上评估一次，并报告所选 checkpoint 的 epoch。

---

## 问题 7：可提升空间存在，但简单融合规则无法利用

### 现象
逐图事后最优选择存在明显上界，说明有空间，但现有简单规则全部失败。

### 证据
- 基于 per-image 指标的 oracle 上界：
  - No prompt：88.70
  - WeSAM 最好变体：87.93
  - **逐图 oracle（no-prompt vs IFP 取较优）上界：89.91**
- 简单规则实测（在已有预测上，按 per-image F1 统计）：

  | 规则 | F1 |
  | --- | ---: |
  | agreement > 0.5 时切换 IFP | 88.04 |
  | agreement > 0.8 | 88.23 |
  | agreement > 0.95 | 88.56 |
  | mask 并集 | 85.86 |
  | mask 交集 | 84.27 |

  （mask 级数值在原始分辨率上重算，与模型分辨率下的报告值存在口径差，
  仅用于判断趋势；oracle 上界直接由 `prediction_metrics.csv` 的 per-image F1 计算。）

- 结论：需要**可靠的选择/蒸馏机制**，而不是阈值调参或 mask 级算术。

### 拟定修改
- 优先实现“提示通道解耦 + 安全回退”的组合：
  - 主分支保留 IFP；
  - 增加一个不依赖 IFP 的辅助分支作为可靠性参照；
  - 仅当 IFP 分支满足多重准则时才采纳其结果，形成不小于参照分支的下界。
- 用 3 种子验证该机制是否稳定超过 88.70。

---

## 附：关键代码位置

| 功能 | 位置 |
| --- | --- |
| 单一 prompt 生成、Teacher/Student 共用 | `adaptation.py:141` |
| GT 点替换逻辑（当前不生效） | `adaptation.py:49` |
| 伪标签接受与 gate | `adaptation.py:241`–`255` |
| 多轮 Teacher 伪标签 | `ifp_alignment/prompt_generator.py:143` |
| IFP 选点 | `ifp_alignment/prompt_generator.py:96` |
| 无 prompt 分支 | `model.py:138`（`cfg.prompt == "none"`） |
| 1pct 训练参数 | `run_semisup_manifest.py:203` |
| prompt transfer 对照脚本 | `scripts/run_1pct_isic_prompt_transfer_control.sh` |

## 附：复现实验路径

- No prompt：`output_current/fair_ablation_1pct_1shot_seed1337/group2/1pct_isic_no_prompt/ISIC/metrics.csv`
- WeSAM baseline：`output_current/fair_ablation_1pct_fixed_seed1337/group2/1pct_isic_wesam/ISIC/metrics.csv`
- dropout 0.5：`output_current/fair_ablation_1pct_fixed_seed1337/group2/1pct_isic_wesam_dropout05/ISIC/metrics.csv`
- prompt transfer 对照：`output_current/fair_ablation_1pct_fixed_seed1337/prompt_transfer_control/`
