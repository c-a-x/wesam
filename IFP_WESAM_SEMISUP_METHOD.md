# IFP-WeSAM 高置信半监督分割流程

本文说明当前迁移到 WeSAM 的 IFP 图文提示与高置信伪标签训练方法。它的目标不是让 IFP 直接输出分割 mask，而是用 IFP 找到更可靠的前景点，再由 SAM2 完成分割和半监督适配。

## 1. 总体结构

```text
少量 GT 图像
    ↓
训练 IFP 图文对齐投影头
    ↓
每张图：文本选择最高分 DINO patch
    ↓
patch 中心点作为 SAM2 正点 prompt
    ↓
Student / EMA Teacher SAM2
    ↓
GT 监督 + 高置信 Teacher 伪标签监督
    ↓
微调 Student LoRA
```

组件职责：

| 组件 | 作用 | 是否由分割 loss 直接更新 |
| --- | --- | --- |
| DINO + CLIP | 提供图像 patch 与文本特征 | 否，冻结 |
| IFP `img_head` / `txt_head` | 对齐图像 patch 与目标文本 | 仅在投影头阶段训练 |
| Student SAM2 | 输出训练预测并接收梯度 | 是 |
| EMA Teacher SAM2 | 为无标签图生成稳定伪标签 | 否，通过 Student 的 EMA 更新 |
| Anchor SAM2 | 原 WeSAM 的额外稳定约束 | 可选；当前严格实验中权重为 0 |

## 2. 数据划分

以 ISIC 1000 张实验为例：

```text
1000 张总图像
├─ 800 张训练图
│  ├─ 100 张有真实 GT mask
│  └─ 700 张无标签图
└─ 200 张独立验证图
```

训练图和验证图没有重合。100 张 GT 同时用于：

1. 训练医学 IFP 投影头。
2. 在 Student 适配阶段提供真实分割监督。

验证图不参与投影头训练、伪标签生成或反向传播。

## 3. IFP 投影头训练

对有 GT 的参考图像，DINO 提取 patch token，CLIP 提取目标文本特征。例如 ISIC 使用：

```text
"a dermoscopic photo of a skin lesion"
```

GT mask 用于标识哪些 patch 属于前景。训练的只有两个投影头：

```text
图像 patch token -> img_head -> 对齐特征
文本 token       -> txt_head -> 对齐特征
```

DINO 与 CLIP 本身保持冻结。训练完成后保存 `ifp_medical_foreground_best.pt`，其中包含 `img_head` 与 `txt_head`。

## 4. IFP 如何生成 SAM2 正点

运行时，对每张图像：

```text
DINO 提取所有 patch 特征
    ↓
img_head 投影 patch 特征
    ↓
与 txt_head 投影后的目标文本计算余弦相似度
    ↓
选最高相似度 patch
    ↓
将 patch 中心映射回图像坐标
    ↓
得到一个 SAM2 正点 prompt
```

因此 IFP 的输出是点，不是 mask。SAM2 接收 `图像 + 正点` 后才输出分割 mask。

## 5. Student 与 Teacher 训练

### 有 GT 的训练样本

```text
IFP 正点 + Student 强增强图像
    ↓
Student prediction
    ↓
与真实 GT mask 计算 Focal + Dice 监督损失
    ↓
更新 Student LoRA
```

有 GT 的样本不使用 Teacher 伪标签 loss，避免伪标签与真实标注冲突。

### 无 GT 的训练样本

```text
IFP 最高分 patch
    ↓
EMA Teacher 在弱增强图像上生成候选 mask
    ↓
通过质量筛选后得到 pseudo_label
    ↓
Student 在强增强图像上预测
    ↓
Student prediction 与 pseudo_label 计算 Teacher loss
    ↓
更新 Student LoRA
```

Student 是唯一接收反向传播和优化器更新的模型。

Teacher 不接收梯度，而是在每次 Student 更新后做 EMA：

```text
Teacher_new = ema_rate * Teacher_old + (1 - ema_rate) * Student_new
```

当前默认 `ema_rate=0.999`。Teacher 变化较慢，因此通常更平滑，但不保证任意 epoch 都优于 Student。

## 6. 高置信伪标签策略

当前稳定配置优先减少伪标签噪声：

```text
max_iters = 1
teacher_iou_threshold = 0.8
teacher_weight = 0.1
anchor_weight = 0.0
max_mask_area_ratio = 0.6
```

含义：

| 参数 | 作用 |
| --- | --- |
| `max_iters=1` | 每张无标签图只用最高分 patch 生成一个 mask，不做多轮 mask 并集。 |
| `teacher_iou_threshold=0.8` | Teacher 自己预测的 mask 质量低于 0.8 时，丢弃该伪标签。 |
| `teacher_weight=0.1` | 通过筛选的伪标签只以较小权重影响 Student。 |
| `anchor_weight=0.0` | 不让未经当前置信度筛选的 Anchor mask 影响梯度。 |
| `max_mask_area_ratio=0.6` | 过滤覆盖图像过大面积的异常 mask。 |

对无标签样本，当前损失可概括为：

```text
loss = rampup * 0.1 * (focal + dice + 0.1 * IoU_regression)
```

`rampup` 会在前 3 个 epoch 从小到大增加伪标签影响。对有 GT 的样本，损失只有真实监督项。

## 7. 为什么要平衡采样

800 张训练图中只有 100 张有 GT。若完全随机采样，GT 更新只占约 12.5%，容易被无标签伪标签淹没。

平衡采样将有 GT 样本的抽样概率设置为约 0.5。batch size 为 4 时，平均每个 batch 约有：

```text
2 张 GT 图像 + 2 张无标签图像
```

这使真实监督持续稳定 Student，而无标签图像用于扩大覆盖范围。

## 8. 验证与 checkpoint 选择

每个 epoch 结束后，使用同一批独立验证图分别评估：

```text
Student -> mIoU / F1
Teacher -> mIoU / F1
```

保存三类权重：

```text
best-student.pth  : 验证 mIoU 历史最高的 Student
best-teacher.pth  : 验证 mIoU 历史最高的 Teacher
best-overall.pth  : 两者中验证 mIoU 历史最高的模型
```

checkpoint 中包含：

```text
model
source_model  # student 或 teacher
epoch
mean_iou
mean_f1
```

训练仅保留 `best-student.pth`。最终测试会重载该权重。

## 9. 指标提升的原因

该方法的提升通常来自以下组合，而不是单一模块：

1. 预训练 SAM2 已具备“正确点 -> 目标 mask”的基础能力。
2. IFP 将正点更可靠地放在目标内部，改善初始提示质量。
3. 少量 GT 在平衡采样下提供约一半训练更新，快速适配目标域。
4. 单轮、高阈值、低权重伪标签减少了错误 mask 对预训练能力的破坏。
5. Student 通过 LoRA 小范围适配，不需要从零训练整个 SAM2。
6. 学习率 warmup 结束后，适配效果常在第 2 至第 4 个 epoch 出现明显提升。

## 10. 结果解读限制

独立验证图不参与反向传播，因此模型不会直接记住它们；但每个 epoch 都在同一验证集上评估并选择最佳 checkpoint，验证指标会存在一定选择偏差。

要得到最终可报告的泛化结果，应：

1. 使用独立的最终测试集，只在训练与 checkpoint 选择完成后评估一次。
2. 使用多个随机 seed 重复训练，报告均值和标准差。
3. 比较 `best-student.pth` 与 `best-teacher.pth` 的最终测试集结果，而不是预设 Teacher 一定更好。

## 11. 关键代码

| 功能 | 位置 |
| --- | --- |
| IFP 选 patch、生成多轮伪标签 | `ifp_alignment/prompt_generator.py` |
| Student/Teacher 训练与 checkpoint 选择 | `adaptation.py` |
| 平衡半监督采样 | `datasets/ISIC.py` |
| ISIC 实验划分与参数 | `run_ifp_semisup_isic.py` |
| 验证 mIoU/F1 | `utils/eval_utils.py` |
