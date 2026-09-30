# WeSAM2-SemiSup 项目评审报告（research-review skill, 2026-09-30）

> 评审方式说明：外部 reviewer backend（Codex / Manual Review MCP）在当前环境未配置，
> 本次由主 Agent 担任资深审稿人（MICCAI/TMI 口径）执行本地深度评审，
> 并结合 2024-2026 顶会/顶刊文献检索（WebSearch，共 4 组查询 12 篇命中）。
> 所有文献建议均附出处链接；所有项目内结论均有 output_current 实验记录支撑。

---

## 0. 一句话结论

**当前不投 CVPR/ICCV（CCF-A）。主力目标 MICCAI 2027 / TMI / Medical Image Analysis（CCF-B），**
可行性高但有 2 个投稿前必须处理的硬伤和 1 个叙事重构任务。
**论文主线必须从"我提出了 IFP"转为"IFP 使能下的标注效率实证研究"**——
后者才是你真正的差异化资产（详见 §3）。

---

## 1. 项目评审（内部视角）

### 1.1 强项（评审意见：strong，远超同类投稿的平均线）

1. **实验体系完备度是顶级资产**：2 数据集 6 测试域 × 1shot/1pct/10pct/Full 四预算 × 三训练机制
   × 多 seed × 提示后端三件套消融，76+ 组完整评测，口径统一（seed 1337/3407 fair 协议）。
   MICCAI 审稿人极度看重这一点。
2. **反直觉且可辩护的核心发现**（全部同协议同 seed 验证）：
   - Polyp：10% 标注（74.71）> 100% 满额全监督（73.89±0.12）；1% 达满额 99%；
     ISIC：10%（81.54）> Full（79.38），+2.16。
   - 域分化跨 seed 稳健：难域（ETIS/CVC-300）Full 增益 +4~5，易域（Kvasir/ColonDB）
     Full 反退化 -2~-3.5（过拟合）。
   - 1shot 三机制全部有害（-5~-21），边界发现诚实且有分析价值。
   - ISIC 上 No-prompt 反超 WeSAM baseline（81.74 vs 79.72）——已知问题，需主动讨论。
3. 代码/文档/表格体系工程化程度高（comparison_examples_8 六张表 + 架构图文档）。

### 1.2 硬伤（投稿前必须处理，按严重度排序）

| # | 问题 | 影响 | 处置 |
|---|---|---|---|
| H1 | **核心指标故事依赖 seed 敏感性**：方案一（Prompt Dropout）在 1pct 多 seed 下均值翻负（ISIC -0.16 / Polyp -0.29），符号随 seed 翻转 | 主表不能用"方案一最优"叙事；三机制降级为消融即可（已在规划中） | 已完成：改写为"机制适用边界"分析线 |
| H2 | **两套协议并存**：主表（vs SOTA）是 seed 3407 + gt-train-prompt，消融全系是 seed 1337 + ifp-train-prompt，差 5~6 个点 | 审稿人必问"为什么主结果和消融不同配置" | 二选一：见 §4-P1 |
| H3 | **9 个 SOTA 对比中 6 个是手工汇总数字**（UA-MT/DTC/URPC/BCP/KnowSAM/MC-Net 的预测掩码在 ablation/1、3，指标未经当前代码重算） | "对比非自现"是 CCF-B 级别也会被打的点；尤其唯一接近的 H-SAM（自跑，可靠） ClinicDB-BCP（手工） | 一键复算脚本，1-2h，见 §4-P2 |
| H4 | **验证集=测试集**（val loader 绑 test_list，best epoch 在测试集上选模） | 严格审稿人会 challenge optimistic model selection | 加"固定 epoch 敏感性分析"（便宜）+ 正文声明 |
| H5 | **SOTA 对比表缺 2025-2026 新作**：RD-Net（CVPR 2025 半监督息肉）、FPGM（2025 自称 SOTA）不在表内 | 时效性被 challenge | 至少 related work + limitation 讨论；有余力复现 FPGM（开源） |
| H6 | **指标单一**：只有 Dice/IoU，无 HD95/边界精度；无参数量、训练/推理耗时对比 | 医学分割审稿惯例要边界指标；ECT-3DMedSAM 等以 HD95 -57.9% 为卖点 | HD95 可从已有自跑实验的 hd95/asd 明细补齐（H-SAM/CPC-SAM 记录里有） |

---

## 2. 相关工作定位（文献检索结果，必须进 related work）

| 工作 | 出处 | 与你的关系 | 对你的要求 |
|---|---|---|---|
| WeSAM (CVPR 2024) | 基线本身 | 你的直接前身 | 引用并明确增量（SAM2-LoRA + IFP 重构 + 预算系统研究） |
| VESSA (arXiv 2511.19759, 2025) | [链接](https://arxiv.org/html/2511.19759v3) | VLM+SSL teacher，template-guided prompt 生成假标签——**与你"IFP提示+半监督"结构最同构** | 必须引用 + 明确区分：VESSA 用 vision exemplar 匹配，你用图文对齐 patch 定位 |
| RFMedSAM 2 (arXiv 2502.02741, 2025) | [链接](https://arxiv.org/pdf/2502.02741) | SAM2 医学：UNet 自动生成 box/mask prompt + adapter 微调 + 92.3% upper bound | 引用（自动提示先例）；其 "upper bound with oracle prompt" 与你的 GT-oracle 实验正好对应 |
| SAM2-SGP (arXiv 2506.19658, 2025) | [链接](https://arxiv.org/pdf/2506.19658.pdf) | support-set guided prompting（support mask→伪mask→box）+ LoRA | 引用；对比"需要 mask 的 support guidance vs 你的免 mask 图文对齐" |
| FoB (arXiv 2603.21287, 2026) | [链接](https://arxiv.org/html/2603.21287) | **背景中心提示**：SAM 过分割问题，用 background prompt 约束，FSMIS | 引用 + 技术借鉴（见 §5-A） |
| INSID3 (arXiv 2603.28480, 2026) | [链接](https://arxiv.org/html/2603.28480) | training-free DINOv3 in-context 分割，ISIC +8.7 | 引用（DINOv3 密集特征可直接支撑 in-context） |
| GazeRefine (arXiv 2609.01310, 2026) | [链接](https://arxiv.org/html/2609.01310v1) | gaze as test-time prompt + DINOv3 原型精化（息肉域有实验） | 引用（future work：人机协同提示） |
| PALADIN (CVPRW 2026) | [链接](https://openaccess.thecvf.com/content/CVPR2026W/VAND/papers/Basaran_PALADIN_Prompt-Aligned_Localization_and_Anomaly_Detection_with_DINOv3_CVPRW_2026_paper.pdf) | DINOv3 patch→CLIP 文本空间的轻量对齐 adapters——**与 IFP 的 img_head 投影同构** | novelty 定位必须引用；IFP 的 novelty 需从"投影本身"移向"医学半监督场景的系统性应用与实证" |
| RD-Net (CVPR 2025) | [链接](https://openaccess.thecvf.com/content/CVPR2025/html/Li_Boost_the_Inference_with_Co-training_A_Depth-guided_Mutual_Learning_Framework_CVPR_2025_paper.html) | 半监督息肉分割 SOTA（depth-guided co-training） | SOTA 表空缺 → 至少 related work 讨论 |
| FPGM (arXiv 2508.06517, 2025) | [链接](https://arxiv.org/html/2508.06517v1/) | 频率先验 + 半监督息肉，自称 SOTA | 同上；开源可复现，见 §5-E |
| ECT-3DMedSAM (MIDL 2026) | [链接](https://proceedings.mlr.press/v315/huang26a.html) | SAM2 半监督 cross-teaching LoRA，HD95 -57.9% 卖点 | 引用（半监督+SAM 家族定位；边界指标重要性） |
| MCP-MedSAM (MELBA 2025) | [链接](https://mstaring.github.io/assets/pdf/2025_j_MELBA.pdf) | modality prompt + content prompt 轻量医学 SAM | 引用；modality-aware prompt 借鉴（见 §5-D） |
| Point Prompt Survey (MMAsia 2025 W) | [链接](https://dl.acm.org/doi/pdf/10.1145/3769748.3773346) | SAM 点提示综述（医学部分列 Self-Prompt SAM/CycleSAM） | related work 定位索引 |

---

## 3. 叙事重构（这是比补实验更重要的事）

**当前叙事风险**：IFP = DINOv3 patch × CLIP 文本投影出的一个前景点。
在 PALADIN（同构 CMAA）、VESSA（同构 template）、SAM2-SGP（同构 support guidance）之后，
单个投影头撑不起顶会 novelty。

**建议主线（三层递进）**：
1. **使能层**：IFP——无需 mask 的弱监督在线提示，证明它在医学半监督里可用且稳
   （但明确承认与 PALADIN CMAA 的技术同源性，引用之）。
2. **实证层（真正贡献）**：完整的标注效率曲线——1shot/1pct/10pct/Full × 2 数据集 × 6 域，
   三个核心实证结论：
   (a) **Full supervision 不再是上界**：两数据集同协议下 10% 半监督 > Full；
   (b) **预算-收益的域分化**：难域吃标注（+4~5）、易域过拟合（-2~-3.5），跨 seed 稳健；
   (c) **机制适用边界**：三机制 1shot 全负、1pct seed 敏感、10pct 弱正——
   把"负面结果"包装成对社区的实证警告（审稿人尊重这个）。
3. **分析层**：prompt 质量天花板（GT-oracle 上限实验：Polyp 81.49/ISIC 82.12）+
   支持集规模敏感性（13 vs 130 → ±1.2，与域相关：ISIC 上 130 更好、Polyp 上 13 更好）。

**标题示例**："Is Full Supervision the Ceiling? Annotation Efficiency of IFP-Prompted
Semi-Supervised SAM2 in Medical Segmentation"。

---

## 4. 投稿前优先级 TODO（作者可执行，按 acceptance lift / GPU 成本排序）

| 优先级 | 动作 | 成本 | 依据 |
|---|---|---|---|
| P1 | **协议统一决策**：论文主协议定为 "seed 1337 或 3407 + labeled-prompt-mode=gt" 单一路线；若选 1337+gt，补 6 实验（三预算 × 2 数据集，~10h 两卡一夜） | 10 GPU-h | H2；消融数据可复用，只需主表层重跑 |
| P2 | **复算 6 个手工 SOTA**：ablation/1、3 的 pred_masks 对 GT 重算 Dice/IoU（改 /tmp 已有评测脚本即可） | 1-2 CPU-h | H3 |
| P3 | **补 HD95 指标** + 参数量/显存/推理速度表 | 低（H-SAM 等自跑实验已有 hd95 记录） | H6；ECT-3DMedSAM 以 HD95 为卖点 |
| P4 | **固定 epoch 敏感性分析**（每实验报告 epoch=10 数字 vs best-epoch 数字） | 低 | H4 |
| P5 | 复现 FPGM（开源）进 SOTA 表，或在 limitation 讨论 RD-Net/FPGM | 4-8 GPU-h（若复现） | H5 |
| P6 | 可选：ISIC Full 补 1337/2027 多 seed | 5 GPU-h | Full 结论目前单 seed（已计划） |

---

## 5. 可借鉴的顶会技术点（全部有文献出处，按"预期增益 × 实现成本"排序）

### A. 背景/负点提示（FoB, arXiv 2603.21287）——最推荐
- **依据**：SAM 系在医学图像上的系统性问题是**过分割**（模糊解剖边界）。FoB 证明
  background-centric prompt 能约束 over-segmentation。Polyp 数据上 IFP 只给前景 top-1 点，
  边界模糊时极易淹掉背景。
- **实现**：IFP 增加一个 background 分支（文本 "background tissue"，取 top-1 背景点作为
  negative point 一并喂 SAM2），对齐阶段加一路对比损失。
- **成本**：1 个生成器模块 + 对齐重训 ~30 分钟 + 重跑 3 个预算实验 ~2-4 GPU-h。
- **与故事契合**：直接呼应 FoB，且可能解开 Polyp 易域（Kvasir/ColonDB）的退化之谜。

### B. top-k 多点提示（SAM2 原生支持多点；密度图选点）
- **依据**：INSID3/PALADIN 都证明 DINOv3 密集 patch 特征可直接做密集对应；FoB 用
  "prompt localization" 视角。IFP 目前只用 top-1 单点。
- **实现**：推理时将 top-1 改为 top-k（k=3~5）或相似度密度图的质心；半监督训练管线不变。
- **成本**：几小时改代码 + 1 个实验验证（几乎零风险，失败也可作为分析）。
- **预期**：对 ETIS/CVC-300 难域可能有增益；single-point 依赖度下降，配合方案一有协同。

### C. 视觉原型提示（text-free 变体，DINOv3 in-context）
- **依据**：SAM2-SGP/INSID3/GazeRefine 都走 "support mask/标志 → DINO 空间原型 → 定位"。
- **实现**：IFP 加一个 mask-free 变体：用 support 集 GT mask 内的 DINOv3 patch 均值做原型，
  与 query patch 相似度选点（**替代 CLIP 文本通路**）。
- **价值**：这直接补齐你论文的"提示来源"三元对照（图文对齐 IFP / 视觉原型 / GT oracle）——
  point_pvt 系列实验已有雏形。审稿人问"没有文本行不行"时你就有数据。

### D. Modality-aware prompt（MCP-MedSAM, MELBA 2025）
- **依据**：MCP-MedSAM 证明 "modality prompt"（数据集标识向量）低成本提升多数据集表现。
- **实现**：IFP 文本模板前加 dataset tag 可学习 token；或者做一个"跨数据集 IFP 头迁移"
  轻实验（Polyp 训的头直接评 ISIC，测域泛化）——后者零训练成本，直接产出 Figure。

### E. 频率/结构先验增强（FPGM, arXiv 2508.06517）
- **依据**：频率先验是 2025 半监督息肉 SOTA 的核心 augmentation。
- **实现**：P5 复现 FPGM 入表；或在 IFP 选点前用频率过滤（息肉边缘带能量高）作为选点先验。
- **成本**：复现 4-8 GPU-h；或仅讨论。

### F. 不确定性加权伪标签（半监督 2025 标准件；ECT-3DMedSAM 思路）
- **依据**：teacher_weight=0.1 固定权重已显粗糙（1shot 崩、1pct seed 敏感与此有关）。
- **实现**：用 student 预测不确定性（Dice-margin）逐样本加权 teacher loss。
- **注意**：三机制多 seed 结论已翻负，此项优先级低，放 future work 更稳。

---

## 6. Claims Matrix（论文允许的断言）

| 断言 | 证据状态 | 允许度 |
|---|---|---|
| IFP 弱监督提示可替代人工提示驱动 SAM2-LoRA 半监督 | 全套实验 | ✅ 可写（主 claim） |
| 10% 半监督 > 100% 全监督（两数据集同协议） | Polyp +0.82（3-seed Full 73.89±0.12）；ISIC +2.16 | ✅ 可写（核心 claim） |
| 1% 标注达 Full 99% / 反超 Full | Polyp 73.15/73.89=99%；ISIC 80.23>79.38 | ✅ 可写 |
| 难域吃标注、易域过拟合的域分化 | 3 seed 一致 | ✅ 可写（分析 claim） |
| 方案一优于其他机制 | 多 seed 均值略负 | ❌ 不可写；只能写"机制适用边界" |
| 1shot 可用 | 三机制全负 | ❌ 不可写；写"1shot 下机制失效"边界 |
| IFP 投影头本身新颖 | PALADIN/VESSA 同构在先 | ❌ 不可写；引用并定位为"系统应用" |
| SOTA 超越 | 6/9 手工数字 | ⚠️ P2 复算后；不则表格加脚注 |

---

## 7. 最终建议路线

1. **本周**：P2 复算 → P3 HD95/效率表 → P4 敏感性分析（都不占 GPU）
2. **协议决策**（P1）：建议主协议 = **seed 1337 + labeled-prompt-mode=gt**（与消融同 seed，
   只补主表层 6 个实验，一夜跑完）
3. **投稿**：MICCAI 2027（主）+ TMI/MedIA（备）；叙事按 §3 重构
4. **可选加分**：借鉴点 A（背景点提示）预期是下一个 paper 的单点，或作为
   "future work + 初步实验"写进 discussion

（本报告由本地评审路径生成；外部 Codex/manual reviewer 未配置。
如需多模型交叉评审，可后续配置 MCP 后补跑。）
