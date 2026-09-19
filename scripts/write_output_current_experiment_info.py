#!/usr/bin/env python3
"""Write a concise experiment note into every output_current top-level folder."""

from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1] / "output_current"

NOTES = {
    "best_full_after_screen_20260820": "自动筛选后选出的最佳配置进行完整 ISIC/Polyp 训练与测试；用于确认小规模筛选结果能否在完整数据上复现。详见 selection.json、isic/、polyp/。",
    "best_full_wesam_20260820": "完整数据集上的最佳 WeSAM 配置复跑；包含 ISIC 和 Polyp 的训练、验证及跨域测试结果。",
    "clip_only": "CLIP-only 提示消融；使用 CLIP 文本/视觉提示，不使用完整 IFP/WeSAM 提示策略，主要用于比较提示来源。",
    "clip_semisupervised": "CLIP-only 半监督 ISIC 1% 实验；用于与 DINO-only、No prompt 等提示策略对照。",
    "dino_only": "DINO-only 直接 SAM2/提示实验；用于评估只使用 DINO 提示的效果。",
    "dino_semisupervised": "DINO-only 半监督 ISIC 1% 实验；用于和 IFP、CLIP-only、No prompt 比较。",
    "ifp_direct_sam2": "直接使用 IFP 点提示进行 SAM2 推理/评估；不代表完整 WeSAM 训练结果，主要用于测量提示质量。",
    "isic_anchor_fast_screen": "ISIC 快速筛选实验；测试 Anchor、点抖动、prompt dropout、置信度回退等策略，结果用于决定后续正式实验配置。",
    "isic_dropout_sweep": "ISIC 1% 完整 dropout 对比；mixed GT/IFP 下比较 dropout=0.05 与 0.1。",
    "isic_fast_screen": "ISIC 小规模快速筛选；包含 No prompt、原始 WeSAM、点抖动、dropout、置信度回退和 GT/IFP 混合等候选策略。",
    "isic_small_sweep": "ISIC 小训练子集筛选；比较 No prompt、mixed GT/IFP + dropout=0.3/0.1，属于 screening，不是正式全量结果。",
    "nested_budget_control_20260823": "严格嵌套 1%/10% 标注预算对照；每个 seed 先固定 10% 集合，再从其中取 1% 子集，两种预算共用同一 10% IFP 支持集，并跑 seed 1337/2027/3407。",
    "no_prompt": "No prompt 半监督基线；不向 SAM 提供点提示，用于衡量仅靠图像/半监督训练的基线表现。",
    "no_semisupervised": "No semi-supervised 基线；关闭半监督伪标签训练，主要用于消融半监督模块本身。",
    "operations": "实验运行维护目录；保存 watchdog、队列、取消任务和 GPU 调度日志，不是单独模型配置。",
    "point_p(gt)_vt(ifp)": "GT 点训练 + IFP 点测试的控制实验；用于分离训练提示来源和测试提示来源的影响。",
    "point_pvt(gt)": "GT 点提示控制/上限实验；使用真实 GT 点进行提示，主要用于估计提示正确时的性能上限。",
    "point_pvt(ifp)": "IFP 点提示控制实验；使用 IFP 点进行提示，主要用于评估自动提示下的推理性能。",
    "polyp_prompt_ablation": "Polyp 多 seed prompt/dropout 消融；包含 No prompt、原始 WeSAM、mixed GT/IFP + dropout=0/0.05/0.1，主要用于 Polyp 五数据集的配置对比。",
    "polyp_prompt_sampling_ablation_20260818": "Polyp 提示采样概率消融；比较不同 labeled prompt sampling probability 和 seed，分析 GT/IFP 提示混合比例影响。",
    "prompt_policy_control": "固定已训练 checkpoint 的提示来源控制；同一模型分别使用 IFP、DINO 和 No prompt，分离训练质量与提示质量。",
    "prompt_robustness_20260817": "ISIC 多 seed 提示鲁棒性实验；包含 No prompt、完整 WeSAM、IFP-only 等配置，主要用于计算随机种子波动。",
    "prompt_sampling_ablation_20260818": "ISIC 提示采样概率消融；比较不同 GT/IFP 混合概率及随机 seed。",
    "remaining_experiments": "ISIC/Polyp 剩余消融队列；包含 mixed GT/IFP 的 dropout=0/0.05/0.1 多 seed，以及 Anchor、Contrastive、Anchor+Contrastive 消融。",
    "reports": "结果汇总和报告文件；包含 Excel、提示点审计 CSV/JSON 及导出日志，不是训练输出。",
    "three_stage_ablation_20260818": "三阶段 ISIC 消融；系统比较 No prompt、原始 IFP、Anchor、不同 GT/IFP 混合概率和 seed。",
    "wesam": "历史正式 WeSAM 结果；包含 ISIC/Polyp 的 1%、10%、少量 GT 等训练和五数据集测试结果。",
    "wesam_mixed_dropout": "较早的 mixed GT/IFP + prompt dropout 实验目录；用于早期快速/正式 ISIC 结果。",
    "wesam_mixed_dropout_full": "完整 ISIC/Polyp 的 mixed GT/IFP + dropout=0.1 实验；包含 ISIC 1% 和 Polyp 五数据集结果。",
}


def main() -> None:
    count = 0
    for directory in sorted(path for path in ROOT.iterdir() if path.is_dir()):
        note = NOTES.get(directory.name)
        if note is None:
            note = "output_current 下的实验结果目录；请结合其中的 logs、dataset_summary.json、metrics.csv 或 test_metrics.csv 查看具体配置。"
        text = (
            f"目录：{directory}\n\n"
            f"实验说明：{note}\n\n"
            "说明：本文件是目录级索引。具体 seed、标注比例、最佳 epoch 和指标以目录内的日志及 metrics.csv/test_metrics.csv 为准。\n"
        )
        (directory / "EXPERIMENT_INFO.txt").write_text(text, encoding="utf-8")
        count += 1
    print(f"Wrote {count} experiment notes under {ROOT}")


if __name__ == "__main__":
    main()
