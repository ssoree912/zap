#!/usr/bin/env python3
"""Build milebench_with_group_summary.csv from eval.json files."""
import json, os, csv
from pathlib import Path

OUTPUTS = Path("/workspace/zap/experiments/EXP-20260430-003-progressive/outputs")
OUT_CSV = OUTPUTS / "milebench_student_with_group_summary.csv"

# ── metadata ────────────────────────────────────────────────────────────────
TASK_META = {
    # dataset_key: (category, task_code, task_name, task_dataset_name, metric)
    "ActionLocalization":     ("Temporal Multi-image", "T-1", "Action Understanding and Prediction", "Action Localization",           "Accuracy"),
    "ActionPrediction":       ("Temporal Multi-image", "T-1", "Action Understanding and Prediction", "Action Prediction",             "Accuracy"),
    "ActionSequence":         ("Temporal Multi-image", "T-1", "Action Understanding and Prediction", "Action Sequence",               "Accuracy"),
    "MovingAttribute":        ("Temporal Multi-image", "T-2", "Object and Scene Understanding",       "Moving Attribute",              "Accuracy"),
    "ObjectExistence":        ("Temporal Multi-image", "T-2", "Object and Scene Understanding",       "Object Existence",              "Accuracy"),
    "ObjectInteraction":      ("Temporal Multi-image", "T-2", "Object and Scene Understanding",       "Object Interaction",            "Accuracy"),
    "ObjectShuffle":          ("Temporal Multi-image", "T-2", "Object and Scene Understanding",       "Object Shuffle",                "Accuracy"),
    "EgocentricNavigation":   ("Temporal Multi-image", "T-3", "Visual Navigation and Spatial Localization", "Egocentric Navigation",  "Accuracy"),
    "MovingDirection":        ("Temporal Multi-image", "T-3", "Visual Navigation and Spatial Localization", "Moving Direction",       "Accuracy"),
    "CharacterOrder":         ("Temporal Multi-image", "T-4", "Counterfactual Reasoning and State Change", "Character Order",         "Accuracy"),
    "CounterfactualInference":("Temporal Multi-image", "T-4", "Counterfactual Reasoning and State Change", "Counterfactual Inference","Accuracy"),
    "SceneTransition":        ("Temporal Multi-image", "T-4", "Counterfactual Reasoning and State Change", "Scene Transition",        "Accuracy"),
    "StateChange":            ("Temporal Multi-image", "T-4", "Counterfactual Reasoning and State Change", "State Change",            "Accuracy"),
    "MultiModalQA":           ("Semantic Multi-image", "S-1", "Knowledge Grounded QA",   "Complex Multimodal QA",                    "Accuracy"),
    "TQA":                    ("Semantic Multi-image", "S-1", "Knowledge Grounded QA",   "Textbook QA",                              "Accuracy"),
    "WebQA":                  ("Semantic Multi-image", "S-1", "Knowledge Grounded QA",   "Webpage QA",                               "Accuracy"),
    "WikiVQA":                ("Semantic Multi-image", "S-1", "Knowledge Grounded QA",   "Long Text with Images QA",                 "Accuracy"),
    "DocVQA":                 ("Semantic Multi-image", "S-2", "Text-Rich Images QA",     "Document QA",                              "Accuracy"),
    "OCR-VQA":                ("Semantic Multi-image", "S-2", "Text-Rich Images QA",     "OCR QA",                                   "Accuracy"),
    "SlideVQA":               ("Semantic Multi-image", "S-2", "Text-Rich Images QA",     "Slide QA",                                 "Accuracy"),
    "CLEVR-Change":           ("Semantic Multi-image", "S-3", "Visual Relation Inference","Visual Change Captioning",                 "ROUGE-L"),
    "IEdit":                  ("Semantic Multi-image", "S-3", "Visual Relation Inference","Visual Relationship Expressing",           "ROUGE-L"),
    "Spot-the-Diff":          ("Semantic Multi-image", "S-3", "Visual Relation Inference","Visual Change Captioning",                 "ROUGE-L"),
    "ALFRED":                 ("Semantic Multi-image", "S-4", "Dialogue",                "Conversational Embodied Dialogue",          "ROUGE-L"),
    "MMCoQA":                 ("Semantic Multi-image", "S-4", "Dialogue",                "Multimodal Dialogue",                      "Accuracy"),
    "ImageNeedleInAHaystack": ("Needle In A Haystack", "NH", "Needle In A Haystack",    "Image Needle In A Haystack",               "Accuracy"),
    "TextNeedleInAHaystack":  ("Needle In A Haystack", "NH", "Needle In A Haystack",    "Text Needle In A Haystack",                "Accuracy"),
    "GPR1200":                ("Image Retrieval",      "IR", "Image Retrieval",          "Image Retrieval",                          "Accuracy"),
}

# task_code → datasets in order (for averaging)
TASK_CODE_DATASETS = {}
for ds, (cat, tc, tn, tdn, m) in TASK_META.items():
    TASK_CODE_DATASETS.setdefault(tc, []).append(ds)

# task_code average metric (mixed S-4)
TASK_CODE_METRIC = {
    "T-1": "Accuracy", "T-2": "Accuracy", "T-3": "Accuracy", "T-4": "Accuracy",
    "S-1": "Accuracy", "S-2": "Accuracy", "S-3": "ROUGE-L",
    "S-4": "Accuracy+ROUGE-L", "NH": "Accuracy", "IR": "Accuracy",
}
TASK_CODE_CATEGORY = {
    "T-1": "Temporal Multi-image", "T-2": "Temporal Multi-image",
    "T-3": "Temporal Multi-image", "T-4": "Temporal Multi-image",
    "S-1": "Semantic Multi-image", "S-2": "Semantic Multi-image",
    "S-3": "Semantic Multi-image", "S-4": "Semantic Multi-image",
    "NH": "Needle In A Haystack", "IR": "Image Retrieval",
}
TASK_CODE_NAME = {
    "T-1": "Action Understanding and Prediction",
    "T-2": "Object and Scene Understanding",
    "T-3": "Visual Navigation and Spatial Localization",
    "T-4": "Counterfactual Reasoning and State Change",
    "S-1": "Knowledge Grounded QA", "S-2": "Text-Rich Images QA",
    "S-3": "Visual Relation Inference", "S-4": "Dialogue",
    "NH": "Needle In A Haystack", "IR": "Image Retrieval",
}

# ── experiments to process ───────────────────────────────────────────────────
EXPERIMENTS = [
    # (dir_name, method_label, keep_ratio)
    ("milebench_keep010", "OneVision-Student", 0.10),
    ("milebench_keep025", "OneVision-Student", 0.25),
    ("milebench_keep050", "OneVision-Student", 0.50),
    ("llava15_milebench_keep010", "LLaVA15-Student", 0.10),
    ("llava15_milebench_keep020", "LLaVA15-Student", 0.20),
    ("llava15_milebench_keep050", "LLaVA15-Student", 0.50),
]


def read_score(eval_path: Path, metric: str) -> float | None:
    try:
        d = json.loads(eval_path.read_text())
    except Exception:
        return None
    if metric == "Accuracy":
        return d.get("Accuracy")
    elif metric == "ROUGE-L":
        return d.get("Rouge-L f")
    return None


def read_n_samples(pred_path: Path) -> int:
    try:
        return len(json.loads(pred_path.read_text()))
    except Exception:
        return 0


rows = []

for dir_name, method, ratio in EXPERIMENTS:
    exp_dir = OUTPUTS / dir_name
    if not exp_dir.exists():
        print(f"[skip] {dir_name} not found")
        continue

    task_scores: dict[str, float] = {}  # dataset → score

    for dataset, (cat, tc, tn, tdn, metric) in TASK_META.items():
        eval_path = exp_dir / dataset / "eval.json"
        pred_path = exp_dir / dataset / "pred.json"
        if not eval_path.exists():
            continue
        score = read_score(eval_path, metric)
        if score is None:
            continue
        n_samples = read_n_samples(pred_path)
        task_scores[dataset] = score
        rows.append({
            "level": "dataset",
            "category": cat,
            "task_code": tc,
            "task_name": tn,
            "task_dataset_name": tdn,
            "dataset": dataset,
            "metric": metric,
            "method": method,
            "ratio": ratio,
            "score": round(score, 6),
            "score_percent": round(score * 100, 2),
            "n_samples": n_samples,
            "n_failures": "",
            "components": "",
        })

    # task_average rows
    for tc, ds_list in TASK_CODE_DATASETS.items():
        available = [d for d in ds_list if d in task_scores]
        if not available:
            continue
        avg = sum(task_scores[d] for d in available) / len(available)
        rows.append({
            "level": "task_average",
            "category": TASK_CODE_CATEGORY[tc],
            "task_code": tc,
            "task_name": TASK_CODE_NAME[tc],
            "task_dataset_name": "",
            "dataset": tc,
            "metric": TASK_CODE_METRIC[tc],
            "method": method,
            "ratio": ratio,
            "score": round(avg, 6),
            "score_percent": round(avg * 100, 2),
            "n_samples": "",
            "n_failures": "",
            "components": ";".join(available),
        })

fieldnames = [
    "level", "category", "task_code", "task_name", "task_dataset_name",
    "dataset", "metric", "method", "ratio", "score", "score_percent",
    "n_samples", "n_failures", "components",
]
with open(OUT_CSV, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

print(f"Saved {len(rows)} rows → {OUT_CSV}")
