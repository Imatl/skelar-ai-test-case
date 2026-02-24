import json
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"
DATASET_FILE = DATA_DIR / "dataset.json"
ANALYSIS_FILE = DATA_DIR / "analysis.json"
OUTPUT_FILE = DATA_DIR / "evaluation.json"

ALL_MISTAKES = ["ignored_question", "incorrect_info", "rude_tone", "no_resolution", "unnecessary_escalation"]


def load_data():
    with open(DATASET_FILE, "r", encoding="utf-8") as f:
        dataset = json.load(f)
    with open(ANALYSIS_FILE, "r", encoding="utf-8") as f:
        analysis = json.load(f)
    analysis_map = {item["id"]: item["analysis"] for item in analysis}
    return dataset, analysis_map


def compute_accuracy(gt_list, pred_list):
    correct = sum(1 for g, p in zip(gt_list, pred_list) if g == p)
    return correct / len(gt_list) if gt_list else 0


def compute_mae(gt_list, pred_list):
    total = sum(abs(g - p) for g, p in zip(gt_list, pred_list))
    return total / len(gt_list) if gt_list else 0


def compute_correlation(gt_list, pred_list):
    n = len(gt_list)
    if n < 2:
        return 0
    mean_g = sum(gt_list) / n
    mean_p = sum(pred_list) / n
    cov = sum((g - mean_g) * (p - mean_p) for g, p in zip(gt_list, pred_list))
    std_g = (sum((g - mean_g) ** 2 for g in gt_list)) ** 0.5
    std_p = (sum((p - mean_p) ** 2 for p in pred_list)) ** 0.5
    if std_g == 0 or std_p == 0:
        return 0
    return cov / (std_g * std_p)


def compute_mistake_metrics(dataset, analysis_map):
    metrics = {}
    for mistake in ALL_MISTAKES:
        tp = fp = fn = 0
        for d in dataset:
            pred = analysis_map.get(d["id"], {})
            gt_has = mistake in d["ground_truth"]["agent_mistakes"]
            pred_has = mistake in pred.get("agent_mistakes", [])
            if gt_has and pred_has:
                tp += 1
            elif not gt_has and pred_has:
                fp += 1
            elif gt_has and not pred_has:
                fn += 1

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        metrics[mistake] = {
            "tp": tp, "fp": fp, "fn": fn,
            "precision": round(precision, 3),
            "recall": round(recall, 3),
            "f1": round(f1, 3),
        }
    return metrics


def evaluate():
    dataset, analysis_map = load_data()

    gt_intents = [d["ground_truth"]["intent"] for d in dataset]
    pred_intents = [analysis_map.get(d["id"], {}).get("intent", "other") for d in dataset]

    gt_satisfaction = [d["ground_truth"]["satisfaction"] for d in dataset]
    pred_satisfaction = [analysis_map.get(d["id"], {}).get("satisfaction", "neutral") for d in dataset]

    gt_quality = [d["ground_truth"]["quality_score"] for d in dataset]
    pred_quality = [analysis_map.get(d["id"], {}).get("quality_score", 3) for d in dataset]

    intent_acc = compute_accuracy(gt_intents, pred_intents)
    sat_acc = compute_accuracy(gt_satisfaction, pred_satisfaction)
    quality_mae = compute_mae(gt_quality, pred_quality)
    quality_corr = compute_correlation(gt_quality, pred_quality)
    quality_exact = compute_accuracy(gt_quality, pred_quality)
    quality_within_1 = sum(1 for g, p in zip(gt_quality, pred_quality) if abs(g - p) <= 1) / len(gt_quality)

    hidden_dialogs = [d for d in dataset if d["metadata"]["has_hidden_dissatisfaction"]]
    hidden_detected = sum(
        1 for d in hidden_dialogs
        if analysis_map.get(d["id"], {}).get("satisfaction") == "unsatisfied"
    )
    hidden_rate = hidden_detected / len(hidden_dialogs) if hidden_dialogs else 0

    mistake_metrics = compute_mistake_metrics(dataset, analysis_map)
    avg_f1 = sum(m["f1"] for m in mistake_metrics.values()) / len(mistake_metrics)

    print("=" * 65)
    print("EVALUATION RESULTS")
    print("=" * 65)
    print(f"Total dialogs: {len(dataset)}")
    print()
    print(f"Intent accuracy:              {intent_acc:.2%}")
    print(f"Satisfaction accuracy:         {sat_acc:.2%}")
    print(f"Quality score MAE:             {quality_mae:.2f}")
    print(f"Quality score exact match:     {quality_exact:.2%}")
    print(f"Quality score within +-1:      {quality_within_1:.2%}")
    print(f"Quality score correlation:     {quality_corr:.3f}")
    print()
    print(f"Hidden dissatisfaction:        {hidden_detected}/{len(hidden_dialogs)} ({hidden_rate:.0%})")
    print()
    print("Agent Mistake Detection:")
    print(f"  {'Mistake':<25} {'Prec':>6} {'Rec':>6} {'F1':>6}  (TP/FP/FN)")
    print(f"  {'-'*25} {'-'*6} {'-'*6} {'-'*6}  {'-'*10}")
    for mistake, m in mistake_metrics.items():
        print(f"  {mistake:<25} {m['precision']:>6.2f} {m['recall']:>6.2f} {m['f1']:>6.2f}  ({m['tp']}/{m['fp']}/{m['fn']})")
    print(f"  {'AVERAGE':<25} {'':>6} {'':>6} {avg_f1:>6.2f}")
    print("=" * 65)

    results = {
        "total_dialogs": len(dataset),
        "intent_accuracy": round(intent_acc, 4),
        "satisfaction_accuracy": round(sat_acc, 4),
        "quality_score": {
            "mae": round(quality_mae, 4),
            "exact_match": round(quality_exact, 4),
            "within_1": round(quality_within_1, 4),
            "correlation": round(quality_corr, 4),
        },
        "hidden_dissatisfaction": {
            "total": len(hidden_dialogs),
            "detected": hidden_detected,
            "rate": round(hidden_rate, 4),
        },
        "agent_mistakes": mistake_metrics,
        "agent_mistakes_avg_f1": round(avg_f1, 4),
    }

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\nResults saved to {OUTPUT_FILE}")
    return results


if __name__ == "__main__":
    evaluate()
