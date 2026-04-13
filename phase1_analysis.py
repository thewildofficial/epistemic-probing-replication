#!/usr/bin/env python3
"""
Phase 1 Analysis: Calibration curves and H1 testing.

Reads JSONL output from phase1_calibration.py and produces:
1. Confidence-accuracy calibration curves
2. H1 test: partial-knowledge danger zone
3. Summary statistics per model/dataset/type
"""

import json, math, sys
from pathlib import Path
from collections import defaultdict

RESULTS_DIR = Path(__file__).parent / "results" / "phase1"


def load_results(model_name=None):
    """Load all calibration results, optionally filtered by model."""
    records = []
    for f in RESULTS_DIR.glob("calibration_*.jsonl"):
        with open(f) as fh:
            for line in fh:
                try:
                    rec = json.loads(line)
                    if model_name and rec.get("model") != model_name:
                        continue
                    records.append(rec)
                except json.JSONDecodeError:
                    pass
    return records


def confidence_accuracy_curve(records, n_bins=10):
    """Bin records by confidence score, compute accuracy per bin."""
    bins = defaultdict(list)
    
    for rec in records:
        g = rec.get("g_score")
        correct = rec.get("correct")
        if g is None or correct is None:
            continue
        
        bin_idx = min(int(g * n_bins), n_bins - 1)
        bins[bin_idx].append((g, correct))
    
    curve = []
    for bin_idx in sorted(bins.keys()):
        items = bins[bin_idx]
        avg_g = sum(g for g, _ in items) / len(items)
        accuracy = sum(1 for _, c in items if c) / len(items)
        curve.append({
            "bin": bin_idx,
            "range": f"{bin_idx/n_bins:.1f}-{(bin_idx+1)/n_bins:.1f}",
            "avg_confidence": round(avg_g, 4),
            "accuracy": round(accuracy, 4),
            "n": len(items),
        })
    
    return curve


def test_h1_danger_zone(records):
    """
    Test H1: Error rate is highest in the moderate confidence zone (0.3-0.7).
    
    We compare:
    - Low confidence bucket: g < 0.3
    - Danger zone: 0.3 <= g <= 0.7
    - High confidence: g > 0.7
    
    H1 predicts: error rate is highest in the danger zone,
    and the confidence-accuracy relationship is non-monotonic.
    """
    buckets = {"low": [], "danger": [], "high": []}
    
    for rec in records:
        g = rec.get("g_score")
        correct = rec.get("correct")
        if g is None or correct is None:
            continue
        
        if g < 0.3:
            buckets["low"].append(correct)
        elif g <= 0.7:
            buckets["danger"].append(correct)
        else:
            buckets["high"].append(correct)
    
    results = {}
    for name, items in buckets.items():
        if items:
            error_rate = sum(1 for c in items if not c) / len(items)
            results[name] = {
                "n": len(items),
                "accuracy": round(1 - error_rate, 4),
                "error_rate": round(error_rate, 4),
            }
        else:
            results[name] = {"n": 0, "accuracy": 0, "error_rate": 0}
    
    # Check if danger zone has highest error rate
    error_rates = {k: v["error_rate"] for k, v in results.items() if v["n"] > 0}
    h1_supported = (
        error_rates.get("danger", 0) > error_rates.get("low", 0) and
        error_rates.get("danger", 0) > error_rates.get("high", 0)
    )
    
    return {
        "buckets": results,
        "h1_supported": h1_supported,
        "interpretation": (
            "H1 SUPPORTED: Danger zone (0.3-0.7) has highest error rate"
            if h1_supported
            else "H1 NOT SUPPORTED: Danger zone does not have highest error rate"
        ),
    }


def analyze_by_uncertainty_type(records):
    """Break down confidence-accuracy by uncertainty type."""
    by_type = defaultdict(list)
    for rec in records:
        utype = rec.get("uncertainty_type", "unknown")
        g = rec.get("g_score")
        correct = rec.get("correct")
        if g is not None and correct is not None:
            by_type[utype].append((g, correct))
    
    results = {}
    for utype, items in by_type.items():
        avg_g = sum(g for g, _ in items) / len(items)
        accuracy = sum(1 for _, c in items if c) / len(items)
        results[utype] = {
            "n": len(items),
            "avg_confidence": round(avg_g, 4),
            "accuracy": round(accuracy, 4),
            "confidence_accuracy_gap": round(avg_g - accuracy, 4),
        }
    
    return results


def calibration_error(records, n_bins=10):
    """
    Compute Expected Calibration Error (ECE).
    ECE = sum over bins of |accuracy - confidence| * (n_bin / n_total)
    """
    bins = defaultdict(list)
    
    for rec in records:
        g = rec.get("g_score")
        correct = rec.get("correct")
        if g is None or correct is None:
            continue
        bin_idx = min(int(g * n_bins), n_bins - 1)
        bins[bin_idx].append((g, correct))
    
    n_total = sum(len(items) for items in bins.values())
    if n_total == 0:
        return None
    
    ece = 0
    for bin_idx, items in bins.items():
        if not items:
            continue
        avg_conf = sum(g for g, _ in items) / len(items)
        accuracy = sum(1 for _, c in items if c) / len(items)
        ece += abs(avg_conf - accuracy) * (len(items) / n_total)
    
    return round(ece, 4)


def print_report(records, model_name="all"):
    """Print a full analysis report."""
    print(f"\n{'='*70}")
    print(f"  CALIBRATION ANALYSIS — Model: {model_name}")
    print(f"{'='*70}")
    
    # Basic counts
    n_total = len(records)
    n_with_score = sum(1 for r in records if r.get("g_score") is not None)
    n_correct = sum(1 for r in records if r.get("correct") is True)
    n_wrong = sum(1 for r in records if r.get("correct") is False)
    n_unscored = n_total - n_correct - n_wrong
    
    print(f"\n  Total records: {n_total}")
    print(f"  With confidence score: {n_with_score}")
    print(f"  Correct: {n_correct}, Wrong: {n_wrong}, Unscored: {n_unscored}")
    if n_correct + n_wrong > 0:
        print(f"  Overall accuracy: {n_correct/(n_correct+n_wrong):.4f}")
    
    # ECE
    ece = calibration_error(records)
    if ece is not None:
        print(f"\n  Expected Calibration Error (ECE): {ece}")
    
    # Calibration curve
    curve = confidence_accuracy_curve(records)
    if curve:
        print(f"\n  {'Confidence Bin':<20} {'Avg Conf':<12} {'Accuracy':<12} {'N':<8}")
        print(f"  {'-'*52}")
        for point in curve:
            print(f"  {point['range']:<20} {point['avg_confidence']:<12.4f} "
                  f"{point['accuracy']:<12.4f} {point['n']:<8}")
    
    # H1 test
    h1 = test_h1_danger_zone(records)
    print(f"\n  H1: Partial-Knowledge Danger Zone Test")
    print(f"  {'Bucket':<20} {'N':<8} {'Accuracy':<12} {'Error Rate':<12}")
    print(f"  {'-'*52}")
    for bucket in ["low", "danger", "high"]:
        b = h1["buckets"][bucket]
        print(f"  {bucket:<20} {b['n']:<8} {b['accuracy']:<12.4f} {b['error_rate']:<12.4f}")
    print(f"\n  → {h1['interpretation']}")
    
    # By uncertainty type
    by_type = analyze_by_uncertainty_type(records)
    if by_type:
        print(f"\n  By Uncertainty Type:")
        print(f"  {'Type':<15} {'N':<8} {'Avg Conf':<12} {'Accuracy':<12} {'Conf-Acc Gap':<15}")
        print(f"  {'-'*62}")
        for utype, stats in sorted(by_type.items()):
            print(f"  {utype:<15} {stats['n']:<8} {stats['avg_confidence']:<12.4f} "
                  f"{stats['accuracy']:<12.4f} {stats['confidence_accuracy_gap']:<15.4f}")


def main():
    records = load_results()
    
    if not records:
        print("No results found. Run phase1_calibration.py first.")
        sys.exit(1)
    
    # Overall report
    print_report(records, "all")
    
    # Per-model reports
    models = set(r.get("model") for r in records)
    for model in sorted(models):
        model_records = [r for r in records if r.get("model") == model]
        print_report(model_records, model)
    
    # Save analysis summary
    summary = {
        "total_records": len(records),
        "models": list(models),
        "h1_overall": test_h1_danger_zone(records),
        "ece_overall": calibration_error(records),
        "by_type": analyze_by_uncertainty_type(records),
    }
    
    out = RESULTS_DIR / "analysis_summary.json"
    with open(out, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\n  Summary saved to {out}")


if __name__ == "__main__":
    main()
