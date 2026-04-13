"""
Deep Analysis of Phase 1 Calibration Data
==========================================
1. Signal collapse quantification (MI, ROC-AUC, information loss)
2. Uncertainty typology from logprob structure (unsupervised + supervised)
3. Calibration curve generation
4. Confidently-wrong cluster analysis
"""

import json, math, numpy as np
from collections import Counter, defaultdict
from pathlib import Path
from scipy import stats
from sklearn.metrics import roc_auc_score, mutual_info_score
from sklearn.preprocessing import KBinsDiscretizer

RESULTS_DIR = Path("results/phase1")
OUTPUT_DIR = Path("results/analysis")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def load_data(model="mixtral-8x22b"):
    path = RESULTS_DIR / f"calibration_{model}.jsonl"
    lines = [json.loads(l) for l in open(path)]
    return lines

def extract_arrays(data):
    """Extract numpy arrays for analysis"""
    correct = np.array([int(d['correct']) for d in data])
    g_score = np.array([d['g_score'] for d in data])
    min_prob = np.array([d['confidence_detail']['min_prob'] for d in data])
    mean_prob = np.array([d['confidence_detail']['mean_prob'] for d in data])
    first_token = np.array([d['confidence_detail']['first_token_prob'] for d in data])
    mean_entropy = np.array([d['confidence_detail']['mean_entropy'] for d in data])
    max_entropy = np.array([d['confidence_detail']['max_entropy'] for d in data])
    mean_alt = np.array([d['confidence_detail']['mean_alt_mass'] for d in data])
    max_alt = np.array([d['confidence_detail']['max_alt_mass'] for d in data])
    n_tokens = np.array([d['confidence_detail']['n_tokens'] for d in data])
    return {
        'correct': correct, 'g_score': g_score, 'min_prob': min_prob,
        'mean_prob': mean_prob, 'first_token': first_token,
        'mean_entropy': mean_entropy, 'max_entropy': max_entropy,
        'mean_alt': mean_alt, 'max_alt': max_alt, 'n_tokens': n_tokens
    }

# ============================================================
# 1. SIGNAL COLLAPSE QUANTIFICATION
# ============================================================
def signal_collapse_analysis(arrs):
    """Quantify how much discriminative information is destroyed by averaging"""
    correct = arrs['correct']
    
    results = {}
    
    # ROC-AUC for each signal (can it predict correctness?)
    signals = ['g_score', 'min_prob', 'first_token', 'mean_entropy', 'max_entropy', 'mean_alt', 'max_alt']
    for sig in signals:
        vals = arrs[sig]
        # Need to handle edge cases
        if len(np.unique(vals)) < 2:
            results[f'auc_{sig}'] = 0.5
            continue
        # For prob signals, lower = more uncertain = more likely wrong, so flip
        if sig in ['g_score', 'min_prob', 'mean_prob', 'first_token']:
            auc = roc_auc_score(correct, vals)  # higher prob -> more likely correct
        else:  # entropy/alt_mass: higher = more uncertain = more likely wrong
            auc = roc_auc_score(correct, -vals)
        results[f'auc_{sig}'] = auc
    
    # Mutual information (discretize continuous variables)
    kbins = KBinsDiscretizer(n_bins=20, encode='ordinal', strategy='quantile')
    for sig in signals:
        vals = arrs[sig].reshape(-1, 1)
        try:
            discretized = kbins.fit_transform(vals).flatten().astype(int)
            mi = mutual_info_score(correct, discretized)
            results[f'mi_{sig}'] = mi
        except:
            results[f'mi_{sig}'] = 0
    
    # Compression ratio: what % of values fall in top decile?
    for sig in ['g_score', 'min_prob', 'mean_prob']:
        vals = arrs[sig]
        p90 = np.percentile(vals, 90)
        compressed = np.mean(vals >= p90)
        results[f'compression_{sig}'] = compressed
        # Effective dynamic range
        results[f'dynamic_range_{sig}'] = float(np.std(vals))
    
    # Correlation between signals
    results['corr_mean_min'] = float(np.corrcoef(arrs['mean_prob'], arrs['min_prob'])[0,1])
    results['corr_g_correct'] = float(np.corrcoef(arrs['g_score'], correct)[0,1])
    results['corr_min_correct'] = float(np.corrcoef(arrs['min_prob'], correct)[0,1])
    
    return results

# ============================================================
# 2. UNCERTAINTY TYPOLOGY
# ============================================================
def uncertainty_typology(data, arrs):
    """Analyze uncertainty type structure in logprob signals"""
    
    # MMLU subject -> domain mapping
    STEM = {'abstract_algebra','astronomy','college_biology','college_chemistry',
            'college_computer_science','college_mathematics','college_physics',
            'computer_security','conceptual_physics','electrical_engineering',
            'elementary_mathematics','formal_logic','high_school_biology',
            'high_school_chemistry','high_school_computer_science',
            'high_school_mathematics','high_school_physics',
            'high_school_statistics','machine_learning','virology','math'}
    
    HUMANITIES = {'high_school_european_history','high_school_us_history',
                  'high_school_world_history','history','philosophy','prehistory',
                  'world_religions','moral_scenarios','moral_disputes',
                  'jurisprudence','international_law','law','logical_fallacies'}
    
    SOCIAL = {'economics','macroeconomics','microeconomics','sociology',
              'psychology','security_studies','public_relations','marketing',
              'management','human_sexuality','business_ethics',
              'clinical_knowledge','college_medicine','medical_genetics',
              'nutrition','professional_accounting','professional_law',
              'professional_medicine','professional_psychology','global_facts',
              'anatomy','econometrics'}
    
    PROFESSIONAL = {'professional_accounting','professional_law',
                    'professional_medicine','professional_psychology',
                    'clinical_knowledge','college_medicine','medical_genetics',
                    'management','marketing','business_ethics',
                    'public_relations','human_sexuality','nutrition'}
    
    # Classify each question
    for d in data:
        subj = d.get('subject','')
        if d['dataset'] == 'gsm8k':
            d['domain'] = 'math_reasoning'
            d['knowledge_type'] = 'procedural'  # requires step-by-step procedure
        elif subj in STEM:
            d['domain'] = 'stem'
            if subj in {'abstract_algebra','college_mathematics','high_school_mathematics',
                        'elementary_mathematics','formal_logic','math','machine_learning',
                        'college_computer_science','high_school_computer_science',
                        'computer_security','high_school_statistics','econometrics'}:
                d['knowledge_type'] = 'formal_reasoning'  # requires symbolic/formal reasoning
            else:
                d['knowledge_type'] = 'factual_stem'  # recall of scientific facts
        elif subj in HUMANITIES:
            d['domain'] = 'humanities'
            d['knowledge_type'] = 'interpretive'  # requires interpretation/judgment
        elif subj in SOCIAL:
            d['domain'] = 'social_science'
            if subj in PROFESSIONAL:
                d['knowledge_type'] = 'professional'  # domain-specific knowledge + application
            else:
                d['knowledge_type'] = 'applied_social'  # social science with application
        else:
            d['domain'] = 'other'
            d['knowledge_type'] = 'unknown'
    
    # Analyze logprob structure by knowledge type
    type_results = {}
    for kt in set(d.get('knowledge_type','') for d in data):
        subset_idx = [i for i,d in enumerate(data) if d.get('knowledge_type') == kt]
        if len(subset_idx) < 5:
            continue
        sub_correct = arrs['correct'][subset_idx]
        sub_min = arrs['min_prob'][subset_idx]
        sub_g = arrs['g_score'][subset_idx]
        sub_ent = arrs['mean_entropy'][subset_idx]
        sub_alt = arrs['mean_alt'][subset_idx]
        sub_max_ent = arrs['max_entropy'][subset_idx]
        
        # Entropy profile: ratio of max_entropy to mean_entropy
        ent_ratio = sub_max_ent / (sub_ent + 1e-10)
        
        type_results[kt] = {
            'n': len(subset_idx),
            'accuracy': float(np.mean(sub_correct)),
            'g_score_mean': float(np.mean(sub_g)),
            'g_score_std': float(np.std(sub_g)),
            'min_prob_mean': float(np.mean(sub_min)),
            'min_prob_std': float(np.std(sub_min)),
            'entropy_mean': float(np.mean(sub_ent)),
            'entropy_std': float(np.std(sub_ent)),
            'alt_mass_mean': float(np.mean(sub_alt)),
            'max_entropy_mean': float(np.mean(sub_max_ent)),
            'entropy_ratio_mean': float(np.mean(ent_ratio)),
            'confidently_wrong_pct': float(np.mean((sub_g > 0.9) & (sub_correct == 0))),
        }
    
    return data, type_results

# ============================================================
# 3. CALIBRATION CURVES & ECE
# ============================================================
def calibration_analysis(arrs):
    """Compute ECE and calibration curve data"""
    correct = arrs['correct']
    
    results = {}
    for sig_name in ['g_score', 'min_prob']:
        conf = arrs[sig_name]
        # ECE with 10 bins
        n_bins = 10
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        ece = 0.0
        calibration_data = []
        for i in range(n_bins):
            lo, hi = bin_boundaries[i], bin_boundaries[i+1]
            mask = (conf >= lo) & (conf < hi)
            if mask.sum() == 0:
                continue
            bin_acc = correct[mask].mean()
            bin_conf = conf[mask].mean()
            bin_count = mask.sum()
            ece += (bin_count / len(correct)) * abs(bin_acc - bin_conf)
            calibration_data.append({
                'bin_low': float(lo), 'bin_high': float(hi),
                'accuracy': float(bin_acc), 'confidence': float(bin_conf),
                'count': int(bin_count)
            })
        results[f'ece_{sig_name}'] = float(ece)
        results[f'calibration_curve_{sig_name}'] = calibration_data
    
    return results

# ============================================================
# 4. CONFIDENTLY-WRONG CLUSTER ANALYSIS
# ============================================================
def confidently_wrong_analysis(data, arrs):
    """Analyze the confidently-wrong cluster"""
    correct = arrs['correct']
    g_score = arrs['g_score']
    min_prob = arrs['min_prob']
    
    confident_wrong = (g_score > 0.9) & (correct == 0)
    confident_correct = (g_score > 0.9) & (correct == 1)
    uncertain_wrong = (g_score <= 0.9) & (correct == 0)
    uncertain_correct = (g_score <= 0.9) & (correct == 1)
    
    results = {
        'confident_wrong_n': int(confident_wrong.sum()),
        'confident_wrong_pct_of_errors': float(confident_wrong.sum() / max((~correct).sum(), 1)),
        'confident_correct_n': int(confident_correct.sum()),
        'uncertain_wrong_n': int(uncertain_wrong.sum()),
        'uncertain_correct_n': int(uncertain_correct.sum()),
    }
    
    # What domains are the confidently-wrong in?
    cw_data = [d for i, d in enumerate(data) if confident_wrong[i]]
    ucw_data = [d for i, d in enumerate(data) if uncertain_wrong[i]]
    
    cw_kt = Counter(d.get('knowledge_type','unknown') for d in cw_data)
    ucw_kt = Counter(d.get('knowledge_type','unknown') for d in ucw_data)
    
    results['confident_wrong_by_type'] = dict(cw_kt.most_common())
    results['uncertain_wrong_by_type'] = dict(ucw_kt.most_common())
    
    # For confidently wrong: does min_prob catch what g_score misses?
    cw_min_probs = min_prob[confident_wrong]
    if len(cw_min_probs) > 0:
        results['cw_min_prob_mean'] = float(np.mean(cw_min_probs))
        results['cw_min_prob_below_07'] = float(np.mean(cw_min_probs < 0.7))
        results['cw_min_prob_below_05'] = float(np.mean(cw_min_probs < 0.5))
    
    # Subject-level overconfidence: which subjects have highest confidently-wrong rate?
    subject_cw_rate = {}
    for d in data:
        subj = d.get('subject','')
        if subj not in subject_cw_rate:
            subject_cw_rate[subj] = {'total': 0, 'cw': 0, 'wrong': 0}
        subject_cw_rate[subj]['total'] += 1
        i = data.index(d)
        if ~correct[i]:
            subject_cw_rate[subj]['wrong'] += 1
            if g_score[i] > 0.9:
                subject_cw_rate[subj]['cw'] += 1
    
    # Top subjects by confidently-wrong rate (among errors)
    subject_cw_list = []
    for subj, counts in subject_cw_rate.items():
        if counts['wrong'] >= 3:  # need at least 3 errors
            rate = counts['cw'] / counts['wrong']
            subject_cw_list.append((subj, rate, counts['cw'], counts['wrong'], counts['total']))
    subject_cw_list.sort(key=lambda x: -x[1])
    results['most_overconfident_subjects'] = [(s, f'{r:.0%}', cw, wrong, total) 
                                               for s, r, cw, wrong, total in subject_cw_list[:15]]
    
    return results

# ============================================================
# 5. ENTROPY PROFILE ANALYSIS (for typology classification)
# ============================================================
def entropy_profile_analysis(data, arrs):
    """Analyze whether entropy profiles differ by knowledge type"""
    correct = arrs['correct']
    
    # Build feature vectors: [min_prob, mean_entropy, max_entropy, entropy_ratio, alt_mass, n_tokens]
    features = np.column_stack([
        arrs['min_prob'],
        arrs['mean_entropy'],
        arrs['max_entropy'],
        arrs['mean_alt'],
        arrs['max_alt'],
        arrs['first_token'],
        (arrs['max_entropy'] / (arrs['mean_entropy'] + 1e-10)),  # entropy ratio
        arrs['n_tokens']
    ])
    
    # Can we predict knowledge_type from features?
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import cross_val_score
    from sklearn.preprocessing import LabelEncoder
    
    kt_labels = [d.get('knowledge_type','unknown') for d in data]
    le = LabelEncoder()
    y = le.fit_transform(kt_labels)
    
    # Only try if we have enough classes with enough samples
    class_counts = Counter(y)
    valid_classes = [c for c, n in class_counts.items() if n >= 20]
    valid_mask = np.isin(y, valid_classes)
    
    if len(valid_classes) >= 2 and valid_mask.sum() >= 50:
        X_valid = features[valid_mask]
        y_valid = y[valid_mask]
        # Balance classes for fair evaluation
        min_class = min(class_counts[c] for c in valid_classes)
        
        rf = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=5)
        scores = cross_val_score(rf, X_valid, y_valid, cv=5, scoring='accuracy')
        
        # Baseline: predict majority class
        majority_acc = max(class_counts[c] for c in valid_classes) / valid_mask.sum()
        
        # Feature importance
        rf.fit(X_valid, y_valid)
        feature_names = ['min_prob', 'mean_entropy', 'max_entropy', 'mean_alt', 
                         'max_alt', 'first_token', 'entropy_ratio', 'n_tokens']
        importances = dict(zip(feature_names, rf.feature_importances_))
        
        return {
            'typology_classification_acc': float(scores.mean()),
            'typology_classification_std': float(scores.std()),
            'majority_baseline': float(majority_acc),
            'feature_importances': {k: float(v) for k, v in sorted(importances.items(), key=lambda x: -x[1])},
            'n_classes': len(valid_classes),
            'class_names': list(le.inverse_transform(valid_classes))
        }
    
    return {'typology_classification': 'insufficient_data'}

# ============================================================
# MAIN
# ============================================================
def main():
    print("Loading data...")
    data = load_data("mixtral-8x22b")
    arrs = extract_arrays(data)
    
    print(f"Loaded {len(data)} entries")
    
    # 1. Signal collapse
    print("\n=== SIGNAL COLLAPSE ANALYSIS ===")
    collapse = signal_collapse_analysis(arrs)
    for k, v in sorted(collapse.items()):
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
    
    # 2. Uncertainty typology
    print("\n=== UNCERTAINTY TYPOLOGY ===")
    data, typology = uncertainty_typology(data, arrs)
    for kt, stats_dict in sorted(typology.items()):
        print(f"  {kt}: n={stats_dict['n']}, acc={stats_dict['accuracy']:.1%}, "
              f"g={stats_dict['g_score_mean']:.3f}, min_p={stats_dict['min_prob_mean']:.3f}, "
              f"ent={stats_dict['entropy_mean']:.3f}, cw%={stats_dict['confidently_wrong_pct']:.1%}")
    
    # 3. Calibration
    print("\n=== CALIBRATION (ECE) ===")
    calib = calibration_analysis(arrs)
    print(f"  ECE (g_score): {calib['ece_g_score']:.4f}")
    print(f"  ECE (min_prob): {calib['ece_min_prob']:.4f}")
    print("  Calibration curve (g_score):")
    for pt in calib['calibration_curve_g_score']:
        print(f"    [{pt['bin_low']:.1f}-{pt['bin_high']:.1f}]: acc={pt['accuracy']:.3f}, conf={pt['confidence']:.3f}, n={pt['count']}")
    print("  Calibration curve (min_prob):")
    for pt in calib['calibration_curve_min_prob']:
        print(f"    [{pt['bin_low']:.1f}-{pt['bin_high']:.1f}]: acc={pt['accuracy']:.3f}, conf={pt['confidence']:.3f}, n={pt['count']}")
    
    # 4. Confidently-wrong
    print("\n=== CONFIDENTLY-WRONG CLUSTER ===")
    cw = confidently_wrong_analysis(data, arrs)
    for k, v in sorted(cw.items()):
        if k in ['most_overconfident_subjects']:
            print(f"  {k}:")
            for item in v:
                print(f"    {item}")
        elif k in ['confident_wrong_by_type', 'uncertain_wrong_by_type']:
            print(f"  {k}: {v}")
        elif isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")
    
    # 5. Entropy profiles
    print("\n=== ENTROPY PROFILE / TYPOLOGY CLASSIFICATION ===")
    entropy_results = entropy_profile_analysis(data, arrs)
    for k, v in sorted(entropy_results.items()):
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        elif isinstance(v, dict):
            print(f"  {k}:")
            for kk, vv in sorted(v.items(), key=lambda x: -x[1]):
                print(f"    {kk}: {vv:.4f}")
        else:
            print(f"  {k}: {v}")
    
    # Save complete results
    full_results = {
        'signal_collapse': collapse,
        'uncertainty_typology': typology,
        'calibration': {k: v for k, v in calib.items() if not k.startswith('calibration_curve')},
        'calibration_curves': {k: v for k, v in calib.items() if k.startswith('calibration_curve')},
        'confidently_wrong': cw,
        'entropy_profiles': entropy_results
    }
    
    with open(OUTPUT_DIR / "deep_analysis_mixtral.json", 'w') as f:
        json.dump(full_results, f, indent=2, default=str)
    print(f"\nResults saved to {OUTPUT_DIR / 'deep_analysis_mixtral.json'}")
    
    # Also save enriched data with knowledge_type
    with open(RESULTS_DIR / "calibration_mixtral-8x22b_enriched.jsonl", 'w') as f:
        for d in data:
            f.write(json.dumps(d, default=str) + '\n')
    print(f"Enriched data saved to {RESULTS_DIR / 'calibration_mixtral-8x22b_enriched.jsonl'}")

if __name__ == "__main__":
    main()
