"""Simulate prior shift on test: subsample positives/negatives to create a test-like distribution
and find the threshold that maximizes macro-F1 in the simulated scenario."""
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

DATA = r'd:\DS\kaggle\PSTU_Datathon'
npz = np.load(f'{DATA}\\lgbm_artifacts.npz')
oof_cal = npz['oof_cal']
y = npz['y']
print(f'oof_cal shape: {oof_cal.shape}, y mean: {y.mean():.4f}')

# Pick a simulated test prior p_test (what if it's 80% positive?)
def simulate(oof, y, p_test, n_test=20000, seed=0):
    rng = np.random.RandomState(seed)
    n_total = len(y)
    pos_idx = np.where(y == 1)[0]
    neg_idx = np.where(y == 0)[0]
    n_pos = int(n_test * p_test)
    n_neg = n_test - n_pos
    sel_pos = rng.choice(pos_idx, size=min(n_pos, len(pos_idx)), replace=False)
    sel_neg = rng.choice(neg_idx, size=min(n_neg, len(neg_idx)), replace=False)
    sel = np.concatenate([sel_pos, sel_neg])
    rng.shuffle(sel)
    return oof[sel], y[sel]

print()
print('=== Simulated test with different priors (using oof_cal probabilities) ===')
print(f'{"p_test":>8s}  {"thr_optimal":>12s}  {"q_at_opt":>9s}  {"F1_at_opt":>10s}  {"F1@0.5":>10s}  {"F1@0.3":>10s}  {"F1@0.1":>10s}  {"F1@0.05":>10s}')
for p_test in [0.04, 0.10, 0.20, 0.30, 0.50, 0.70, 0.80, 0.90]:
    ot, yt = simulate(oof_cal, y, p_test, n_test=20000, seed=42)
    # Find optimal threshold
    best_f1 = -1
    best_thr = None
    for thr in np.arange(0.01, 0.99, 0.01):
        pred = (ot >= thr).astype(int)
        if pred.sum() == 0: continue
        f1 = f1_score(yt, pred, average='macro')
        if f1 > best_f1:
            best_f1 = f1
            best_thr = thr
    q_at_opt = (ot >= best_thr).mean()
    f1_05 = f1_score(yt, (ot >= 0.5).astype(int), average='macro')
    f1_03 = f1_score(yt, (ot >= 0.3).astype(int), average='macro')
    f1_01 = f1_score(yt, (ot >= 0.1).astype(int), average='macro')
    f1_005 = f1_score(yt, (ot >= 0.05).astype(int), average='macro')
    print(f'{p_test:>8.2f}  {best_thr:>12.3f}  {q_at_opt:>9.4f}  {best_f1:>10.4f}  {f1_05:>10.4f}  {f1_03:>10.4f}  {f1_01:>10.4f}  {f1_005:>10.4f}')

# Try a different rule: predict top q as positive for q in {0.05, 0.10, 0.30, 0.50, 0.70, 0.80, 0.90}
print()
print('=== Predict-top-q rule ===')
print(f'{"p_test":>8s}  {"q=0.05":>8s}  {"q=0.10":>8s}  {"q=0.30":>8s}  {"q=0.50":>8s}  {"q=0.70":>8s}  {"q=0.80":>8s}  {"q=0.90":>8s}')
for p_test in [0.04, 0.10, 0.20, 0.30, 0.50, 0.70, 0.80, 0.90]:
    ot, yt = simulate(oof_cal, y, p_test, n_test=20000, seed=42)
    out = []
    for q in [0.05, 0.10, 0.30, 0.50, 0.70, 0.80, 0.90]:
        thr_q = np.quantile(ot, 1 - q)
        pred = (ot >= thr_q).astype(int)
        f1 = f1_score(yt, pred, average='macro')
        out.append(f'{f1:.4f}')
    print(f'{p_test:>8.2f}  ' + '  '.join(f'{v:>8s}' for v in out))