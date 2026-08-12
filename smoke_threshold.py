"""Smoke test: show how OOF macro-F1 varies as threshold sweeps (sensitive to predicted-positive rate)."""
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

DATA = r'd:\DS\kaggle\PSTU_Datathon'

# Load train labels and oof
train = pd.read_csv(f'{DATA}\\train.csv')
y = train['TARGET'].values

# Load saved oof (from previous run)
try:
    npz = np.load(f'{DATA}\\lgbm_artifacts.npz')
    print('NPZ keys:', list(npz.keys()))
    print('oof_blend shape:', npz['oof_blend'].shape if 'oof_blend' in npz else 'MISSING')
except Exception as e:
    print(f'No artifacts npz: {e}')
    npz = None

if npz is not None and 'oof_blend' in npz:
    p = npz['oof_blend']
    print(f'y mean: {y.mean():.4f}, n_pos: {y.sum()}, n: {len(y)}')

    # Sweep thresholds, show both F1 and predicted-positive rate
    print()
    print(' thr    pos_rate   F1_macro    F1_0     F1_1     recall   precision')
    for thr in np.arange(0.005, 0.5, 0.005):
        pred = (p >= thr).astype(int)
        n_pos_pred = pred.sum()
        if n_pos_pred == 0: continue
        pos_rate_pred = n_pos_pred / len(p)
        f1 = f1_score(y, pred, average='macro')
        f1_0 = f1_score(y, pred, average=None)[0]
        f1_1 = f1_score(y, pred, average=None)[1]
        tp = ((pred == 1) & (y == 1)).sum()
        recall = tp / max(y.sum(), 1)
        precision = tp / max(n_pos_pred, 1)
        print(f' {thr:.3f}   {pos_rate_pred:.4f}    {f1:.4f}     {f1_0:.4f}   {f1_1:.4f}    {recall:.3f}    {precision:.3f}')