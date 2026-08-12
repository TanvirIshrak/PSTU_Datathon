"""Inspect the other npz artifacts (from notebook.ipynb / notebook2.ipynb / stack.ipynb)."""
import numpy as np

DATA = r'd:\DS\kaggle\PSTU_Datathon'
for name in ['oof_and_test.npz', 'oof_predictions.npz', 'test_predictions.npz']:
    p = f'{DATA}\\{name}'
    npz = np.load(p)
    print(f'=== {name} ===')
    print('keys:', list(npz.keys()))
    for k in npz.keys():
        v = npz[k]
        print(f'  {k}: shape={v.shape}, dtype={v.dtype}, mean={v.mean():.4f}' if v.ndim == 1 else f'  {k}: shape={v.shape}, dtype={v.dtype}')
    print()