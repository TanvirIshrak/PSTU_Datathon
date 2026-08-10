"""Smoke test for notebook2.ipynb: execute every code cell but stub out heavy training.

Strategy:
  * Pre-define `train_cat_one_fold`, `train_cat_bagged`, `train_xgb_bagged`,
    `train_cat_pseudo` etc. as fast stubs that return dummy predictions with the
    correct shape/dtype so downstream cells run.
  * Execute each code cell with `exec(src, g)` and capture exceptions.
"""
import json
import sys
import time
import numpy as np
import pandas as pd
from pathlib import Path

NB = json.loads(Path(r'd:\DS\kaggle\PSTU_Datathon\notebook2.ipynb').read_text(encoding='utf-8'))
cells = NB['cells']

# Build a global namespace and inject stubs.
g = {}
g['__name__'] = '__main__'

# Pre-seeded dummy data so the stubs return sane shapes
N_TRAIN = 76020
N_TEST = 60654
N_FEAT = 350

# Import heavy libs (we want this for stub execution context)
import os, gc, json as _json, time, math, warnings, itertools, sys
warnings.filterwarnings('ignore')
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import xgboost as xgb
from catboost import CatBoostClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler, QuantileTransformer, LabelEncoder
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

g.update(dict(os=os, gc=gc, json=_json, time=time, math=math, warnings=warnings,
              itertools=itertools, sys=sys,
              np=np, pd=pd, torch=torch, nn=nn, F=F,
              DataLoader=DataLoader, TensorDataset=TensorDataset,
              xgb=xgb, CatBoostClassifier=CatBoostClassifier,
              StratifiedKFold=StratifiedKFold, f1_score=f1_score, roc_auc_score=roc_auc_score,
              StandardScaler=StandardScaler, QuantileTransformer=QuantileTransformer,
              LabelEncoder=LabelEncoder,
              HistGradientBoostingClassifier=HistGradientBoostingClassifier,
              LogisticRegression=LogisticRegression))

# Pre-load train/test to keep cell-2 (data load) honest
DATA = Path(r'd:\DS\kaggle\PSTU_Datathon')
train = pd.read_csv(DATA / 'train.csv')
test  = pd.read_csv(DATA / 'test.csv')
sample_sub = pd.read_csv(DATA / 'sample_submission.csv')
g.update(dict(train=train, test=test, sample_sub=sample_sub))

# Stubs - these replace the heavy trainers.
class _Stub:
    pass

def _make_stub_trainer(name):
    def stub(*args, **kwargs):
        # Detect which arrays are passed by positional or kw
        # We need to find the largest numpy array (X) to derive its length.
        largest = None
        for a in args:
            if isinstance(a, np.ndarray):
                if largest is None or a.size > largest.size:
                    largest = a
        # Also check Xtr / Xva / X_df-like in args/kwargs
        cand = None
        for k, v in kwargs.items():
            if isinstance(v, np.ndarray):
                if cand is None or v.size > cand.size: cand = v
        # Last resort: check any list-of-arrays param
        for a in args:
            if isinstance(a, list):
                arrs = [x for x in a if isinstance(x, np.ndarray)]
                if arrs:
                    cand = max(arrs, key=lambda x: x.size)
                    break
        # Try to also read from positional Xtr_df or X_df (DataFrame)
        for a in args:
            if isinstance(a, pd.DataFrame):
                cand = a.values; break
        # If still nothing, fall back to global shapes
        if cand is None:
            cand = train.iloc[:, :-1].values
        # Xva size = cand.shape[0]
        n = cand.shape[0]
        # Xte length — passed as last positional arg usually
        xte = None
        for a in args[::-1]:
            if isinstance(a, np.ndarray) and a.shape[0] not in (n,):
                xte = a; break
            if isinstance(a, pd.DataFrame):
                xte = a.values; break
        # Also search kw
        if xte is None:
            for v in kwargs.values():
                if isinstance(v, pd.DataFrame):
                    xte = v.values; break
                if isinstance(v, np.ndarray) and v.shape[0] != n:
                    xte = v; break
        if xte is None:
            xte_len = N_TEST
        else:
            xte_len = xte.shape[0]
        rng = np.random.default_rng(42)
        pv = rng.uniform(0.0, 1.0, size=n).astype(np.float32)
        pt = rng.uniform(0.0, 1.0, size=xte_len).astype(np.float32)
        return pv, pt, 0.5  # arbitrary AUC
    stub.__name__ = name
    return stub

# Define stubs in g BEFORE running the cells
g['train_cat_one_fold'] = _make_stub_trainer('train_cat_one_fold')
g['train_cat_bagged']   = _make_stub_trainer('train_cat_bagged')
g['train_xgb_bagged']   = _make_stub_trainer('train_xgb_bagged')
g['train_mlp_fold']     = _make_stub_trainer('train_mlp_fold')
g['train_cat_pseudo']   = _make_stub_trainer('train_cat_pseudo')

# Also inject heavy constants/structures that sections 7+ need
g['FOLDS'] = [(np.arange(0, N_TRAIN//2), np.arange(N_TRAIN//2, N_TRAIN))] * 5  # placeholder; will be overwritten
g['SEED'] = 42
g['N_SPLITS'] = 5
g['SEEDS_BAG'] = [42, 1024, 2025, 7, 99]
g['EPOCHS_ANN'] = 30
g['BATCH_ANN'] = 1024
g['DEVICE'] = 'cuda' if torch.cuda.is_available() else 'cpu'

# Run each cell, skipping imports / utility cells that re-import from g (idempotent)
SKIP_PATTERNS = ['import os', 'from sklearn', 'from torch', 'from torch.optim',
                 'from torch.utils', 'import xgboost', 'from catboost',
                 'warnings.filterwarnings', 'pd.set_option',
                 'def set_seed(', 'set_seed(SEED)']

errs = 0
for i, c in enumerate(cells):
    if c['cell_type'] != 'code':
        continue
    src = ''.join(c['source']) if isinstance(c['source'], list) else c['source']
    # Detect duplicate-import cells and skip
    first_line = src.lstrip().splitlines()[0] if src.strip() else ''
    if first_line.startswith(('import ', 'from ', 'warnings.filterwarnings', 'pd.set_option')):
        # These should still execute but produce no observable effect
        pass
    try:
        exec(src, g)
        print(f'  cell {i+1:2d}: OK ({len(src.splitlines())} lines)')
    except Exception as e:
        errs += 1
        print(f'  cell {i+1:2d}: ERROR -> {type(e).__name__}: {e}')
        print('    First lines:', '\n'.join(src.splitlines()[:3]))
        # Continue running to find more errors

print(f'\nDone. {errs} errors. Cells executed:', sum(1 for c in cells if c['cell_type']=='code'))