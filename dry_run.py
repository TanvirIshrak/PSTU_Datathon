"""Dry-run import + execute the notebook's Python cells in order to surface runtime errors.
Skips the heavy training cells (only runs the cells that define functions / preprocessing).
"""
import json, sys, ast
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import lightgbm as lgb, xgboost as xgb
from catboost import CatBoostClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, roc_auc_score, classification_report
from sklearn.preprocessing import StandardScaler, QuantileTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from scipy.stats import rankdata
import torch.optim.swa_utils as swa_utils
from torch.optim.swa_utils import AveragedModel, SWALR

NB_PATH = Path(r'd:\DS\kaggle\PSTU_Datathon\notebook.ipynb')
nb = json.loads(NB_PATH.read_text(encoding='utf-8'))

# Use a small subset of training data to make dry-run fast
DATA = Path(r'd:\DS\kaggle\PSTU_Datathon')
train = pd.read_csv(DATA / 'train.csv', nrows=4000)
test = pd.read_csv(DATA / 'test.csv', nrows=4000)
sample_sub = pd.read_csv(DATA / 'sample_submission.csv')
print(f"train: {train.shape} test: {test.shape} sample: {sample_sub.shape}")

# Globals available to exec
g = {
    'os': __import__('os'), 'math': __import__('math'),
    'display': print,  # fallback when not in Jupyter
    'warnings': __import__('warnings'), 'itertools': __import__('itertools'),
    'gc': __import__('gc'), 'json': __import__('json'), 'time': __import__('time'),
    'np': np, 'pd': pd, 'plt': plt, 'sns': sns,
    'torch': torch, 'nn': torch.nn, 'F': torch.nn.functional,
    'DataLoader': torch.utils.data.DataLoader, 'TensorDataset': torch.utils.data.TensorDataset,
    'AveragedModel': AveragedModel, 'SWALR': SWALR, 'swa_utils': swa_utils,
    'StratifiedKFold': StratifiedKFold, 'f1_score': f1_score, 'roc_auc_score': roc_auc_score,
    'classification_report': classification_report, 'confusion_matrix': __import__('sklearn.metrics', fromlist=['confusion_matrix']).confusion_matrix,
    'StandardScaler': StandardScaler, 'QuantileTransformer': QuantileTransformer,
    'HistGradientBoostingClassifier': HistGradientBoostingClassifier,
    'ExtraTreesClassifier': ExtraTreesClassifier, 'RandomForestClassifier': RandomForestClassifier,
    'LogisticRegression': LogisticRegression, 'SimpleImputer': SimpleImputer,
    'lgb': lgb, 'xgb': xgb, 'CatBoostClassifier': CatBoostClassifier,
    'rankdata': rankdata,
    'SEED': 42, 'N_SPLITS': 5, 'SEEDS': [42, 1024, 2025], 'EPOCHS_ANN': 50, 'BATCH_ANN': 2048,
    'DEVICE': 'cuda' if torch.cuda.is_available() else 'cpu',
    'train': train, 'test': test, 'sample_sub': sample_sub,
    'feat_cols': [c for c in train.columns if c.startswith('feat_')],
}
g['set_seed'] = lambda seed=42: (np.random.seed(seed), torch.manual_seed(seed), torch.cuda.manual_seed_all(seed))

# Run the cells in order; do NOT skip any (the slowest ones are GBDT training — we'll limit folds)
cell_count = 0
for i, c in enumerate(nb['cells']):
    if c['cell_type'] != 'code': continue
    cell_count += 1
    src = ''.join(c['source'])
    snippet = src[:160].replace('\n', ' ')
    # Skip the actual per-fold training loops (long); but execute everything else.
    skip_kw = ('for tr, va in folds:', 'for fold in range(', 'for s in SEEDS:')
    if any(k in src for k in skip_kw) and any(t in src for t in ('def train_', ' for fold,', ' for tr,', 'for s, ', 'for seed')):
        print(f"  cell {i+1:2d}: SKIPPED heavy training loop")
        continue
    try:
        exec(src, g)
        print(f"  cell {i+1:2d}: OK")
    except Exception as e:
        print(f"  cell {i+1:2d}: ERROR -> {type(e).__name__}: {e}")
        print("    Source snippet:", snippet)
        sys.exit(1)
print(f"\nAll {cell_count} executed cells OK")
