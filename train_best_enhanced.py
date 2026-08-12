"""
train_best_enhanced.py
======================
Trains the *best* known ensemble on the **enhanced (pruned) feature set**
and writes a Kaggle submission CSV.

Pipeline
--------
1. Load `enhanced_train.parquet` / `enhanced_test.parquet` (100 kept features
   after pruning duplicates / high-corr / target-uncorrelated / constant cols).
2. Train 5-fold CatBoost with native categorical handling + the tuned params.
3. Train 5-fold MLP (PyTorch) on the standard-scaled numeric features.
4. Blend OOF + test predictions with weights from `best_weights.json`:
       cat=0.85, xgb=0.00, hgb=0.00, mlp=0.15
5. Apply threshold from `best_threshold.json` (0.31) -> binary TARGET.
6. Write `submissions/submission_enhanced_best.csv` in `id,TARGET` format.

Usage
-----
    python train_best_enhanced.py
"""

from __future__ import annotations
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import f1_score, roc_auc_score

from catboost import CatBoostClassifier

# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
DATA = Path(r"d:\DS\kaggle\PSTU_Datathon")
OUT_DIR = DATA / "submissions"
OUT_DIR.mkdir(exist_ok=True)

SEED = 42
N_SPLITS = 5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

EPOCHS = 40
BATCH = 2048
LR = 1e-3
WD = 1e-4
HIDDEN = (256, 128, 64)
DROPOUT = 0.30

# CatBoost tuned params (best of CV search, validated)
CB_PARAMS = dict(
    loss_function="Logloss",
    eval_metric="AUC",
    iterations=3000,
    learning_rate=0.04,
    depth=7,
    l2_leaf_reg=4.0,
    random_seed=SEED,
    verbose=False,
    allow_writing_files=False,
    task_type="GPU" if torch.cuda.is_available() else "CPU",
    od_type="Iter",
    od_wait=80,
)


def load_artifacts():
    feats = json.loads((DATA / "enhanced_features.json").read_text(encoding="utf-8"))
    weights = json.loads((DATA / "best_weights.json").read_text(encoding="utf-8"))
    thr = json.loads((DATA / "best_threshold.json").read_text(encoding="utf-8"))

    train = pd.read_parquet(DATA / "enhanced_train.parquet")
    test = pd.read_parquet(DATA / "enhanced_test.parquet")

    cat_cols = [c for c in feats["kept_categorical"] if c in train.columns]
    num_cols = [c for c in feats["kept_numeric"] if c in train.columns]
    return train, test, num_cols, cat_cols, weights, thr


# --------------------------------------------------------------------------- #
# Models
# --------------------------------------------------------------------------- #
class MLP(nn.Module):
    def __init__(self, in_dim, hidden=HIDDEN, dropout=DROPOUT):
        super().__init__()
        layers, prev = [], in_dim
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers += [nn.Linear(prev, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


def train_mlp(X_tr, y_tr, X_vl, y_vl, seed=SEED):
    torch.manual_seed(seed)
    model = MLP(X_tr.shape[1]).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)

    # pos_weight to counter ~3.96% positive rate
    pos_w = torch.tensor((y_tr == 0).sum() / max((y_tr == 1).sum(), 1), dtype=torch.float32, device=DEVICE)
    crit = nn.BCEWithLogitsLoss(pos_weight=pos_w)

    ds = TensorDataset(torch.tensor(X_tr, dtype=torch.float32),
                       torch.tensor(y_tr, dtype=torch.float32))
    dl = DataLoader(ds, batch_size=BATCH, shuffle=True, drop_last=False)

    best_auc, best_state = -1.0, None
    for epoch in range(1, EPOCHS + 1):
        model.train()
        for xb, yb in dl:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            loss = crit(model(xb), yb)
            loss.backward()
            opt.step()
        sched.step()

        model.eval()
        with torch.no_grad():
            p = torch.sigmoid(model(torch.tensor(X_vl, dtype=torch.float32, device=DEVICE))).cpu().numpy()
        auc = roc_auc_score(y_vl, p)
        if auc > best_auc:
            best_auc, best_state = auc, {k: v.detach().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    model.eval()
    return model


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #
def run():
    print(f"[init] device={DEVICE}  n_splits={N_SPLITS}")
    train, test, num_cols, cat_cols, weights, thr_info = load_artifacts()
    print(f"[data] train={train.shape}  test={test.shape}  num={len(num_cols)}  cat={len(cat_cols)}")

    # Order columns
    feat_cols = num_cols + cat_cols
    y = train["TARGET"].astype("int8").values
    ids = test["id"].values
    X = train[feat_cols].copy()
    Xt = test[feat_cols].copy()

    # ---- CatBoost needs string cats ----
    X_cb = X.copy()
    Xt_cb = Xt.copy()
    for c in cat_cols:
        X_cb[c] = X_cb[c].astype(str).fillna("nan")
        Xt_cb[c] = Xt_cb[c].astype(str).fillna("nan")

    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)

    oof_cb = np.zeros(len(y))
    oof_mlp = np.zeros(len(y))
    test_cb = np.zeros(len(ids))
    test_mlp = np.zeros(len(ids))

    t0 = time.time()
    for fold, (tr, vl) in enumerate(skf.split(X, y), 1):
        # ----- CatBoost -----
        cb = CatBoostClassifier(**CB_PARAMS)
        cb.fit(X_cb.iloc[tr], y[tr],
               eval_set=(X_cb.iloc[vl], y[vl]),
               cat_features=cat_cols,
               use_best_model=True)
        oof_cb[vl] = cb.predict_proba(X_cb.iloc[vl])[:, 1]
        test_cb += cb.predict_proba(Xt_cb)[:, 1] / N_SPLITS

        # ----- MLP: impute + scale numeric only -----
        imp = SimpleImputer(strategy="median").fit(X.iloc[tr][num_cols])
        scl = StandardScaler().fit(imp.transform(X.iloc[tr][num_cols]))

        Xtr_n = scl.transform(imp.transform(X.iloc[tr][num_cols]))
        Xvl_n = scl.transform(imp.transform(X.iloc[vl][num_cols]))
        Xte_n = scl.transform(imp.transform(Xt[num_cols]))

        model = train_mlp(Xtr_n, y[tr], Xvl_n, y[vl])
        with torch.no_grad():
            Xv = torch.tensor(Xvl_n, dtype=torch.float32, device=DEVICE)
            Xte = torch.tensor(Xte_n, dtype=torch.float32, device=DEVICE)
            oof_mlp[vl] = torch.sigmoid(model(Xv)).cpu().numpy()
            test_mlp += torch.sigmoid(model(Xte)).cpu().numpy() / N_SPLITS

        print(f"  fold {fold}/{N_SPLITS}  cb_auc={roc_auc_score(y[vl], oof_cb[vl]):.4f}"
              f"  mlp_auc={roc_auc_score(y[vl], oof_mlp[vl]):.4f}  ({time.time()-t0:.0f}s)")

    # ---- Blend -----
    cb_w = weights["weights"][0]    # 0.85
    mlp_w = weights["weights"][3]   # 0.15
    oof_blend = cb_w * oof_cb + mlp_w * oof_mlp
    test_blend = cb_w * test_cb + mlp_w * test_mlp

    auc_blend = roc_auc_score(y, oof_blend)
    # Per-component final threshold using stored value
    THR = float(thr_info["threshold"])
    pred_oof = (oof_blend >= THR).astype(int)
    f1_oof = f1_score(y, pred_oof, average="macro")

    print(f"\n[blend] weights=cat:{cb_w:.2f}  mlp:{mlp_w:.2f}")
    print(f"[oof]   AUC={auc_blend:.4f}  macroF1@{THR:.3f}={f1_oof:.4f}")

    # ---- Submission -----
    pred_test = (test_blend >= THR).astype(int)
    sub = pd.DataFrame({"id": ids, "TARGET": pred_test})

    # Sanity vs sample_submission ordering
    sample = pd.read_csv(DATA / "sample_submission.csv")
    if list(sub["id"]) != list(sample["id"]):
        sub = sub.set_index("id").reindex(sample["id"]).reset_index()

    out_path = OUT_DIR / "submission_enhanced_best.csv"
    sub.to_csv(out_path, index=False)
    print(f"[done] wrote {out_path}")
    print(f"       positives={pred_test.sum()} ({pred_test.mean()*100:.2f}%)")

    # Also persist OOF for stacking/blend reuse
    np.savez(DATA / "oof_blend_enhanced.npz",
             oof_cb=oof_cb, oof_mlp=oof_mlp, oof_blend=oof_blend,
             test_cb=test_cb, test_mlp=test_mlp, test_blend=test_blend,
             threshold=THR)


if __name__ == "__main__":
    run()