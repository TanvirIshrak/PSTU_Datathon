"""Smoke-test the heavy pure-Python cells of notebook3 on the first 10k rows."""
import time, sys
import numpy as np, pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
import lightgbm as lgb

DATA = r"d:\DS\kaggle\PSTU_Datathon"
CAT_COLS = ["feat_142","feat_157","feat_318","feat_320","feat_325","feat_337"]

t0 = time.time()
train = pd.read_csv(f"{DATA}\\train.csv", nrows=10000)
test  = pd.read_csv(f"{DATA}\\test.csv",  nrows=2000)
print("loaded", train.shape, test.shape, f"{time.time()-t0:.1f}s")

y = train["TARGET"].astype("int8").values
ids_test = test["id"].values
train = train.drop(columns=["TARGET"])
feat_cols = [c for c in train.columns if c != "id"]
train = train[feat_cols]; test = test.reindex(columns=feat_cols, fill_value=np.nan)
for c in CAT_COLS:
    train[c] = train[c].astype("category")
    test[c]  = test[c].astype("category")

NUM = [c for c in feat_cols if c not in CAT_COLS]
def add_row_stats(df, cols):
    b = df[cols]
    s = pd.DataFrame(index=df.index)
    s["row_nan_cnt"] = b.isna().sum(axis=1).astype("int32")
    s["row_mean"]  = b.mean(axis=1); s["row_std"] = b.std(axis=1)
    s["row_min"]   = b.min(axis=1);  s["row_max"] = b.max(axis=1)
    s["row_med"]   = b.median(axis=1); s["row_skew"] = b.skew(axis=1)
    s["row_kurt"]  = b.kurt(axis=1)
    s["row_q01"]   = b.quantile(0.01, axis=1); s["row_q99"] = b.quantile(0.99, axis=1)
    s["row_rng"]   = s["row_max"] - s["row_min"]
    s["row_iqr"]   = b.quantile(0.75, axis=1) - b.quantile(0.25, axis=1)
    s["row_nzero"] = (b != 0).sum(axis=1).astype("int32")
    return s.astype("float32")

t0 = time.time()
train = pd.concat([train.reset_index(drop=True), add_row_stats(train, NUM).reset_index(drop=True)], axis=1)
test  = pd.concat([test.reset_index(drop=True),  add_row_stats(test, NUM).reset_index(drop=True)],  axis=1)
print("FE done", train.shape, f"{time.time()-t0:.1f}s")

# TE
def add_te(Xt, yt, Xe, cols, n=3, seed=42, sm=20.0, pref="te"):
    skf = StratifiedKFold(n_splits=n, shuffle=True, random_state=seed)
    gm = float(np.mean(yt))
    oof = pd.DataFrame(index=Xt.index, columns=cols, dtype=float)
    full = pd.DataFrame(index=Xe.index, columns=cols, dtype=float)
    for c in cols:
        for tri, vli in skf.split(Xt, yt):
            grp = pd.DataFrame({"x": Xt.iloc[tri][c].astype("object").fillna("__nan__"),
                                "y": yt[tri]}).groupby("x")["y"].agg(["sum","count"])
            enc = (grp["sum"] + sm*gm) / (grp["count"] + sm)
            oof.iloc[vli, oof.columns.get_loc(c)] = Xt.iloc[vli][c].astype("object").fillna("__nan__").map(enc).fillna(gm).values
        grp_f = pd.DataFrame({"x": Xt[c].astype("object").fillna("__nan__"),
                              "y": yt}).groupby("x")["y"].agg(["sum","count"])
        enc_f = (grp_f["sum"] + sm*gm) / (grp_f["count"] + sm)
        full[c] = Xe[c].astype("object").fillna("__nan__").map(enc_f).fillna(gm).values
    oof.columns = [f"{pref}_{c}" for c in cols]
    full.columns = [f"{pref}_{c}" for c in cols]
    Xt = pd.concat([Xt.reset_index(drop=True), oof.reset_index(drop=True).astype("float32")], axis=1)
    Xe = pd.concat([Xe.reset_index(drop=True), full.reset_index(drop=True).astype("float32")], axis=1)
    return Xt, Xe

t0 = time.time()
X_train, X_test = add_te(train, y, test, CAT_COLS, n=3)
print("TE done", X_train.shape, X_test.shape, f"{time.time()-t0:.1f}s")

# Tiny LGBM fold (3 folds) for timing
params = dict(objective="binary", metric="binary_logloss", learning_rate=0.05,
              num_leaves=64, min_data_in_leaf=40, feature_fraction=0.8, bagging_fraction=0.8,
              bagging_freq=1, lambda_l2=1.0, verbose=-1, n_jobs=-1, is_unbalance=True,
              seed=42)
oof = np.zeros(len(y))
feats = X_train.columns.tolist()
cat_idx = [feats.index(c) for c in CAT_COLS]
for fold,(tr,vl) in enumerate(StratifiedKFold(n_splits=3, shuffle=True, random_state=42).split(X_train,y),1):
    d_tr = lgb.Dataset(X_train.iloc[tr], y[tr], categorical_feature=CAT_COLS, free_raw_data=False)
    d_vl = lgb.Dataset(X_train.iloc[vl], y[vl], categorical_feature=CAT_COLS, reference=d_tr, free_raw_data=False)
    m = lgb.train(params, d_tr, num_boost_round=800, valid_sets=[d_vl],
                  callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(0)])
    oof[vl] = m.predict(X_train.iloc[vl], num_iteration=m.best_iteration)
    f1 = f1_score(y[vl], (oof[vl]>=0.5).astype(int), average="macro")
    print(f"  fold {fold} best_iter={m.best_iteration} f1@.5={f1:.4f}")
print("OOF macroF1 @0.5:", f1_score(y,(oof>=0.5).astype(int), average="macro"))
print("DONE")
