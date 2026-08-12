"""Smoke test the new prob-blend + isotonic + threshold logic."""
import numpy as np
from sklearn.metrics import f1_score
from sklearn.isotonic import IsotonicRegression

def find_best_threshold(yt, yp, lo=0.05, hi=0.95, step=0.005):
    best_t, best_f = 0.5, 0.0
    for t in np.arange(lo, hi, step):
        f = f1_score(yt, (yp >= t).astype(int), average="macro")
        if f > best_f: best_f, best_t = float(f), float(t)
    return best_t, best_f

rng = np.random.RandomState(0)
N = 76020; pos = 3008  # dataset prior
y = np.zeros(N, int); y[:pos] = 1
np.random.shuffle(y)

# Three "seed" predictions with calibrated probs (~0.04 positives, AUC ~0.88)
def gen(seed):
    s = np.random.RandomState(seed)
    base = s.beta(2, 25, N)
    base = base + 0.10*y  # AUC ~0.88
    base = np.clip(base + 0.005*s.randn(N), 0, 1)
    return base

oof_a, oof_b, oof_c = gen(42), gen(1337), gen(2024)
test_a, test_b, test_c = gen(101), gen(202), gen(303)

oof_blend  = (oof_a + oof_b + oof_c) / 3.0
test_blend = (test_a + test_b + test_c) / 3.0

iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
iso.fit(oof_blend, y)
oof_cal  = iso.transform(oof_blend)
test_cal = iso.transform(test_blend)

thr_a, f1_a = find_best_threshold(y, oof_a)
thr_b, f1_b = find_best_threshold(y, oof_b)
thr_c, f1_c = find_best_threshold(y, oof_c)
thr_b_, f1_b_ = find_best_threshold(y, oof_blend)
thr_c_, f1_c_ = find_best_threshold(y, oof_cal)

print(f"seed A   F1={f1_a:.4f}  thr={thr_a:.3f}")
print(f"seed B   F1={f1_b:.4f}  thr={thr_b:.3f}")
print(f"seed C   F1={f1_c:.4f}  thr={thr_c:.3f}")
print(f"blend    F1={f1_b_:.4f}  thr={thr_b_:.3f}")
print(f"iso      F1={f1_c_:.4f}  thr={thr_c_:.3f}")

# Pick the better, apply to test
if f1_c_ >= f1_b_:
    FINAL_THR, FINAL_TEST = thr_c_, test_cal
    label = "calibrated"
else:
    FINAL_THR, FINAL_TEST = thr_b_, test_blend
    label = "raw"
pred = (FINAL_TEST >= FINAL_THR).astype(int)
print(f"\nUsing {label} threshold {FINAL_THR:.3f}")
print(f"test positives = {pred.sum()}  ({pred.mean()*100:.2f}%)  [expected ~4%]")