"""Compare OOF probability distribution with TEST probability distribution. If they differ a lot, there's a distribution shift."""
import numpy as np

DATA = r'd:\DS\kaggle\PSTU_Datathon'
npz = np.load(f'{DATA}\\lgbm_artifacts.npz')

oof_cal = npz['oof_cal']
test_cal = npz['test_cal']

print(f'oof_cal: n={len(oof_cal)}, mean={oof_cal.mean():.4f}, median={np.median(oof_cal):.4f}, p99={np.quantile(oof_cal, 0.99):.4f}')
print(f'test_cal: n={len(test_cal)}, mean={test_cal.mean():.4f}, median={np.median(test_cal):.4f}, p99={np.quantile(test_cal, 0.99):.4f}')
print()
print('Distribution of test_cal vs oof_cal:')
print('quantile     oof       test')
for q in [0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99, 0.995, 0.999]:
    print(f'  {q:.3f}    {np.quantile(oof_cal, q):.4f}    {np.quantile(test_cal, q):.4f}')

print()
print('Histogram of test_cal (rounded):')
for lo, hi in [(0, 0.05), (0.05, 0.10), (0.10, 0.20), (0.20, 0.30), (0.30, 0.50), (0.50, 0.70), (0.70, 0.90), (0.90, 1.001)]:
    n_oof = ((oof_cal >= lo) & (oof_cal < hi)).sum()
    n_te = ((test_cal >= lo) & (test_cal < hi)).sum()
    print(f'  [{lo:.2f}, {hi:.3f}): oof={n_oof:6d} ({n_oof/len(oof_cal)*100:.2f}%)   test={n_te:6d} ({n_te/len(test_cal)*100:.2f}%)')

# Predicted positive rates at various thresholds
print()
print('Predicted positive count and rate at various thresholds:')
print(' thr       oof_q    test_q')
for thr in [0.005, 0.01, 0.02, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]:
    oof_q = (oof_cal >= thr).mean()
    test_q = (test_cal >= thr).mean()
    print(f'  {thr:.3f}    {oof_q:.4f}    {test_q:.4f}')

# Top-k predict
print()
print('If we predict top-q fraction as positive:')
print('  q         test_thr')
for q in [0.05, 0.10, 0.20, 0.30, 0.50, 0.70, 0.80, 0.90]:
    thr_q = np.quantile(test_cal, 1 - q)
    print(f'  {q:.3f}     {thr_q:.4f}')