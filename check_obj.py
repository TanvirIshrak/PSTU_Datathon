"""Detail each object column."""
import pandas as pd
df = pd.read_csv(r'd:\DS\kaggle\PSTU_Datathon\train.csv', nrows=8000)
out = ['=== per-object-column details ===']
for c in df.columns:
    if df[c].dtype != object: continue
    if c == 'TARGET': continue
    try:
        u = int(df[c].nunique(dropna=True))
        nan_count = int(df[c].isna().sum())
        sample = list(df[c].dropna().astype(str).unique()[:15])
        out.append(f'{c} | unique={u} | nan={nan_count}')
        out.append(f'  sample: {sample}')
    except Exception as e:
        out.append(f'{c} | ERROR: {e}')
with open(r'd:\DS\kaggle\PSTU_Datathon\obj_report.txt', 'w', encoding='utf-8') as f:
    f.write('\n'.join(out))
print('written')