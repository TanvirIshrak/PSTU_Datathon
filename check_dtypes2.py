"""Inspect data types and find the PRD_ column - writes to file."""
import pandas as pd, json
df = pd.read_csv(r'd:\DS\kaggle\PSTU_Datathon\train.csv', nrows=3000)
out = []
out.append('=== dtypes value_counts ===')
out.append(str(df.dtypes.value_counts()))
out.append('')
out.append('=== Object columns ===')
obj = [c for c in df.columns if df[c].dtype == object]
out.append(f'count: {len(obj)}')
out.append(f'first 10: {obj[:10]}')
out.append('')
out.append('=== All non-numeric column samples ===')
for c in df.columns:
    if c == 'TARGET': continue
    if df[c].dtype == object:
        u = df[c].nunique(dropna=True)
        nan_count = int(df[c].isna().sum())
        if u <= 30:
            sample = list(df[c].dropna().unique()[:8])
            out.append(f'{c} | unique={u} | nan={nan_count} | sample={sample}')
with open(r'd:\DS\kaggle\PSTU_Datathon\dtypes_report.txt', 'w', encoding='utf-8') as f:
    f.write('\n'.join(out))
print('written')