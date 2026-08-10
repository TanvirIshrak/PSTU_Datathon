"""Inspect data types and find the PRD_ column."""
import pandas as pd
df = pd.read_csv(r'd:\DS\kaggle\PSTU_Datathon\train.csv', nrows=3000)
print('dtypes value_counts:')
print(df.dtypes.value_counts())
print()
print('Object columns:')
obj = [c for c in df.columns if df[c].dtype == object]
print('count:', len(obj))
print('first 10:', obj[:10])
print()
print('Sample PRD col values:')
if obj:
    c = obj[0]
    print(c, df[c].dropna().unique()[:8])
print()
print('Sample numeric-vs-object per column:')
for c in df.columns:
    if c == 'TARGET': continue
    if df[c].dtype == object:
        # how many unique values?
        u = df[c].nunique(dropna=True)
        nan_count = df[c].isna().sum()
        if u <= 20:
            print(c, '| unique=', u, '| nan=', nan_count, '| sample=', df[c].dropna().unique()[:5])