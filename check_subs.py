import pandas as pd
import os

DATA = r'd:\DS\kaggle\PSTU_Datathon'

for name in ['submission_mhcn.csv', 'submission_mhcn2.csv']:
    p = f'{DATA}\\{name}'
    if not os.path.exists(p):
        print(f'{name}: MISSING')
        continue
    sub = pd.read_csv(p)
    print(f'=== {name} ===')
    print('shape:', sub.shape)
    print('cols:', list(sub.columns))
    print('TARGET dtype:', sub['TARGET'].dtype)
    print('TARGET dist:', sub['TARGET'].value_counts(dropna=False).to_dict())
    print('NaN count:', sub['TARGET'].isna().sum())
    print('min/max:', sub['TARGET'].min(), sub['TARGET'].max())
    print('id min/max:', sub['id'].min(), sub['id'].max())
    print('id unique:', sub['id'].nunique())
    print()