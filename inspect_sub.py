"""Verify submission_lgbm.csv format and ids match expected test ids."""
import pandas as pd

DATA = r'd:\DS\kaggle\PSTU_Datathon'

sample = pd.read_csv(f'{DATA}\\sample_submission.csv')
sub = pd.read_csv(f'{DATA}\\submission_mhcn2.csv')
test = pd.read_csv(f'{DATA}\\test.csv')

print('=== sample_submission.csv ===')
print('shape:', sample.shape, 'cols:', list(sample.columns))
print('first 5 ids:', sample['id'].head(5).tolist())
print('last 5 ids:', sample['id'].tail(5).tolist())

print()
print('=== submission_lgbm.csv ===')
print('shape:', sub.shape, 'cols:', list(sub.columns))
print('first 5 ids:', sub['id'].head(5).tolist())
print('last 5 ids:', sub['id'].tail(5).tolist())
print('TARGET dist:', sub['TARGET'].value_counts(dropna=False).to_dict())

print()
print('=== test.csv (first/last ids) ===')
print('test ids first 5:', test['id'].head(5).tolist())
print('test ids last 5:', test['id'].tail(5).tolist())

print()
print('=== ID alignment ===')
print('same id set: sub ids == sample ids:', set(sub['id']) == set(sample['id']))
print('same id set: sub ids == test ids:', set(sub['id']) == set(test['id']))
print('rows same order: sub ids == test ids (positional):', (sub['id'].values == test['id'].values).all())
print('rows same order: sub ids == sample ids (positional):', (sub['id'].values == sample['id'].values).all())
print('NaN in sub TARGET:', sub['TARGET'].isna().sum())