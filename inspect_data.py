import pandas as pd
import numpy as np

DATA = r'd:\DS\kaggle\PSTU_Datathon'

train = pd.read_csv(f'{DATA}\\train.csv')
test  = pd.read_csv(f'{DATA}\\test.csv')
samp  = pd.read_csv(f'{DATA}\\sample_submission.csv')

print('=== TRAIN ===')
print('shape:', train.shape)
print('cols[0..5]:', list(train.columns[:6]))
print('cols[-3:]:', list(train.columns[-3:]))
print('TARGET dtype:', train['TARGET'].dtype if 'TARGET' in train.columns else 'MISSING')
print('TARGET dist:', train['TARGET'].value_counts(dropna=False).to_dict() if 'TARGET' in train.columns else 'NA')
print('total NaNs in TARGET:', train['TARGET'].isna().sum() if 'TARGET' in train.columns else 'NA')

print()
print('=== TEST ===')
print('shape:', test.shape)
print('cols[0..5]:', list(test.columns[:6]))
print('cols[-3:]:', list(test.columns[-3:]))
print('id dtype:', test['id'].dtype if 'id' in test.columns else 'MISSING')
print('id min/max:', (test['id'].min(), test['id'].max()) if 'id' in test.columns else 'NA')
print('id unique:', test['id'].nunique() if 'id' in test.columns else 'NA')

print()
print('=== SAMPLE SUBMISSION ===')
print('shape:', samp.shape)
print('cols:', list(samp.columns))
print('id min/max:', (samp['id'].min(), samp['id'].max()))
print('id unique:', samp['id'].nunique())
print('TARGET dist:', samp['TARGET'].value_counts(dropna=False).to_dict() if 'TARGET' in samp.columns else 'NA')

print()
print('=== ID INTERSECTION ===')
ids_te = set(test['id']) if 'id' in test.columns else set()
ids_ss = set(samp['id'])
print('test ids == sample ids:', ids_te == ids_ss)
print('test ids - sample ids:', len(ids_te - ids_ss))
print('sample ids - test ids:', len(ids_ss - ids_te))