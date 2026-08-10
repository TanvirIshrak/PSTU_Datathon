"""Test what dtype a mixed pandas DataFrame converts to."""
import numpy as np, pandas as pd
df = pd.DataFrame({'a': [1, 2, 3], 'b': [1.5, 2.5, 3.5], 'c': [10, 20, 30]})
df['a'] = df['a'].astype('int32')
df['b'] = df['b'].astype('float32')
df['c'] = df['c'].astype('int32')
print('dtypes:', df.dtypes.tolist())
arr = df.values
print('arr.dtype:', arr.dtype)
# Slice
print('arr[0:2].dtype:', arr[0:2].dtype)
# Test catboost behavior with the DataFrame
from catboost import CatBoostClassifier
m = CatBoostClassifier(iterations=5, verbose=False)
try:
    m.fit(df, [0, 1, 0], cat_features=[0, 2])
    print('DataFrame path OK')
except Exception as e:
    print('DataFrame path FAIL:', e)
try:
    m.fit(arr, [0, 1, 0], cat_features=[0, 2])
    print('object array path OK')
except Exception as e:
    print('object array path FAIL:', e)
try:
    m.fit(arr[0:2], [0, 1], cat_features=[0, 2])
    print('object slice path OK')
except Exception as e:
    print('object slice path FAIL:', e)