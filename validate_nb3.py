import json, sys
p = r'd:\DS\kaggle\PSTU_Datathon\notebook3.ipynb'
with open(p, 'r', encoding='utf-8') as f:
    nb = json.load(f)
print('OK cells=', len(nb['cells']))
for i, c in enumerate(nb['cells']):
    src = ''.join(c.get('source', []))
    head = src.splitlines()[:1]
    print(f"  [{i+1}] {c['cell_type']}: {head[0] if head else ''}")
