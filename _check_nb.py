"""Validate every code cell + notebook format."""
import json, ast
import nbformat

p = r'd:\DS\kaggle\PSTU_Datathon\Enhanced_nb.ipynb'
nb = json.load(open(p, encoding='utf-8'))
errs = 0
for i, c in enumerate(nb['cells'], 1):
    if c['cell_type'] == 'code':
        src = ''.join(c['source'])
        try:
            ast.parse(src)
        except SyntaxError as e:
            errs += 1
            print(f'cell {i:>2} SYNTAX: {e}')
print(f'cells={len(nb["cells"])} syntax_errors={errs}')

# Strict nbformat validate
nb2 = nbformat.read(p, as_version=4)
nbformat.validate(nb2)
print('nbformat validate OK')
print('cell types:', sorted({c.cell_type for c in nb2.cells}))
