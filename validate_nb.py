"""Validate notebook: JSON + every Python cell syntax."""
import ast, json, sys
from pathlib import Path

NB_PATH = Path(r'd:\DS\kaggle\PSTU_Datathon\notebook.ipynb')

# JSON parse
nb = json.loads(NB_PATH.read_text(encoding='utf-8'))
print(f"JSON OK | cells={len(nb['cells'])} | nbformat={nb['nbformat']}.{nb['nbformat_minor']}")

# Python syntax
errors = 0
for i, c in enumerate(nb['cells']):
    if c['cell_type'] != 'code': continue
    src = ''.join(c['source'])
    try:
        ast.parse(src, filename=f"cell_{i+1}")
        print(f"  cell {i+1:2d}: OK ({src.count(chr(10))} lines)")
    except SyntaxError as e:
        errors += 1
        print(f"  cell {i+1:2d}: SYNTAX ERROR at line {e.lineno}: {e.msg}")
        for j, line in enumerate(src.splitlines()[max(0, e.lineno-2):e.lineno+1], start=max(1, e.lineno-1)):
            print(f"      {j:4d}: {line}")

print(f"\nTotal syntax errors: {errors}")
sys.exit(1 if errors else 0)