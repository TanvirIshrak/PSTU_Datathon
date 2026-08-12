"""Execute Enhanced_nb.ipynb in-place and time it."""
import time, sys
from pathlib import Path
import nbformat
from nbclient import NotebookClient

p = Path(r'd:\DS\kaggle\PSTU_Datathon\Enhanced_nb.ipynb')
nb = nbformat.read(p, as_version=4)

client = NotebookClient(nb, timeout=600, kernel_name='python3', resources={'metadata': {'path': str(p.parent)}})
t0 = time.time()
client.execute()
print(f"executed in {time.time()-t0:.1f}s")
nbformat.write(nb, p)
print(f"saved executed notebook to {p}")
