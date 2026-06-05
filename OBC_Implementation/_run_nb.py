import sys, nbformat
from nbclient import NotebookClient
nb = nbformat.read('1-layer-jet-obc.ipynb', as_version=4)
client = NotebookClient(nb, timeout=480, kernel_name='python3', resources={'metadata':{'path':'.'}})
try:
    client.execute()
    print("EXECUTE_OK")
except Exception as e:
    print("EXECUTE_FAILED:", type(e).__name__)
    print(str(e)[:4000])
finally:
    nbformat.write(nb, '_obc_executed.ipynb')
