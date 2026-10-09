from pathlib import Path
import nbformat
from nbclient import NotebookClient

root = Path(__file__).resolve().parents[1]
for path in sorted((root / "tutorials").glob("*.ipynb")):
    print("Executing", path.name, flush=True)
    notebook = nbformat.read(path, as_version=4)
    NotebookClient(
        notebook,
        timeout=300,
        kernel_name="fieldlab",
        resources={"metadata": {"path": str(root)}},
    ).execute()
    # Persist outputs so learners can inspect the verified experiments.
    nbformat.write(notebook, path)
