"""Execution tests for the maintained educational notebooks."""

import json
from pathlib import Path
from typing import Any

import pytest

NOTEBOOKS = sorted((Path(__file__).parents[1] / "notebooks").glob("[0-9][0-9]_*.ipynb"))


@pytest.mark.parametrize("notebook_path", NOTEBOOKS, ids=lambda path: path.stem)
def test_notebook_code_cells_execute(notebook_path: Path) -> None:
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    namespace: dict[str, Any] = {"__name__": "__notebook_test__"}

    for cell_number, cell in enumerate(notebook["cells"], start=1):
        if cell["cell_type"] != "code":
            continue
        source = "".join(cell["source"])
        code = compile(source, f"{notebook_path.name}:cell-{cell_number}", "exec")
        exec(code, namespace)
