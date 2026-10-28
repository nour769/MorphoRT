"""Vérifie que le parcours initial possède bien ses cinq pages."""

import ast
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PAGE_FILES = (
    "charger_irm.py",
    "segmentation.py",
    "mesures.py",
    "croissance.py",
    "radiotherapie.py",
)


def test_entrypoint_and_pipeline_pages_exist() -> None:
    """Chaque étape du parcours doit avoir un module Python syntaxiquement valide."""
    entrypoint = PROJECT_ROOT / "streamlit_app.py"
    assert entrypoint.is_file()
    ast.parse(entrypoint.read_text(encoding="utf-8"))

    for page_file in PAGE_FILES:
        page_path = PROJECT_ROOT / "pages" / page_file
        assert page_path.is_file()
        page_tree = ast.parse(page_path.read_text(encoding="utf-8"))
        assert any(
            isinstance(node, ast.FunctionDef) and node.name == "afficher"
            for node in page_tree.body
        )
