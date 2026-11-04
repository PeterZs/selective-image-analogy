"""Import rule of the package, checked on the source with ``ast`` (nothing is imported or executed).

shared layer (hf, lora, adapters, prompt, data, concept_suppression, viz, backbones)
    imports nothing role-specific: no ``sia.inference``, ``sia.training``, ``sia.eval``;
backbones
    are independent of each other (the Klein env must never import Qwen code and vice versa);
inference
    imports the shared layer only;
training
    imports the shared layer, inference and backbones (no ``sia.eval``);
eval (``sia/eval/``)
    imports the shared layer and inference (no ``sia.training``);
nothing under ``sia/``
    imports the ``training`` or ``scripts`` folders;
scripts
    import first-party code only from ``sia`` (never another script), and only
    ``scripts/train_*.py`` import ``sia.training``.

Modules that do not exist yet are skipped.
"""
import ast
from pathlib import Path

import pytest

SIA = Path(__file__).resolve().parent.parent / "sia"
SCRIPTS = SIA.parent / "scripts"

SHARED = ["hf.py", "lora.py", "adapters.py", "prompt.py", "data.py", "concept_suppression.py", "viz.py"]
BACKBONES = ["flux2_klein.py", "qwen_image21.py"]
ROLE_SPECIFIC = ["sia.inference", "sia.training", "sia.eval"]
OUTSIDE_PACKAGE = ["training", "scripts"]


def module_name(path):
    """Dotted module name of a file under ``sia/`` (``sia/backbones/qwen_image21.py`` -> ``sia.backbones.qwen_image21``)."""
    parts = path.relative_to(SIA.parent).with_suffix("").parts
    return ".".join(parts[:-1] if parts[-1] == "__init__" else parts)


def imported_modules(path):
    """Absolute names of every module imported anywhere in ``path`` (including function-level imports)."""
    package = module_name(path) if path.name == "__init__.py" else module_name(path).rpartition(".")[0]
    names = set()
    for node in ast.walk(ast.parse(path.read_text(), filename=str(path))):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level:  # relative import
                base = package.split(".")
                base = base[:len(base) - node.level + 1]
                module = ".".join(base + ([node.module] if node.module else []))
            else:
                module = node.module
            names.add(module)
            # ``from sia import inference`` imports the module sia.inference.
            names.update(f"{module}.{alias.name}" for alias in node.names)
    return names


def forbidden_imports(path, prefixes):
    return sorted(name for name in imported_modules(path)
                  if any(name == p or name.startswith(p + ".") for p in prefixes))


def existing(path):
    if not path.exists():
        pytest.skip(f"{path.relative_to(SIA.parent)} does not exist yet")
    return path


@pytest.mark.parametrize("filename", SHARED)
def test_shared_modules_import_nothing_role_specific(filename):
    path = existing(SIA / filename)
    assert forbidden_imports(path, ROLE_SPECIFIC + OUTSIDE_PACKAGE) == []


@pytest.mark.parametrize("filename", BACKBONES)
def test_backbones_import_only_the_shared_layer(filename):
    path = existing(SIA / "backbones" / filename)
    assert forbidden_imports(path, ROLE_SPECIFIC + OUTSIDE_PACKAGE) == []


@pytest.mark.parametrize("filename, other", [
    ("flux2_klein.py", "sia.backbones.qwen_image21"),
    ("qwen_image21.py", "sia.backbones.flux2_klein"),
])
def test_backbones_do_not_import_each_other(filename, other):
    path = existing(SIA / "backbones" / filename)
    assert forbidden_imports(path, [other]) == []


def test_inference_imports_only_the_shared_layer():
    path = existing(SIA / "inference.py")
    assert forbidden_imports(path, ["sia.training", "sia.eval"] + OUTSIDE_PACKAGE) == []


def test_training_imports_no_evaluation_code():
    path = existing(SIA / "training.py")
    assert forbidden_imports(path, ["sia.eval"] + OUTSIDE_PACKAGE) == []


@pytest.mark.parametrize("path", sorted(SIA.rglob("*.py")), ids=lambda p: str(p.relative_to(SIA.parent)))
def test_package_does_not_import_training_or_scripts(path):
    assert forbidden_imports(path, OUTSIDE_PACKAGE) == []


@pytest.mark.parametrize("path", sorted((SIA / "eval").glob("*.py")), ids=lambda p: p.name)
def test_eval_imports_no_training_code(path):
    assert forbidden_imports(path, ["sia.training"] + OUTSIDE_PACKAGE) == []


def script_imports(path):
    """Top-level module names imported by a script (it is not part of a package)."""
    names = set()
    for node in ast.walk(ast.parse(path.read_text(), filename=str(path))):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            assert node.level == 0, f"relative import in {path.name}"
            names.add(node.module)
            names.update(f"{node.module}.{alias.name}" for alias in node.names)
    return names


@pytest.mark.parametrize("path", sorted(SCRIPTS.glob("*.py")), ids=lambda p: p.name)
def test_scripts_import_only_sia(path):
    first_party = OUTSIDE_PACKAGE + [p.stem for p in SCRIPTS.glob("*.py")]
    names = script_imports(path)
    assert sorted(n for n in names if any(n == p or n.startswith(p + ".") for p in first_party)) == []
    if not path.name.startswith("train_"):
        assert sorted(n for n in names if n == "sia.training" or n.startswith("sia.training.")) == []
