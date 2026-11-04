"""Every script's command line builds (``--help`` of the script and of each subcommand) in the
current environment, without loading a model."""
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
COMMANDS = [
    ["scripts/infer.py"],
    ["scripts/eval.py"],
    ["scripts/eval.py", "generate"],
    ["scripts/eval.py", "score"],
    ["scripts/eval.py", "report"],
    ["scripts/eval.py", "compare"],
    ["scripts/build_csv.py"],
    ["scripts/train_analora.py"],
    ["scripts/train_exlora.py"],
]


@pytest.mark.parametrize("command", COMMANDS, ids=" ".join)
def test_help(command):
    env = dict(os.environ, PYTHONPATH=str(REPO), CUDA_VISIBLE_DEVICES="", WANDB_MODE="disabled")
    result = subprocess.run([sys.executable, *command, "--help"], cwd=REPO, env=env,
                            capture_output=True, text=True, timeout=300)
    assert result.returncode == 0, result.stderr
    assert "usage:" in result.stdout
