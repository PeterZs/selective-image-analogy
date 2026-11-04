"""Checkpoint compatibility of the LoRA layers and the adapter files.

The parameter names of ``AnaLoRALinear`` / ``ExLoRALinear`` (``base``, ``lora_A``, ``lora_B``) are the keys of
every released checkpoint, so they are pinned here together with the loaders that read those files.
"""
import json
from pathlib import Path

import pytest
import torch
import torch.nn as nn
from safetensors.torch import save_file

from sia.adapters import (
    checkpoint_files,
    checkpoint_lora_rank,
    find_latest_checkpoint,
    inject_analora_modules,
    inject_exlora_modules,
    load_exlora_state_dict,
    load_tensors,
    save_analora_state_dict,
    save_exlora_state_dict,
)
from sia.lora import AnaLoRALinear, ExLoRALinear

WEIGHTS_REPO = "selective-image-analogy/SIA-weights"


class TinyBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.to_q = nn.Linear(8, 8)
        self.add_q_proj = nn.Linear(8, 8)


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = nn.ModuleList([TinyBlock(), TinyBlock()])


def test_analora_linear_parameters():
    layer = AnaLoRALinear(nn.Linear(8, 6), r=4, alpha=4)
    shapes = {name: tuple(p.shape) for name, p in layer.named_parameters()}
    assert shapes == {
        "base.weight": (6, 8),
        "base.bias": (6,),
        "lora_A.weight": (4, 8),
        "lora_B.weight": (6, 4),
    }
    assert layer.scaling == 1.0


def test_exlora_linear_parameters_and_mask_is_not_saved():
    layer = ExLoRALinear(nn.Linear(8, 6), r=4, alpha=8)
    shapes = {name: tuple(p.shape) for name, p in layer.named_parameters()}
    assert shapes == {
        "base.weight": (6, 8),
        "base.bias": (6,),
        "lora_A.weight": (4, 8),
        "lora_B.weight": (6, 4),
    }
    layer.set_token_mask(torch.ones(1, 3, dtype=torch.bool))
    assert set(layer.state_dict()) == set(shapes)
    assert layer.scaling == 2.0


def test_exlora_only_changes_masked_tokens():
    torch.manual_seed(0)
    layer = ExLoRALinear(nn.Linear(8, 6), r=4, alpha=4)
    nn.init.normal_(layer.lora_B.weight)
    x = torch.randn(1, 3, 8)
    base = layer.base(x)
    assert torch.equal(layer(x), base)  # no mask set: base output only

    layer.set_token_mask(torch.tensor([[True, False, True]]))
    out = layer(x)
    assert torch.equal(out[0, 1], base[0, 1])
    assert not torch.allclose(out[0, 0], base[0, 0])


def test_stage2_checkpoint_keys_after_injection(tmp_path):
    model = TinyModel()
    inject_analora_modules(model, ["to_q", "add_q_proj"], r=2, alpha=2)
    inject_exlora_modules(model, ["add_q_proj"], r=2, alpha=2, dropout=0.0)

    save_analora_state_dict(model, tmp_path / "base_lora.pt")
    save_exlora_state_dict(model, tmp_path / "selective_lora.pt")

    analora = torch.load(tmp_path / "base_lora.pt", weights_only=True)
    exlora = torch.load(tmp_path / "selective_lora.pt", weights_only=True)
    # The AnaLoRA wrapped inside an ExLoRA is saved without its ".base", so both files share module names.
    assert set(analora) == {
        f"blocks.{i}.{m}.lora_{ab}.weight"
        for i in (0, 1) for m in ("to_q", "add_q_proj") for ab in "AB"
    }
    assert set(exlora) == {f"blocks.{i}.add_q_proj.lora_{ab}.weight" for i in (0, 1) for ab in "AB"}
    assert tuple(exlora["blocks.0.add_q_proj.lora_A.weight"].shape) == (2, 8)


def write_pt_and_safetensors(directory, state):
    """Write ``state`` as the trainer file (.pt) in ``directory/pt`` and the release file in ``directory/st``."""
    (directory / "pt").mkdir()
    (directory / "st").mkdir()
    torch.save(state, directory / "pt" / "selective_lora.pt")
    save_file(state, directory / "st" / "exlora.safetensors")


def synthetic_exlora_state():
    torch.manual_seed(0)
    return {
        f"blocks.{i}.add_q_proj.lora_{ab}.weight": torch.randn(shape)
        for i in (0, 1) for ab, shape in (("A", (2, 8)), ("B", (8, 2)))
    }


def test_checkpoint_files_prefers_release_names(tmp_path):
    assert checkpoint_files(tmp_path) == (tmp_path / "base_lora.pt", tmp_path / "selective_lora.pt")
    (tmp_path / "analora.safetensors").touch()
    (tmp_path / "exlora.safetensors").touch()
    assert checkpoint_files(tmp_path) == (tmp_path / "analora.safetensors", tmp_path / "exlora.safetensors")


def test_pt_and_safetensors_load_identical_tensors(tmp_path):
    state = synthetic_exlora_state()
    write_pt_and_safetensors(tmp_path, state)

    from_pt = load_tensors(checkpoint_files(tmp_path / "pt")[1])
    from_st = load_tensors(checkpoint_files(tmp_path / "st")[1])
    assert set(from_pt) == set(from_st) == set(state)
    for key in state:
        assert torch.equal(from_pt[key], from_st[key])
        assert torch.equal(from_pt[key], state[key])


@pytest.mark.parametrize("subdir", ["pt", "st"])
def test_load_exlora_state_dict_round_trip(tmp_path, subdir):
    state = synthetic_exlora_state()
    write_pt_and_safetensors(tmp_path, state)
    model = TinyModel()
    inject_exlora_modules(model, ["add_q_proj"], r=2, alpha=2, dropout=0.0)

    load_exlora_state_dict(model, checkpoint_files(tmp_path / subdir)[1])
    loaded = model.state_dict()
    for key, value in state.items():
        assert torch.equal(loaded[key], value)


def test_load_exlora_state_dict_rejects_zero_matching_keys(tmp_path):
    model = TinyModel()
    inject_exlora_modules(model, ["add_q_proj"], r=2, alpha=2, dropout=0.0)
    # Keys that exist in the model but are not ExLoRA weights.
    state = {"blocks.0.to_q.weight": torch.zeros(8, 8)}
    save_file(state, tmp_path / "exlora.safetensors")
    with pytest.raises(RuntimeError, match="No ExLoRA keys"):
        load_exlora_state_dict(model, tmp_path / "exlora.safetensors")


def test_load_exlora_state_dict_rejects_unknown_keys(tmp_path):
    model = TinyModel()
    inject_exlora_modules(model, ["add_q_proj"], r=2, alpha=2, dropout=0.0)
    save_file({"not.in.model.lora_A.weight": torch.zeros(2, 8)}, tmp_path / "exlora.safetensors")
    with pytest.raises(ValueError, match="Unexpected keys"):
        load_exlora_state_dict(model, tmp_path / "exlora.safetensors")


def test_checkpoint_lora_rank_checks_config(tmp_path):
    state = synthetic_exlora_state()
    assert checkpoint_lora_rank(tmp_path, state) == 2
    (tmp_path / "config.json").write_text(json.dumps({"lora_rank": 4, "lora_alpha": 4}))
    with pytest.raises(ValueError):
        checkpoint_lora_rank(tmp_path, state)


def test_find_latest_checkpoint_ignores_non_numeric_names(tmp_path):
    assert find_latest_checkpoint(tmp_path / "missing") is None
    assert find_latest_checkpoint(tmp_path) is None
    for name in ("checkpoint-500", "checkpoint-2000", "checkpoint-250", "checkpoint-best", "checkpoints"):
        (tmp_path / name).mkdir()
    assert find_latest_checkpoint(tmp_path) == str(tmp_path / "checkpoint-2000")


# ---------------------------------------------------------------------------
# Released weights (HF cache)
# ---------------------------------------------------------------------------

def cached_release_file(filename):
    """Local path of ``filename`` in the cached SIA-weights snapshot; skips when it is not cached."""
    from huggingface_hub import try_to_load_from_cache
    path = try_to_load_from_cache(WEIGHTS_REPO, filename)
    if not isinstance(path, str):
        pytest.skip(f"{WEIGHTS_REPO}/{filename} not in the Hugging Face cache")
    return Path(path)


@pytest.mark.parametrize("backbone", ["flux2_klein", "qwen_image21"])
def test_target_modules_match_released_config(backbone):
    module = pytest.importorskip(f"sia.backbones.{backbone}", exc_type=ImportError)
    config = json.loads(cached_release_file(f"{backbone}/exlora/config.json").read_text())
    assert list(module.ANALORA_TARGET_MODULES) == config["analora_target_modules"]
    assert list(module.EXLORA_TARGET_MODULES) == config["exlora_target_modules"]
