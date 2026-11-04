"""The Klein image size defaults to the native 384x512 the released checkpoints were trained at."""
import pytest

pytest.importorskip("diffusers")
from sia.backbones import flux2_klein  # noqa: E402


def test_native_resolution_is_the_default(monkeypatch):
    monkeypatch.delenv("SIA_NATIVE_RES", raising=False)
    assert flux2_klein._native_wh() == (384, 512)


@pytest.mark.parametrize("value", ["0", ""])
def test_square_recipe_can_be_selected(monkeypatch, value):
    monkeypatch.setenv("SIA_NATIVE_RES", value)
    assert flux2_klein._native_wh() is None


def test_explicit_sizes(monkeypatch):
    monkeypatch.setenv("SIA_NATIVE_RES", "1")
    assert flux2_klein._native_wh() == (384, 512)
    monkeypatch.setenv("SIA_NATIVE_RES", "640x480")
    assert flux2_klein._native_wh() == (640, 480)
