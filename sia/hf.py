"""Hugging Face Hub access for SIA weights.

Models and datasets are loaded with the standard ``from_pretrained`` / ``load_dataset`` calls, which use the
HF cache. Set ``HF_HUB_OFFLINE=1`` to run from the cache only.
"""
from pathlib import Path

from huggingface_hub import snapshot_download
from huggingface_hub.errors import HfHubHTTPError, LocalEntryNotFoundError


def resolve_checkpoint(spec, revision=None) -> Path:
    """Directory of an adapter checkpoint given as a local dir or as a Hub path.

    An existing local directory wins. Otherwise ``spec`` is read as ``<org>/<repo>/<subfolder>[@revision]``
    (e.g. ``selective-image-analogy/SIA-weights/qwen_image21/exlora``): only that subfolder is downloaded
    into the HF cache and its local path is returned. ``revision`` applies when ``spec`` names none.
    """
    if Path(spec).is_dir():
        return Path(spec)
    repo_path, _, spec_revision = str(spec).partition("@")
    parts = repo_path.strip("/").split("/")
    reason = "a Hub path needs at least <org>/<repo>/<subfolder>"
    if len(parts) >= 3:
        repo_id, subfolder = "/".join(parts[:2]), "/".join(parts[2:])
        try:
            snapshot = snapshot_download(repo_id, allow_patterns=f"{subfolder}/*",
                                         revision=spec_revision or revision)
        except (HfHubHTTPError, LocalEntryNotFoundError, ValueError) as exc:
            reason = f"{type(exc).__name__}: {exc}"
        else:
            folder = Path(snapshot) / subfolder
            if folder.is_dir():
                return folder
            reason = f"{repo_id} has no folder {subfolder!r}"
    raise FileNotFoundError(
        f"Checkpoint {spec!r} is neither an existing local directory nor a Hub path "
        f"<org>/<repo>/<subfolder>[@revision] ({reason})")
