"""Resolution of SIA adapter checkpoints from a local directory or the Hugging Face Hub.

Base models and datasets are loaded with the standard ``from_pretrained`` / ``load_dataset``
calls, which use the HF cache. Set ``HF_HUB_OFFLINE=1`` to run from the cache only.
"""
from pathlib import Path

from huggingface_hub import snapshot_download
from huggingface_hub.errors import HfHubHTTPError, LocalEntryNotFoundError


def resolve_checkpoint(spec, revision=None, token=None) -> Path:
    """Return the local directory of an adapter checkpoint.

    Args:
        spec: an existing local directory, or a Hub path ``<org>/<repo>/<subfolder>[@revision]``,
            e.g. ``selective-image-analogy/SIA-weights/qwen_image21/exlora@main``. A local
            directory takes precedence. For a Hub path only ``<subfolder>`` is downloaded into
            the HF cache.
        revision: Hub revision (branch, tag or commit) used when ``spec`` names none.
        token: Hugging Face token for a private repo (default: the logged-in token / ``HF_TOKEN``).

    Returns:
        Path to the checkpoint directory (inside the HF cache for a Hub path).

    Raises:
        FileNotFoundError: ``spec`` is neither a local directory nor a downloadable Hub folder.
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
                                         revision=spec_revision or revision, token=token)
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
