"""Convert a Lightning checkpoint to HuggingFace model format."""

from pathlib import Path

import torch

from deepforest.main import deepforest


def export(ckpt_path: str, output_dir: str) -> None:
    """Export a Lightning checkpoint as a HuggingFace model.

    Args:
        ckpt_path: Path to a Lightning ``.ckpt`` file.
        output_dir: Directory to write the HuggingFace model into.
    """
    m = deepforest.load_from_checkpoint(ckpt_path, map_location="cpu")

    # Reload weights from checkpoint to strip EMA / trainer state
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = {
        k.removeprefix("model."): v
        for k, v in ckpt["state_dict"].items()
        if k.startswith("model.")
    }
    m.model.load_state_dict(state)

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    m.model.save_pretrained(output_dir)
    print(f"Exported HF model to {output_dir}")
