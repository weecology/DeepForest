"""Push trained bird detection model to HuggingFace Hub via PR.

This script loads a trained model checkpoint and creates a pull request
on HuggingFace Hub to update the weecology/deepforest-bird model.

Example usage:
    python push_bird_model_to_hf.py --checkpoint path/to/checkpoint.ckpt
"""

import argparse
import os
from pathlib import Path

from dotenv import load_dotenv
from huggingface_hub import login

from deepforest import main


def run():
    """Main function to push model to HuggingFace via PR."""
    parser = argparse.ArgumentParser(
        description="Push trained bird model to HuggingFace Hub via PR"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to the model checkpoint file (.ckpt)",
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        default="weecology/deepforest-bird",
        help="HuggingFace repository ID (default: weecology/deepforest-bird)",
    )
    parser.add_argument(
        "--commit-message",
        type=str,
        default="Update model weights",
        help="Commit message for the PR",
    )

    args = parser.parse_args()

    # Load HF token from .env
    load_dotenv()
    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        raise ValueError(
            "HF_TOKEN not found in .env file. Please add your HuggingFace token to .env"
        )

    # Login to HuggingFace
    login(token=hf_token)

    # Verify checkpoint exists
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    print(f"Loading model from checkpoint: {checkpoint_path}")
    # Load model from checkpoint
    model = main.deepforest.load_from_checkpoint(str(checkpoint_path))

    # Ensure label_dict is set (should be loaded from checkpoint, but verify)
    if not hasattr(model, "label_dict") or model.label_dict is None:
        print("Warning: label_dict not found in checkpoint, setting default Bird label")
        model.label_dict = {"Bird": 0}
        model.numeric_to_label_dict = {0: "Bird"}

    print(f"Model loaded with label_dict: {model.label_dict}")

    # Push to HuggingFace Hub - this will automatically create a PR
    print(f"Pushing model to {args.repo_id} and creating PR...")
    model.model.push_to_hub(
        args.repo_id,
        commit_message=args.commit_message,
        create_pr=True,
    )

    print(f"\nSuccessfully created PR to update {args.repo_id}!")


if __name__ == "__main__":
    run()

