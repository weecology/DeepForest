import argparse
import os
import sys
import tempfile
from typing import List, Tuple

import numpy as np
import torch

from deepforest import get_data, model


def prepare_five_class_crops(output_dir: str) -> None:
	"""Write deterministic 5-class crops into output_dir using bundled sample data."""
	df_path = get_data("testfile_multi.csv")
	image_example = get_data("SOAP_061.png")

	import pandas as pd

	df = pd.read_csv(df_path)
	boxes = df[["xmin", "ymin", "xmax", "ymax"]].values.tolist()
	root_dir = os.path.dirname(image_example)
	images = df.image_path.values
	class_names = ["A", "B", "C", "D", "E"]
	labels = [class_names[i % len(class_names)] for i in range(len(df))]

	cm = model.CropModel()
	cm.write_crops(
		boxes=boxes,
		labels=labels,
		root_dir=root_dir,
		images=images,
		savedir=output_dir,
	)


def get_val_predictions(cm: model.CropModel) -> Tuple[List[int], List[int]]:
	"""Return true and predicted labels for the current validation set."""
	true_labels, pred_labels = cm.val_dataset_confusion(return_images=False)
	return true_labels, pred_labels


def verify_identical_predictions(
	true_before: List[int], pred_before: List[int], true_after: List[int], pred_after: List[int]
) -> None:
	"""Assert identical predictions (and length) before and after reload."""
	if len(pred_before) != len(pred_after):
		raise AssertionError(
			f"Prediction length differs after reload: {len(pred_before)} vs {len(pred_after)}"
		)
	if not np.array_equal(np.array(pred_before), np.array(pred_after)):
		raise AssertionError("Predictions differ after reload.")
	if len(true_before) != len(true_after):
		raise AssertionError(
			f"Truth length differs after reload: {len(true_before)} vs {len(true_after)}"
		)
	if not np.array_equal(np.array(true_before), np.array(true_after)):
		# Ground truth should be identical when we reattach the same dataset
		raise AssertionError("Ground-truth labels differ after reload.")


def main() -> int:
	parser = argparse.ArgumentParser(
		description="Train a 5-class CropModel, save checkpoint, reload and verify identical predictions."
	)
	parser.add_argument(
		"--output-dir",
		type=str,
		default=None,
		help="Directory to write crops and checkpoint. Defaults to a temporary directory.",
	)
	parser.add_argument("--epochs", type=int, default=1, help="Number of training epochs.")
	parser.add_argument(
		"--limit-train-batches",
		type=int,
		default=1,
		help="Limit number of training batches for quick runs.",
	)
	parser.add_argument(
		"--limit-val-batches",
		type=int,
		default=1,
		help="Limit number of validation batches for quick runs.",
	)
	parser.add_argument(
		"--seed",
		type=int,
		default=123,
		help="Random seed to improve determinism.",
	)
	args = parser.parse_args()

	# Seed for better determinism of training
	torch.manual_seed(args.seed)
	np.random.seed(args.seed)

	# Workspace
	workspace = args.output_dir or tempfile.mkdtemp(prefix="cropmodel_5class_")
	os.makedirs(workspace, exist_ok=True)
	checkpoint_path = os.path.join(workspace, "cropmodel-5class.ckpt")

	print(f"[info] Writing 5-class crops into: {workspace}")
	prepare_five_class_crops(workspace)

	# Train a small 5-class model
	cm = model.CropModel()
	cm.create_trainer(
		fast_dev_run=False,
		limit_train_batches=args.limit_train_batches,
		limit_val_batches=args.limit_val_batches,
		max_epochs=args.epochs,
	)
	cm.load_from_disk(train_dir=workspace, val_dir=workspace)
	cm.create_model(num_classes=len(cm.label_dict))
	print(f"[info] Training model with classes: {cm.label_dict}")
	cm.trainer.fit(cm)

	# Gather predictions before saving
	true_before, pred_before = get_val_predictions(cm)
	print(f"[info] Collected {len(pred_before)} validation predictions before save.")

	# Save checkpoint
	cm.trainer.save_checkpoint(checkpoint_path)
	print(f"[info] Saved checkpoint: {checkpoint_path}")

	# Reload from checkpoint and verify mappings
	cm_loaded = model.CropModel.load_from_checkpoint(checkpoint_path)
	if cm_loaded.label_dict != cm.label_dict:
		raise AssertionError(
			f"label_dict mismatch after reload: {cm_loaded.label_dict} vs {cm.label_dict}"
		)
	if cm_loaded.numeric_to_label_dict != cm.numeric_to_label_dict:
		raise AssertionError(
			f"numeric_to_label_dict mismatch after reload: {cm_loaded.numeric_to_label_dict} vs {cm.numeric_to_label_dict}"
		)
	print("[info] Label mappings preserved across reload.")

	# Reattach datasets and compute predictions again
	cm_loaded.load_from_disk(train_dir=workspace, val_dir=workspace)
	cm_loaded.create_trainer(fast_dev_run=False)
	true_after, pred_after = get_val_predictions(cm_loaded)
	print(f"[info] Collected {len(pred_after)} validation predictions after reload.")

	# Compare
	verify_identical_predictions(true_before, pred_before, true_after, pred_after)
	print("[success] Validation predictions are identical before and after reload.")

	return 0


if __name__ == "__main__":
	sys.exit(main())

import argparse
import os
import sys
import tempfile
from typing import List, Tuple

import numpy as np
import torch

from deepforest import get_data, model


def prepare_five_class_crops(output_dir: str) -> None:
	"""Write deterministic 5-class crops into output_dir using bundled sample data."""
	df_path = get_data("testfile_multi.csv")
	image_example = get_data("SOAP_061.png")

	import pandas as pd

	df = pd.read_csv(df_path)
	boxes = df[["xmin", "ymin", "xmax", "ymax"]].values.tolist()
	root_dir = os.path.dirname(image_example)
	images = df.image_path.values
	class_names = ["A", "B", "C", "D", "E"]
	labels = [class_names[i % len(class_names)] for i in range(len(df))]

	cm = model.CropModel()
	cm.write_crops(
		boxes=boxes,
		labels=labels,
		root_dir=root_dir,
		images=images,
		savedir=output_dir,
	)


def get_val_predictions(cm: model.CropModel) -> Tuple[List[int], List[int]]:
	"""Return true and predicted labels for the current validation set."""
	true_labels, pred_labels = cm.val_dataset_confusion(return_images=False)
	return true_labels, pred_labels


def verify_identical_predictions(
	true_before: List[int], pred_before: List[int], true_after: List[int], pred_after: List[int]
) -> None:
	"""Assert identical predictions (and length) before and after reload."""
	if len(pred_before) != len(pred_after):
		raise AssertionError(
			f"Prediction length differs after reload: {len(pred_before)} vs {len(pred_after)}"
		)
	if not np.array_equal(np.array(pred_before), np.array(pred_after)):
		raise AssertionError("Predictions differ after reload.")
	if len(true_before) != len(true_after):
		raise AssertionError(
			f"Truth length differs after reload: {len(true_before)} vs {len(true_after)}"
		)
	if not np.array_equal(np.array(true_before), np.array(true_after)):
		# Ground truth should be identical when we reattach the same dataset
		raise AssertionError("Ground-truth labels differ after reload.")


def main() -> int:
	parser = argparse.ArgumentParser(
		description="Train a 5-class CropModel, save checkpoint, reload and verify identical predictions."
	)
	parser.add_argument(
		"--output-dir",
		type=str,
		default=None,
		help="Directory to write crops and checkpoint. Defaults to a temporary directory.",
	)
	parser.add_argument("--epochs", type=int, default=1, help="Number of training epochs.")
	parser.add_argument(
		"--limit-train-batches",
		type=int,
		default=1,
		help="Limit number of training batches for quick runs.",
	)
	parser.add_argument(
		"--limit-val-batches",
		type=int,
		default=1,
		help="Limit number of validation batches for quick runs.",
	)
	parser.add_argument(
		"--seed",
		type=int,
		default=123,
		help="Random seed to improve determinism.",
	)
	args = parser.parse_args()

	# Seed for better determinism of training
	torch.manual_seed(args.seed)
	np.random.seed(args.seed)

	# Workspace
	workspace = args.output_dir or tempfile.mkdtemp(prefix="cropmodel_5class_")
	os.makedirs(workspace, exist_ok=True)
	checkpoint_path = os.path.join(workspace, "cropmodel-5class.ckpt")

	print(f"[info] Writing 5-class crops into: {workspace}")
	prepare_five_class_crops(workspace)

	# Train a small 5-class model
	cm = model.CropModel()
	cm.create_trainer(
		fast_dev_run=False,
		limit_train_batches=args.limit_train_batches,
		limit_val_batches=args.limit_val_batches,
		max_epochs=args.epochs,
	)
	cm.load_from_disk(train_dir=workspace, val_dir=workspace)
	cm.create_model(num_classes=len(cm.label_dict))
	print(f"[info] Training model with classes: {cm.label_dict}")
	cm.trainer.fit(cm)

	# Gather predictions before saving
	true_before, pred_before = get_val_predictions(cm)
	print(f"[info] Collected {len(pred_before)} validation predictions before save.")

	# Save checkpoint
	cm.trainer.save_checkpoint(checkpoint_path)
	print(f"[info] Saved checkpoint: {checkpoint_path}")

	# Reload from checkpoint and verify mappings
	cm_loaded = model.CropModel.load_from_checkpoint(checkpoint_path)
	if cm_loaded.label_dict != cm.label_dict:
		raise AssertionError(
			f"label_dict mismatch after reload: {cm_loaded.label_dict} vs {cm.label_dict}"
		)
	if cm_loaded.numeric_to_label_dict != cm.numeric_to_label_dict:
		raise AssertionError(
			f"numeric_to_label_dict mismatch after reload: {cm_loaded.numeric_to_label_dict} vs {cm.numeric_to_label_dict}"
		)
	print("[info] Label mappings preserved across reload.")

	# Reattach datasets and compute predictions again
	cm_loaded.load_from_disk(train_dir=workspace, val_dir=workspace)
	cm_loaded.create_trainer(fast_dev_run=False)
	true_after, pred_after = get_val_predictions(cm_loaded)
	print(f"[info] Collected {len(pred_after)} validation predictions after reload.")

	# Compare
	verify_identical_predictions(true_before, pred_before, true_after, pred_after)
	print("[success] Validation predictions are identical before and after reload.")

	return 0


if __name__ == "__main__":
	sys.exit(main())

