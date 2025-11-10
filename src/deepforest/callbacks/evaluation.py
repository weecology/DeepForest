import gzip
import json
import os
import shutil
import tempfile
import warnings
from glob import glob
from pathlib import Path

import torch
from pytorch_lightning import Callback, Trainer
from pytorch_lightning.core import LightningModule


class EvaluationCallback(Callback):
    """Accumulate validation predictions per batch, write one shard per rank,
    optionally merge shards on rank 0, and optionally run evaluation.

    File names:
      - Shards: predictions_epoch_{E}_rank{R}.csv[.gz]
      - Merged: predictions_epoch_{E}.csv[.gz]
      - Meta:   predictions_epoch_{E}_metadata.json
    """

    def __init__(
        self,
        save_dir: str | None = None,
        every_n_epochs: int = 5,
        iou_threshold: float = 0.4,
        run_evaluation: bool = False,
        compress: bool = False,
    ) -> None:
        super().__init__()
        self._user_save_dir = save_dir
        self.compress = compress
        self.every_n_epochs = every_n_epochs
        self.iou_threshold = iou_threshold
        self.run_evaluation = run_evaluation

        self.save_dir: Path | None = None
        self._is_temp = save_dir is None
        self._rank_base: Path | None = None
        self.csv_file = None
        self.csv_path: Path | None = None
        self.predictions_written = 0  # rows written by *this rank* this epoch

    def _active_epoch(self, trainer: Trainer) -> bool:
        e = trainer.current_epoch + 1
        return not (
            trainer.sanity_checking
            or trainer.fast_dev_run
            or self.every_n_epochs == -1
            or (e % self.every_n_epochs != 0)
        )

    def _open_writer(self, path: Path):
        if self.compress:
            return gzip.open(path, "wt", encoding="utf-8")
        return open(path, "w", encoding="utf-8")

    def setup(
        self, trainer: Trainer, pl_module: LightningModule, stage: str | None = None
    ):
        # Rank 0 creates/determines the save directory, then broadcasts to all ranks
        # This ensures all ranks write shards to the same location
        if trainer.is_global_zero:
            if self._is_temp:
                base = Path(tempfile.mkdtemp(prefix="preds_"))
                self._rank_base = base
                self.save_dir = base
            else:
                self.save_dir = Path(self._user_save_dir)  # type: ignore[arg-type]
            self.save_dir.mkdir(parents=True, exist_ok=True)

        # Broadcast the directory from rank 0 to all other ranks
        if trainer.world_size > 1:
            save_dir_str = str(self.save_dir) if trainer.is_global_zero else None
            save_dir_str = trainer.strategy.broadcast(save_dir_str, src=0)
            self.save_dir = Path(save_dir_str)

        # Non-rank-0 processes ensure the directory exists
        if not trainer.is_global_zero:
            self.save_dir.mkdir(parents=True, exist_ok=True)

    def on_validation_epoch_start(
        self, trainer: Trainer, pl_module: LightningModule
    ) -> None:
        if self._active_epoch(trainer):
            epoch = trainer.current_epoch + 1
            rank = trainer.global_rank
            suffix = ".csv.gz" if self.compress else ".csv"
            self.csv_path = (
                self.save_dir / f"predictions_epoch_{epoch}_rank{rank}{suffix}"
            )
            self.csv_file = self._open_writer(self.csv_path)
        else:
            self.csv_path = None
            self.csv_file = None

        self.predictions_written = 0

        trainer.strategy.barrier()

    def on_validation_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs,
        batch,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        if not self._active_epoch(trainer) or self.csv_file is None:
            return
        # expected: pl_module.last_preds is list[pd.DataFrame]
        batch_preds = getattr(pl_module, "last_preds", None)
        if not batch_preds:
            return
        for df in batch_preds:
            if df is None or df.empty:
                continue
            df.to_csv(self.csv_file, index=False, header=(self.predictions_written == 0))
            self.predictions_written += len(df)

    def on_validation_epoch_end(
        self, trainer: Trainer, pl_module: LightningModule
    ) -> None:
        strategy = trainer.strategy
        world_size = strategy.world_size

        if self.csv_file is not None:
            self.csv_file.close()
            self.csv_file = None

        strategy.barrier()  # all ranks finished writing

        # Collect each rank's save_dir and row count
        if (
            world_size > 1
            and torch.distributed.is_available()
            and torch.distributed.is_initialized()
        ):
            rank_dirs: list[str | None] = [None] * world_size
            rank_counts: list[int] = [0] * world_size
            torch.distributed.all_gather_object(rank_dirs, str(self.save_dir))
            torch.distributed.all_gather_object(
                rank_counts, int(self.predictions_written)
            )
        else:
            rank_dirs = [str(self.save_dir)]
            rank_counts = [int(self.predictions_written)]

        if self._active_epoch(trainer) and trainer.is_global_zero:
            self._reduce_and_evaluate(
                trainer, pl_module, [Path(d) for d in rank_dirs if d], sum(rank_counts)
            )

        strategy.barrier()  # allow rank 0 to finish

    def teardown(
        self, trainer: Trainer, pl_module: LightningModule, stage: str | None = None
    ):
        if self._is_temp and self._rank_base is not None:
            shutil.rmtree(self._rank_base, ignore_errors=True)

    def _reduce_and_evaluate(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        rank_dirs: list[Path],
        total_written: int,
    ) -> None:
        epoch = trainer.current_epoch + 1
        suffix = ".csv.gz" if self.compress else ".csv"

        # Deduplicate rank_dirs
        unique_dirs = list(dict.fromkeys(rank_dirs))
        if len(unique_dirs) < len(rank_dirs):
            warnings.warn(
                f"Detected {len(rank_dirs) - len(unique_dirs)} duplicate directories "
                f"in rank_dirs. This may indicate a configuration issue.",
                stacklevel=2,
            )

        # discover shards
        shard_paths: list[Path] = []
        for d in unique_dirs:
            pattern = str(d / f"predictions_epoch_{epoch}_rank*.csv")
            if self.compress:
                pattern += ".gz"
            shard_paths.extend(sorted(Path(p) for p in glob(pattern)))

        # Validate shard count matches world size
        world_size = trainer.strategy.world_size
        if len(shard_paths) != world_size:
            warnings.warn(
                f"Expected {world_size} shard files but found {len(shard_paths)}. "
                f"Shards: {[p.name for p in shard_paths]}",
                stacklevel=2,
            )

        merged_path = (
            (self.save_dir / f"predictions_epoch_{epoch}{suffix}")
            if shard_paths
            else None
        )

        # stream-merge shards into a single file without repeating headers
        if merged_path is not None:
            merged_path.parent.mkdir(parents=True, exist_ok=True)
            open_out = gzip.open if self.compress else open
            with open_out(merged_path, "wt", encoding="utf-8") as out_f:
                wrote_header = False
                for shard in shard_paths:
                    open_in = (
                        gzip.open
                        if shard.suffix == ".gz" or shard.suffixes[-2:] == [".csv", ".gz"]
                        else open
                    )
                    with open_in(shard, "rt", encoding="utf-8") as in_f:
                        for i, line in enumerate(in_f):
                            if i == 0 and wrote_header:
                                continue
                            out_f.write(line)
                    wrote_header = True

        # metadata
        cfg = getattr(pl_module, "config", None)
        val = getattr(cfg, "validation", None)
        meta = {
            "epoch": epoch,
            "current_step": trainer.global_step,
            "predictions_count": int(total_written),
            "target_csv_file": getattr(val, "csv_file", None),
            "target_root_dir": getattr(val, "root_dir", None),
            "shards": [str(p) for p in shard_paths],
            "merged_predictions": str(merged_path) if merged_path else None,
            "world_size": trainer.strategy.world_size,
        }
        with open(
            self.save_dir / f"predictions_epoch_{epoch}_metadata.json",
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(meta, f, indent=2)

        # optional shard cleanup
        for p in shard_paths:
            try:
                os.remove(p)
            except OSError:
                pass

        # optional evaluation
        if self.run_evaluation:
            if merged_path and total_written > 0:
                try:
                    pl_module.evaluate(
                        predictions=str(merged_path),
                        csv_file=meta["target_csv_file"],
                        iou_threshold=self.iou_threshold,
                    )
                except Exception as e:
                    warnings.warn(f"Evaluation failed: {e}", stacklevel=2)
            else:
                warnings.warn(
                    "No predictions written to disk, skipping evaluate.", stacklevel=2
                )
