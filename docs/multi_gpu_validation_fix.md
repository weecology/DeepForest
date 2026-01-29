# Multi-GPU validation: gather predictions in on_validation_epoch_end

## Problem

With DDP (e.g. 2 GPUs), each rank only has its share of validation batches in `self.predictions`. In `on_validation_epoch_end` we previously did `pd.concat(self.predictions)` without gathering from other ranks, so metrics (box_recall, box_precision, etc.) were computed on only one rank's subset of the validation set. Result: **2-GPU validation metrics are lower than 1-GPU** because the evaluation sees only half the data.

## Fix

In `src/deepforest/main.py`, in `on_validation_epoch_end`, before building `predictions` and calling `__evaluate__`, gather predictions from all ranks when `world_size > 1`:

```python
        if (self.current_epoch + 1) % self.config.validation.val_accuracy_interval == 0:
            # In DDP, each rank only has its share of validation batches in
            # self.predictions. Gather from all ranks so metrics use the full set.
            if self.trainer.world_size > 1 and torch.distributed.is_initialized():
                object_list = [None] * self.trainer.world_size
                torch.distributed.all_gather_object(object_list, self.predictions)
                all_predictions = [
                    df
                    for rank_list in object_list
                    for df in (rank_list if rank_list is not None else [])
                ]
                predictions = (
                    pd.concat(all_predictions, ignore_index=True)
                    if all_predictions
                    else pd.DataFrame()
                )
            elif len(self.predictions) > 0:
                predictions = pd.concat(self.predictions, ignore_index=True)
            else:
                predictions = pd.DataFrame()

            results = self.__evaluate__(
```

## Verification

- **Scripts:** `src/deepforest/scripts/test_multi_gpu_eval_1gpu.sh` and `test_multi_gpu_eval_2gpu.sh` run the same validation with 1 GPU and 2 GPUs.
- **Without fix:** 2-GPU box_recall (and related metrics) are lower than 1-GPU.
- **With fix:** 1-GPU and 2-GPU metrics match (full validation set used in both cases).

Run from repo root:
```bash
sbatch src/deepforest/scripts/test_multi_gpu_eval_1gpu.sh
sbatch src/deepforest/scripts/test_multi_gpu_eval_2gpu.sh
```
Compare `box_recall` and `box_precision` in the two `.out` log files.
