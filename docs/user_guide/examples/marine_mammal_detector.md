# Marine Mammal Detector (Reproducible HPC Workflow)

For an internal, concise reproducibility workflow to train a general marine
mammal detector from BOEM/USGS data, see:

`reproducibility/marine_mammal_detector/README.md`

That bundle includes:
- Marine-mammal-only CSV filtering from BOEM outputs
- Reproducible training entrypoint with optional zero-shot and Hugging Face push
- SLURM script template (`srun` + DDP)
- Quick visualization script for blog post figures

This workflow is intentionally minimal and geared toward repeatable HPC runs,
not a new end-user API.
