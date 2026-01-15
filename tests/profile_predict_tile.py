# NOTE: Added profiling for predict_tile performance
import gc
import os
import time
import numpy as np
from tabulate import tabulate
from deepforest import main

try:
    import torch
    import psutil
except ImportError:
    pass  # Optional if psutil or torch not available

def profile_predict_tile(model, image_paths, device="cpu", workers=0, patch_size=1500, patch_overlap=0.05, num_runs=2, strategy="single"):
    """Profile predict_tile function with given parameters"""
    print(f"\nProfiling predict_tile on {device} with {workers} workers using {strategy} strategy...")

    # Configure model
    model.config["workers"] = workers

    times = []
    for i in range(num_runs):
        start_time = time.time()
        if strategy == "batch":
            model.config["batch_size"] = 2
            model.predict_tile(
                paths=image_paths,
                patch_size=patch_size,
                patch_overlap=patch_overlap,
                dataloader_strategy=strategy
            )
        else:
            for path in image_paths:
                model.predict_tile(
                    path=path,
                    patch_size=patch_size,
                    patch_overlap=patch_overlap,
                    dataloader_strategy=strategy
                )
        times.append(time.time() - start_time)
        print(f"Run {i+1}/{num_runs}: {times[-1]:.2f} seconds")

        # Cleanup
        gc.collect()
        if device == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()

    return {
        "device": device,
        "workers": workers,
        "strategy": strategy,
        "mean_time": np.mean(times),
        "std_time": np.std(times)
    }


def run():
    # Initialize model
    model = main.deepforest()
    model.create_model()
    model.use_release()  # load the default pretrained model

    # Use sample images included in tests/sample_images folder
    # You can add a few small JPGs here to test
    sample_dir = os.path.join(os.path.dirname(__file__), "sample_images")
    os.makedirs(sample_dir, exist_ok=True)
    image_paths = [os.path.join(sample_dir, f) for f in os.listdir(sample_dir) if f.endswith(".jpg")]
    if not image_paths:
        print("No sample images found in tests/sample_images/. Add a few JPGs to run the profiler.")
        return

    # Test different configurations
    strategies = ["single", "batch"]
    worker_configs = [0, 2]  # small numbers to keep test quick
    devices = ["cpu"]
    if torch.cuda.is_available():
        devices.append("cuda")

    results = []

    for strategy in strategies:
        for device in devices:
            if strategy == "single":
                results.append(profile_predict_tile(model, image_paths, device=device, workers=0, strategy=strategy))
            else:
                for workers in worker_configs:
                    results.append(profile_predict_tile(model, image_paths, device=device, workers=workers, strategy=strategy))

    # Display table
    headers = ["Device", "Workers", "Strategy", "Mean Time (s)", "Std Time (s)"]
    table = []
    for r in results:
        table.append([r["device"], r["workers"], r["strategy"], f"{r['mean_time']:.2f}", f"{r['std_time']:.2f}"])
    print("\nProfiling Results Comparison:")
    print("="*80)
    from tabulate import tabulate
    print(tabulate(table, headers=headers, tablefmt="grid"))


if __name__ == "__main__":
    run()
