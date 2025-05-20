import os
import time
import torch
import numpy as np
from deepforest import main
import psutil
import gc
import glob
from tabulate import tabulate
import matplotlib.pyplot as plt

def get_memory_usage():
    """Get current memory usage in MB"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024  # Convert to MB

def profile_predict_tile(model, paths, device, workers=0, patch_size=1500, patch_overlap=0.05, num_runs=2, dataloader_strategy="single"):
    """Profile predict_tile function for a given device and worker configuration"""
    print(f"\nProfiling predict_tile on {device} with {workers} workers using {dataloader_strategy} strategy...")
    
    # Update worker configuration
    model.config["workers"] = workers
    
    # Time profiling
    times = []
    for i in range(num_runs):
        start_time = time.time()
        if dataloader_strategy == "batch":
            model.config["batch_size"] = 1
            model.predict_tile(
                paths=paths, 
                patch_size=patch_size, 
                patch_overlap=patch_overlap,
                dataloader_strategy=dataloader_strategy
            )
        else:
            for path in paths:
                model.predict_tile(
                    path=path, 
                    patch_size=patch_size, 
                    patch_overlap=patch_overlap,
                    dataloader_strategy=dataloader_strategy
                )    
        end_time = time.time()
        times.append(end_time - start_time)
        print(f"Run {i+1}/{num_runs}: {times[-1]:.2f} seconds")
    
    # Clean up
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()
    
    return {
        "device": device,
        "workers": workers,
        "strategy": dataloader_strategy,
        "mean_time": np.mean(times),
        "std_time": np.std(times),
    }

def run():
    # Initialize model
    m = main.deepforest()
    m.create_model()
    m.load_model("Weecology/deepforest-bird")
    m.config["train"]["fast_dev_run"] = False
    m.config["batch_size"] = 10
    strategies = ["single", "batch"]
    
    # Image counts to test
    image_counts = [5, 10, 20]
    worker_configs = [0, 5]
    devices = ["cuda"]
    

    all_results = []

    for n_images in image_counts:
        paths = glob.glob("/blue/ewhite/b.weinstein/BOEM/JPG_20241220_145900/*.jpg")[:n_images]
        for strategy in strategies:
            for device in devices:
                if strategy == "single":
                    m.create_trainer()
                    result = profile_predict_tile(m, paths, device, workers=0, dataloader_strategy=strategy)
                    result["n_images"] = n_images
                    all_results.append(result)
                else:
                    for workers in worker_configs:
                        m.create_trainer()
                        result = profile_predict_tile(m, paths, device, workers, dataloader_strategy=strategy)
                        result["n_images"] = n_images
                        all_results.append(result)

    # Plotting
    plt.figure(figsize=(10,6))
    for strategy in strategies:
        for workers in [0, 5, 10]:
            subset = [r for r in all_results if r["strategy"] == strategy and r["workers"] == workers]
            if not subset:
                continue
            x = [r["n_images"] for r in subset]
            y = [r["mean_time"] for r in subset]
            plt.plot(x, y, marker='o', label=f"{strategy}, workers={workers}")

    plt.xlabel("Number of Images")
    plt.ylabel("Mean Runtime (s)")
    plt.title("Runtime vs Number of Images: {} GPUs".format(torch.cuda.device_count()))
    plt.legend()
    plt.tight_layout()
    plt.show()
    plt.savefig("{}_profiling.png".format(torch.cuda.device_count()), dpi=300)

    # Create comparison table
    table_data = []
    headers = ["Device", "Workers", "Strategy", "Num Images", "Mean Time (s)", "Std Time (s)"]
    
    for result in all_results:
        table_data.append([
            result["device"],
            result["workers"],
            result["strategy"],
            result["n_images"],  # Add number of images here
            f"{result['mean_time']:.2f}",
            f"{result['std_time']:.2f}",
        ])
    
    # Print results
    print("\nProfiling Results Comparison:")
    print("=" * 140)
    print(tabulate(table_data, headers=headers, tablefmt="grid"))

if __name__ == "__main__":
    run()