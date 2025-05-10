import os
import time
import torch
import numpy as np
from deepforest import main
import psutil
import gc
import glob
from tabulate import tabulate

def get_memory_usage():
    """Get current memory usage in MB"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024  # Convert to MB

def profile_predict_tile(model, paths, device, workers=0, patch_size=1500, patch_overlap=0.05, num_runs=2, dataloader_strategy="single"):
    """Profile predict_tile function for a given device and worker configuration"""
    print(f"\nProfiling predict_tile on {device} with {workers} workers using {dataloader_strategy} strategy...")
    
    # Update worker configuration
    model.config["workers"] = workers
    
    # Move model to device
    if device == "cuda":
        model.model = model.model.cuda()
    else:
        model.model = model.model.cpu()
    
    # Time profiling
    times = []
    for i in range(num_runs):
        start_time = time.time()
        if dataloader_strategy == "batch":
            #change batch size to 1 for batch strategy
            model.config["batch_size"] = 2
            model.predict_tile(
                path=paths, 
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
    #m.load_model("Weecology/deepforest-bird")
    m.config["train"]["fast_dev_run"] = False
    m.config["batch_size"] = 5
    m.config["predict"]["pin_memory"] = False
    
    strategies = ["single", "batch"]
    
    # Get test data
    paths = glob.glob("/blue/ewhite/b.weinstein/BOEM/JPG_20241220_145900/*.jpg")[:20]
    
    # Test configurations
    worker_configs = [0, 5, 10]
    devices = ["cuda"]
    
    # Run all configurations
    results = []
    for strategy in strategies:
        for device in devices:
            for workers in worker_configs:
                m.create_trainer()  # Recreate trainer for each configuration
                result = profile_predict_tile(m, paths, device, workers, dataloader_strategy=strategy)
                results.append(result)
    
    # Create comparison table
    table_data = []
    headers = ["Device", "Workers", "Strategy", "Mean Time (s)", "Std Time (s)"]
    
    for result in results:
        table_data.append([
            result["device"],
            result["workers"],
            result["strategy"],
            f"{result['mean_time']:.2f}",
            f"{result['std_time']:.2f}",
        ])
    
    # Print results
    print("\nProfiling Results Comparison:")
    print("persistant workers")
    print("=" * 140)
    print(tabulate(table_data, headers=headers, tablefmt="grid"))
    
if __name__ == "__main__":
    run()