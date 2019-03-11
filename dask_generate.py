import glob
import subprocess
import socket
import os
import sys

from DeepForest import Generate, config

def find_csvs():
    """
    Find training csvs in site path
    """
    DeepForest_config = config.load_config()
    data_paths = {}
    for site in DeepForest_config['training_csvs']:
        file_path = DeepForest_config[site]["training_csvs"]
        search_path = os.path.join(file_path,"*.csv")
        found_csvs = glob.glob(search_path)
        data_paths[site] = found_csvs
    
    return data_paths

def run_test(data_paths):
    DeepForest_config = config.load_config()    
    site=DeepForest_config['training_csvs'][0]
    Generate.run(data_paths[site][0], DeepForest_config,site=site)
    
def start_tunnel():
    """
    Start a juypter session and ssh tunnel to view task progress
    """
    host = socket.gethostname()        
    print("To tunnel into dask dashboard:")
    print("ssh -N -L 8787:%s:8787 -l b.weinstein hpg2.rc.ufl.edu" % (host))
    
    #flush system
    sys.stdout.flush()

def run_HPC(data_paths):
        
    #################
    # Setup dask cluster
    #################
    
    from dask_jobqueue import SLURMCluster
    from dask.distributed import Client, wait
    
    DeepForest_config = config.load_config()
    num_workers = DeepForest_config["num_hipergator_workers"]
    
    #job args
    extra_args=[
        "--error=/home/b.weinstein/logs/dask-worker-%j.err",
        "--account=ewhite",
        "--output=/home/b.weinstein/logs/dask-worker-%j.out"
    ]
    
    cluster = SLURMCluster(
        processes=1,
        queue='hpg2-compute',
        cores=1, 
        memory='10GB', 
        walltime='48:00:00',
        job_extra=extra_args,
        local_directory="/home/b.weinstein/logs/", death_timeout=300)
    
    print(cluster.job_script())
    cluster.scale(num_workers)
    
    dask_client = Client(cluster)
        
    #Start dask
    dask_client.run_on_scheduler(start_tunnel)  
    
    for site in data_paths:
        futures = dask_client.map(Generate.run, data_paths[site], site=site)
        wait(futures)

if __name__ == "__main__":
    
    #Local Debugging
    data_paths=find_csvs()
    
    #Optionally limit
    #data_paths = data_paths[:100]
    total_files = [len(data_paths[x]) for x in data_paths]
    print("{s} csv files found for training".format(s=sum(total_files)))
    
    #run_test(data_paths)
    
    #On Hypergator
    run_HPC(data_paths)
