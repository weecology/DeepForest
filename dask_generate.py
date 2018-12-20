import glob
import subprocess
import socket
import os
import sys

import logging
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)

from DeepForest import Generate,config
from dask import compute, delayed

def find_csvs():
    """
    Find training csvs on path
    """
    DeepForest_config = config.load_config("train")
    data_paths = glob.glob(DeepForest_config['training_csvs']+"/*.csv")
    
    return data_paths

def run_local(data_paths):
    """
    Run training processes on local laptop
    """
    
    ##Local threading/processes, set scheduler.
    #values = [delayed(Generate.run)(x) for x in data_paths]
    
    ##Compute tiles    
    #results = compute(*values, scheduler='processes')    
        
def start_tunnel():
    """
    Start a juypter session and ssh tunnel to view task progress
    """
    host = socket.gethostname()        
    print("To tunnel into dask dashboard:")
    print("ssh -N -L 8787:%s:8787 -l b.weinstein hpg2.rc.ufl.edu" % (host))
    
    #Unset env
    #del os.environ['XDG_RUNTIME_DIR']
    #proc = subprocess.Popen(['jupyter', 'lab', '--notebook-dir', '/home/b.weinstein/logs/', '--ip', host, '--no-browser'])
    
    #flush system
    sys.stdout.flush()

def run_HPC(data_paths):
        
    #################
    # Setup dask cluster
    #################
    
    from dask_jobqueue import SLURMCluster
    from dask.distributed import Client
    from dask import delayed
    
    DeepForest_config = config.load_config("train")
    
    num_workers=DeepForest_config["num_hipergator_workers"]
    
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
        memory='15GB', 
        walltime='48:00:00',
        job_extra=extra_args,
        local_directory="/home/b.weinstein/logs/",death_timeout=300)
    
    print(cluster.job_script())
    cluster.scale(num_workers)
    
    dask_client = Client(cluster)
        
    #Start dask dashboard? Not clear yet.
    dask_client.run_on_scheduler(start_tunnel)  
        
    ### Local threading/processes, set scheduler.
    values = [delayed(Generate.run)(x) for x in data_paths]
    
    ###Compute tiles    
    try:
        compute(*values,scheduler='distributed')    
    except Exception as e:
        print(e)
    
    #futures = dask_client.map(Generate.run, data_paths)


if __name__ == "__main__":
    
    #Local Debugging
    data_paths=find_csvs()

    print("{s} csv files found for training".format(s=len(data_paths)))
    
    #run_local(data_paths)
    
    #On Hypergator
    run_HPC(data_paths)
    