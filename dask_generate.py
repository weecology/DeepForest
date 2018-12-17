import glob
from DeepForest import Generate,config
from dask import compute, delayed

import subprocess
import socket

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
    
    #Local threading/processes, set scheduler.
    values = [delayed(Generate.run)(x) for x in data_paths]
    
    #Compute tiles    
    results = compute(*values, scheduler='processes')    
    
    return results
    
### HiperGator ####

#Start juypter notebook to watch
def start_tunnel(dask_scheduler):
    """
    Start a juypter session and ssh tunnel to view task progress
    """
    
    host = client.run_on_scheduler(socket.gethostname)    
    proc = subprocess.Popen(['jupyter', 'lab', '--ip', host, '--no-browser'])
    print("To tunnel into dask dashboard:")
    print("ssh -N -L 8888:%s:8888 -l b.weinstein hpg2.rc.ufl.edu" % (host))
    
def run_HPC(data_paths):
        
    #################
    # Setup dask cluster
    #################
    
    from dask_jobqueue import SLURMCluster
    from dask.distributed import Client
    from dask import delayed
    cluster = SLURMCluster(processes=2,queue='hpg2-compute',cores=2, memory='8GB', walltime='144:00:00')
    
    #Load config
    DeepForest_config = config.load_config("train")
    
    print('Starting up workers')
    workers = []
    for _ in range(DeepForest_config.num_hipergator_workers):
        workers.extend(cluster.scale(1))
        sleep(60) #How long to wait?
    dask_client = Client(cluster)
    
    wait_time=0
    while len(dask_client.scheduler_info()['workers']) < DeepForest_config.num_hipergator_workers:
        print('waiting on workers: {s} sec. so far'.format(s=wait_time))
        sleep(10)
        wait_time+=10
        
        # If 5 minutes goes by try adding them again
        if wait_time > 300:
            workers.extend(cluster.start_workers(1))
    
    print('All workers accounted for')
    
    #Start dask dashboard? Not clear yet.
    client.run_on_scheduler(start_tunnel)    
    #start_tunnel(dask_scheduler)
    
    # Local threading/processes, set scheduler.
    values = [delayed(Generate.run)(x) for x in data_paths]
    
    #Compute tiles    
    results = compute(*values, scheduler='processes')    
    return results

if __name__ == "__main__":
    
    #Local Debugging
    data_paths=find_csvs()[:2]

    print("{s} csv files found for training".format(s=len(data_paths)))
    
    #run_local(data_paths)
    
    #On Hypergator
    run_HPC(data_paths)
    