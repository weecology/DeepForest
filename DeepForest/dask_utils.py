from dask_jobqueue import SLURMCluster
from dask.distributed import Client
from dask import delayed

def start_dask(workers):
    
    ######################################################
    # Setup dask cluster
    ######################################################
    
    cluster = SLURMCluster(processes=1,queue='hpg2-compute', threads=2, memory='4GB', walltime='144:00:00')
    
    print('Starting up workers')
    workers = []
    for _ in range(config.num_hipergator_workers):
        workers.extend(cluster.start_workers(1))
        sleep(60)
    dask_client = Client(cluster)
    
    wait_time=0
    while len(dask_client.scheduler_info()['workers']) < config.num_hipergator_workers:
        print('waiting on workers: {s} sec. so far'.format(s=wait_time))
        sleep(10)
        wait_time+=10
        
        # If 5 minutes goes by try adding them again
        if wait_time > 300:
            workers.extend(cluster.start_workers(1))
    
    print('All workers accounted for')
    # xr import must be after dask.array, and I think after setup
    # up the cluster/client. 
    import dask.array as da
    import xarray as xr
