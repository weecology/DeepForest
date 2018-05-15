'''
Util functions, including dask clients
'''

from dask_jobqueue import SLURMCluster
from dask.distributed import Client
from dask import delayed

import geojson

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

def load_config(data_folder=None):
    with open('_config.yaml', 'r') as f:
        config = yaml.load(f)

    if data_folder is None:
        data_folder = config['data_folder']
        
def data2geojson(df):
    features = []
    insert_features = lambda X: features.append(
            geojson.Feature(geometry=geojson.Polygon([
                (float(X["xmin"]),float(X["ymin"])),
                (float(X["xmax"]),float(X["ymin"])),
                (float(X["xmax"]),float(X["ymax"])),
                (float(X["xmin"]),float(X["ymax"]))],
                properties=dict(name=str(X["box"])))))
    df.apply(insert_features, axis=1)
    return geojson.FeatureCollection(features)
