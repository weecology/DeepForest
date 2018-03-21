from dask_jobqueue import SLURMCluster

cluster = SLURMCluster(project='ewhite')
cluster.start_workers(3)

from dask.distributed import Client
client = Client(cluster)