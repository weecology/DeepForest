import timeit
from DeepForest import Generate
from dask.distributed import Client
from dask import compute, delayed

#For debugging
import dask
dask.config.set(scheduler='synchronous')

#import dask.threaded
#results = compute(*values, scheduler='threads')

#import dask.multiprocessing
#results = compute(*values, scheduler='processes')

#Profiling
def function_to_time():
    results = [process(x) for x in inputs]

#Start dask cluster
client = Client("cluster-address:8786")
results = compute(*values, scheduler='distributed')

#Compute tiles
values = [delayed(process)(x) for x in inputs]
