import glob
from DeepForest import Generate,config
from dask import compute, delayed

#find training csvs
DeepForest_config=config.load_config("train")
data_paths=glob.glob(DeepForest_config['training_csvs']+"/*.csv")

#Local threading/processes, set scheduler.
#Compute tiles
values = [delayed(Generate.run)(x) for x in data_paths]
results = compute(*values, scheduler='processes')

#from dask.distributed import Client
client = Client(scheduler_file='scheduler.json')

#Start juypter notebook to watch
import socket
host = client.run_on_scheduler(socket.gethostname)

def start_jlab(dask_scheduler):
    import subprocess
    proc = subprocess.Popen(['/path/to/jupyter', 'lab', '--ip', host, '--no-browser'])
    dask_scheduler.jlab_proc = proc

client.run_on_scheduler(start_jlab)

results.visualize()
#Hipergator 
#Loca
#
#import dask.multiprocessing
#results = compute(*values, scheduler='processes')
#print(data_paths)
#results = [Generate.run(x) for x in data_paths]


