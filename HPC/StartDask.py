#srun --ntasks=1 --cpus-per-task=2 --mem=2gb -t 90 --pty bash -i

from dask_jobqueue import SLURMCluster
from datetime import datetime
from time import sleep

cluster = SLURMCluster(project='ewhite',death_timeout=100)
cluster.start_workers(1)

print(cluster.job_script())

from dask.distributed import Client
client = Client(cluster)

client

counter=0
while counter < 10:
    print(datetime.now().strftime("%a, %d %B %Y %I:%M:%S"))
    print(client)
    sleep(20)
    counter+=1

import socket
host = client.run_on_scheduler(socket.gethostname)

def start_jlab(dask_scheduler):
    import subprocess
    proc = subprocess.Popen(['jupyter', 'lab', '--ip', host, '--no-browser'])
    dask_scheduler.jlab_proc = proc

client.run_on_scheduler(start_jlab)

print("ssh -N -L 8787:%s:8787 -L 8888:%s:8888 -l b.weinstein hpg2.rc.ufl.edu" % (host, host))