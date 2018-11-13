srun --ntasks=1 --cpus-per-task=2 --mem=2gb -t 90 --pty bash -i

import getpass
import socket
import subprocess
host = socket.gethostname()
proc = subprocess.Popen(['jupyter', 'lab', '--ip', host, '--no-browser'])

print("ssh -N -L 8787:{0}:8787 -L 8888:{0}:8888 -l {1} hpg2.rc.ufl.edu".format(host, getpass.getuser()))