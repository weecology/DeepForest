import laspy
from pointcloud import CloudInfo
class Lidar:
    def __init__(self,filename):
        self.cloud=CloudInfo(filename)
    
    def plot(self):
        pass
        