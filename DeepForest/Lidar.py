import laspy

class Lidar:
    def __init__(self,filename):
        self.filename=filename
        self.tile = laspy.file.File(filename)        