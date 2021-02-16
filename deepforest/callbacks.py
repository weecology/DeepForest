# A deepforest callback has the following methods

class Callback():
    def before_fit(self):
        return True
    def after_fit(self):
        return True
    def before_epoch(self, epoch):
        return True
    def after_epoch(self, epoch):
        return True
    
        