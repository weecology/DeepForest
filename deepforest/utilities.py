#utility functions for demo
import os
import yaml
import sys
from tqdm import tqdm
import json
import urllib
from keras_retinanet import models
def label_to_name(label):
        """ Map label to name.
        """
        return "Tree"

def read_config():
        try:
                with open("deepforest_config.yml", 'r') as f:
                        config = yaml.load(f)
        except Exception as e:
                raise FileNotFoundError("There is no config file in dir:{}, yields {}".format(os.getcwd(),e))
                
        return config

def read_model(model_path, config):
        model = models.load_model(model_path, backbone_name='resnet50')
        return model

#Download progress bar
class DownloadProgressBar(tqdm):
        def update_to(self, b=1, bsize=1, tsize=None):
                if tsize is not None:
                        self.total = tsize
                self.update(b * bsize - self.n)
                
def use_release():
        '''
        Check the existance of, or download the latest model release from github
        
        Returns:
                output_path (str): path to downloaded model weights'''
        #Find latest github tag release from the DeepLidar repo
        _json = json.loads(urllib.request.urlopen(urllib.request.Request(
                'https://api.github.com/repos/Weecology/DeepForest/releases/latest',
            headers={'Accept': 'application/vnd.github.v3+json'},
             )).read())     
        asset = _json['assets'][0]
        output_path = os.path.join('data',asset['name'])    
        url = asset['browser_download_url']
        
        #Download if it doesn't exist
        if not os.path.exists(output_path):
                print("Downloading model from DeepForest release {}, see {} for details".format(_json["tag_name"],_json["html_url"]))                
                with DownloadProgressBar(unit='B', unit_scale=True,
                                     miniters=1, desc=url.split('/')[-1]) as t:
                        urllib.request.urlretrieve(url, filename=output_path, reporthook=t.update_to)           
        else:
                print("Model from DeepForest release {} was already downloaded. Loading model from file.".format(_json["html_url"]))
                
        return output_path