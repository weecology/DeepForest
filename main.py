#entry point for deepforest model

class deepforest:
    """Class for training and predicting tree crowns in RGB images
    
    Args:
    Attributes:
    """
    
    def __init__(saved_model=None):
        # Read config file - if a config file exists in local dir use it,
        # if not use installed.
        if os.path.exists("deepforest_config.yml"):
            config_path = "deepforest_config.yml"
        else:
            try:
                config_path = get_data("deepforest_config.yml")
            except Exception as e:
                raise ValueError(
                    "No deepforest_config.yml found either in local "
                    "directory or in installed package location. {}".format(e))

        print("Reading config file: {}".format(config_path))
        self.config = utilities.read_config(config_path)        
        
        # release version id to flag if release is being used
        self.__release_version__ = None
        
        if saved_model:
            utilities.load_saved_model(saved_model)
    
    def use_release(self, gpus=1):
        """Use the latest DeepForest model release from github and load model.
        Optionally download if release doesn't exist.
        Returns:
            model (object): A trained keras model
            gpus: number of gpus to parallelize, default to 1
        """
        # Download latest model from github release
        release_tag, self.weights = utilities.use_release()

        # load saved model and tag release
        self.__release_version__ = release_tag
        print("Loading pre-built model: {}".format(release_tag))
        
    def predict_image():
        """Predict an image with a deepforest model"""
        pass
    
    def predict_raster():
        pass
    
    def load_dataset(csv):
        pass
    
    def evaluate(csv):
        pass
    
        
    
