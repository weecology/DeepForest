#Profile the dataset class on gpu
from deepforest import main
from deepforest import get_data
import os
import pandas as pd
import numpy as np
import cProfile, pstats
import tempfile
from PIL import Image
import cv2

def run(m, csv_file, root_dir):  
    predictions = m.predict_file(csv_file=csv_file, root_dir=root_dir)
    
if __name__ == "__main__":
    m = main.deepforest()
    m.use_release()
    
    csv_file = get_data("OSBS_029.csv")
    image_path = get_data("OSBS_029.png")
    tmpdir = tempfile.gettempdir()    
    df = pd.read_csv(csv_file)    
    
    big_frame = []
    for x in range(10):
        img = Image.open("{}/{}".format(os.path.dirname(csv_file), df.image_path.unique()[0]))
        cv2.imwrite("{}/{}.png".format(tmpdir, x), np.array(img))
        new_df = df.copy()
        new_df.image_path = "{}.png".format(x)
        big_frame.append(new_df)
    
    big_frame = pd.concat(big_frame)
    big_frame.to_csv("{}/annotations.csv".format(tmpdir))
    
    profiler = cProfile.Profile()
    profiler.enable()
    run(m, csv_file = "{}/annotations.csv".format(tmpdir), root_dir = tmpdir)
    profiler.disable()
    stats = pstats.Stats(profiler).sort_stats('cumtime')
    stats.print_stats()    
    stats.dump_stats('predict_file.prof')