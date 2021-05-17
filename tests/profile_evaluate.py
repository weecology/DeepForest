#Profile the dataset class
from deepforest import evaluate
from deepforest import main
from deepforest import get_data
import pandas as pd
import os
import cProfile, pstats

def run(m):  

    csv_file = get_data("OSBS_029.csv")
    predictions = m.predict_file(csv_file=csv_file, root_dir=os.path.dirname(csv_file))
    predictions.label = "Tree"
    ground_truth = pd.read_csv(csv_file)
    results = evaluate.evaluate(predictions=predictions, ground_df=ground_truth, root_dir=os.path.dirname(csv_file), savedir=None)     
    
if __name__ == "__main__":
    m = main.deepforest()
    m.use_release()
    
    profiler = cProfile.Profile()
    profiler.enable()
    m = main.deepforest()
    m.use_release()
    run(m)
    profiler.disable()
    stats = pstats.Stats(profiler).sort_stats('cumtime')
    stats.print_stats()    
    stats.dump_stats('evaluate.prof')
    