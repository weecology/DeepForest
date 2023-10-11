# Evaluation module
"""The goal of the evaluation module is to allow evalution metrics for each dataset"""
from deepforest import evaluate
from deepforest import main
import glob
import pandas as pd
import os

def wrapper(m):
    eval_csvs = {
    #"Radogoshi_Sweden":"/blue/ewhite/DeepForest/Radogoshi_Sweden/test.csv",
    #"NeonTreeEvaluation":"/orange/idtrees-collab/NeonTreeEvaluation/evaluation/RGB/benchmark_annotations.csv",
    "Beloiu_2023": "/blue/ewhite/DeepForest/Beloiu_2023/test.csv",
    "NeonTreeEvaluation_local":"/Users/benweinstein/Documents/NeonTreeEvaluation/evaluation/RGB/benchmark_annotations.csv"}
    dfs = []
    for name in eval_csvs:
        try:
            csv_file = eval_csvs[name]
            rootdir = os.path.dirname(csv_file)
            results = m.evaluate(csv_file, rootdir)
            results = {key:value for key, value in results.items() if key in ["box_precision","box_recall"]}
        except Exception as e:
            print(e)
            continue
        dfs.append(results)
    dfs = pd.DataFrame(dfs)

    return dfs
    
