# Evaluation module
"""The goal of the evaluation module is to allow evalution metrics for each dataset"""
import glob
import pandas as pd
import os

def box_wrapper(m):
    eval_csvs = {
    #"Radogoshi_Sweden":"/blue/ewhite/DeepForest/Radogoshi_Sweden/images/test.csv",
    #"NeonTreeEvaluation":"/orange/idtrees-collab/NeonTreeEvaluation/evaluation/RGB/benchmark_annotations.csv",
    #"Beloiu_2023": "/blue/ewhite/DeepForest/Beloiu_2023/images/test.csv",
    #"justdigit-drone": "/blue/ewhite/DeepForest/justdiggit-drone/label_sample/test.csv",
    #"ReForestTree": "/blue/ewhite/DeepForest/ReForestTree/images/test.csv",
    "Siberia": "/blue/ewhite/DeepForest/Siberia/orthos/test.csv"
    }

    dfs = []
    for name in eval_csvs:
        print(name)
        csv_file = eval_csvs[name]
        rootdir = os.path.dirname(csv_file)
        results = m.evaluate(csv_file, rootdir, savedir="plots")
        results = {key:value for key, value in results.items() if key in ["box_precision","box_recall"]}
        dfs.append(results)
    dfs = pd.DataFrame(dfs)
    
    return dfs
    
def point_wrapper(m):
    eval_csvs = {
        "Ventura_2022":"/blue/ewhite/DeepForest/Ventura_2022/urban-tree-detection-data/images/test.csv",
        "TreeFormer":"/blue/ewhite/DeepForest/TreeFormer/all_images/test.csv"
    }

    dfs = []
    for name in eval_csvs:
        print(name)
        csv_file = eval_csvs[name]
        rootdir = os.path.dirname(csv_file)
        results = m.point_recall(csv_file, rootdir, savedir="plots")
        results = {key:value for key, value in results.items() if key in ["box_precision","box_recall"]}
        dfs.append(results)
    dfs = pd.DataFrame(dfs)
    
    return dfs