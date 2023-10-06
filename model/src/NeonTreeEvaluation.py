# Benchmark data for validation
import glob
import os
import pandas as pd
from deepforest import utilities
from deepforest import main

## Benchmark data for validation ##
def generate_benchmark(BENCHMARK_PATH):
    tifs = glob.glob(BENCHMARK_PATH + "evaluation/RGB/*.tif")
    xmls = [os.path.splitext(os.path.basename(x))[0] for x in tifs] 
    xmls = [os.path.join(BENCHMARK_PATH, "annotations", x) + ".xml" for x in xmls] 
    
    #Load and format xmls, not every RGB image has an annotation
    annotation_list = []   
    for xml_path in xmls:
        try:
            annotation = utilities.xml_to_annotations(xml_path)
        except:
            continue
        annotation_list.append(annotation)
    benchmark_annotations = pd.concat(annotation_list, ignore_index=True)      
    
    #save evaluation annotations
    fname = os.path.join(BENCHMARK_PATH + "evaluation/RGB/benchmark_annotations.csv")
    benchmark_annotations.to_csv(fname, index=False, header=None)

def predict_benchmark(m, PATH_TO_BENCHMARK_DATA, output_csv):  
    """Create a csv of predictions for the NeonTreeEvaluation Benchmark"""  
    files = glob.glob(PATH_TO_BENCHMARK_DATA)

    #Load model
    boxes_output = []
    for f in files:
        print(f)
        #predict plot image
        boxes = m.predict_image(f, show=False, return_plot=False)
        box_df = pd.DataFrame(boxes)
        
        #plot name
        plot_name = os.path.splitext(os.path.basename(f))[0]
        box_df["plot_name"] = plot_name
        boxes_output.append(box_df)
        
    boxes_output = pd.concat(boxes_output)

    #name columns and add to submissions folder
    boxes_output.columns = ["xmin","ymin","xmax","ymax","plot_name"]
    boxes_output = boxes_output.reindex(columns= ["plot_name","xmin","ymin","xmax","ymax"])    
    boxes_output.to_csv("../submissions/Weinstein_unpublished.csv",index=False)