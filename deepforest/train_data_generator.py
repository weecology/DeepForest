"""This module generates a .csv file called "combined_csv.csv", which will store annotations of all training images. Later on, "combined_csv.csv" will be used for re-training
the existing NEON.h5 model. 

To generate "combined_csv.csv" using this module, firstly create a directory "dataset" at any local system location. Then place all training images inside the "images" folder and all 
corresponding annotations present in the form of XML format (created by LabelImg or RectLabel) inside the "annotations" folder located inside directory "dataset" present at local system.

Then create a new python file, "get_data.py", and pass the location of the "dataset" directory with reference to the user's local system inside function "train_data_generator.prepare_traindata". 
Here "utilities" is imported through "from deepforest import deepforest", following with the "from deepforest import utilities" command in the python file.

On execution of the "get_data.py" python file, the module will return two folders, namely "train_annotations" and "training_images" inside the "dataset" directory.
The "train_annotations" folder contains "combined_csv.csv", and the "training_images" folder contains images referenced inside the "combined_csv.csv" file. 
These two files will help to re-train the existing model weights."""
 
import os,glob
import utilities
import preprocess
import pandas as pd




#Convert hand annotations from multiple xml files present in "annotations" folder (which is present inside "data" named directory at local system) into retinanet formatted 
#DataFrame which is further converted into seperate csv files corresponding to each xml file.
def xml_to_csv(location):
    folders_list = os.listdir(location+"/annotations")
    saving_location = location+"/xml_to_csv/"
    if not os.path.exists(saving_location):
        os.makedirs(saving_location)
    try: 
        for file in folders_list:
            name = file.split(".")
            name = name[0]
            source_location = location+"/annotations/"+str(file)
            annotation = utilities.xml_to_annotations(source_location)
            csv_file = "data"+name+".csv"
            annotations_file = os.path.join(saving_location, csv_file)
            annotation.to_csv(annotations_file,index=False)
    except Exception as e:
        print(" ")



#Extending "split_raster.py module" to every image and csv annotation so as to get a headerless csv file having annotations and image paths for re-training the existing model.
def split_raster_all(location):
    image_location = location+"/images"
    folders_list = os.listdir(image_location)
    
    saving_location=location+"/training_images"
    if not os.path.exists(saving_location):
        os.makedirs(saving_location)
    try:
        annotations_files = []
        for file in folders_list:
            name = file.split(".")
            name = name[0]
            annotation_file = location+"/xml_to_csv/data"+name+".csv"

            train_annotations= preprocess.split_raster(path_to_raster = location+"/images/"+name+".jpg",
                                             annotations_file = annotation_file,
                                             base_dir = saving_location,
                                             patch_size=400,
                                             patch_overlap=0.05)
            annotations_files.append(train_annotations)
    except Exception as e:
        print(" ")

    df = pd.concat(annotations_files)
    if not os.path.exists(location+"/training_annotations"):
        os.makedirs(location+"/training_annotations")
    df.to_csv(location+"/training_annotations/combined_csv.csv",index=False, header=None)
	
    
    
#Deleting extra intermediate files and folders so to free up local system memory space	
def delete_extra(location):
    files_in_directory = os.listdir(location+"/xml_to_csv")
    filtered_files = [file for file in files_in_directory]
    for file in filtered_files:
        os.remove(location+"/xml_to_csv/"+file)
    os.rmdir(location+"/xml_to_csv")        
    
    files_in_directory = os.listdir(location+"/training_images")  
    filtered_files = [file for file in files_in_directory if file.endswith(".csv")]
    for file in filtered_files:
        os.remove(location+"/training_images/"+file)



#Driver code to execute above three modules
def prepare_traindata(location):
    xml_to_csv(location)
    split_raster_all(location)
    delete_extra(location)
	
	
	
