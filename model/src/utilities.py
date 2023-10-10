# Utilities Module
import argparse
import yaml
import json
import xml.etree.ElementTree as ET
import pandas as pd
import glob

def read_config(config_path):
    """Read config yaml file"""
    #Allow command line to override 
    parser = argparse.ArgumentParser("DeepTreeAttention config")
    parser.add_argument('-d', '--my-dict', type=json.loads, default=None)
    args = parser.parse_known_args()
    try:
        with open(config_path, 'r') as f:
            config = yaml.load(f, Loader=yaml.FullLoader)

    except Exception as e:
        raise FileNotFoundError("There is no config at {}, yields {}".format(
            config_path, e))
    
    #Update anything in argparse to have higher priority
    if args[0].my_dict:
        for key, value in args[0].my_dict:
            config[key] = value
        
    return config

def read_xml_Beloiu(path):
    tree = ET.parse(path)
    root = tree.getroot()

    # Initialize lists to store data
    filename_list = []
    name_list = []
    xmin_list = []
    ymin_list = []
    xmax_list = []
    ymax_list = []

    # Iterate through each 'object' element
    for obj in root.findall('.//object'):
        filename = root.find('.//filename').text
        name = obj.find('name').text
        xmin = float(obj.find('bndbox/xmin').text)
        ymin = float(obj.find('bndbox/ymin').text)
        xmax = float(obj.find('bndbox/xmax').text)
        ymax = float(obj.find('bndbox/ymax').text)
        
        # Append data to lists
        filename_list.append(filename)
        name_list.append(name)
        xmin_list.append(xmin)
        ymin_list.append(ymin)
        xmax_list.append(xmax)
        ymax_list.append(ymax)

    # Create a DataFrame
    data = {
        'image_path': filename_list,
        'name': name_list,
        'xmin': xmin_list,
        'ymin': ymin_list,
        'xmax': xmax_list,
        'ymax': ymax_list
    }

    df = pd.DataFrame(data)

    return df

def read_Beloiu_2023():
    xmls = glob.glob("/blue/ewhite/DeepForest/Beloiu_2023/labels/*")

    annotations = []
    for path in xmls:
        df = read_xml_Beloiu(path)

    annotations.append(df)
    annotations = pd.concat(annotations)
    