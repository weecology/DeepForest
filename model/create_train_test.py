from src.utilities import *
import random

def Beloiu_2023():
    annotations = read_Beloiu_2023()
    print("There are {} annotations in {} images".format(annotations.shape[0], len(annotations.image_path.unique())))
    images = annotations.image_path.unique()
    random.shuffle(images)
    train_images = images[0:int(len(images)*0.9)]
    train = annotations[annotations.image_path.isin(train_images)]
    test = annotations[~(annotations.image_path.isin(train_images))]
    train.to_csv("/blue/ewhite/DeepForest/Beloiu_2023/train.csv")
    test.to_csv("/blue/ewhite/DeepForest/Beloiu_2023/test.csv")

def Siberia():
    annotations = read_Siberia()
    print("There are {} annotations in {} images".format(annotations.shape[0], len(annotations.image_path.unique())))
    images = annotations.image_path.unique()
    random.shuffle(images)
    train_images = images[0:int(len(images)*0.9)]
    train = annotations[annotations.image_path.isin(train_images)]
    test = annotations[~(annotations.image_path.isin(train_images))]
    train.to_csv("/blue/ewhite/DeepForest/Siberia/train.csv")
    test.to_csv("/blue/ewhite/DeepForest/Siberia/test.csv")


def justdiggit():
    train = read_justdiggit("/blue/ewhite/DeepForest/justdiggit-drone/traintest_original/trainannotations.coco.json")
    test = read_justdiggit("/blue/ewhite/DeepForest/justdiggit-drone/traintest_original/testannotations.coco.json")
    train.to_csv("/blue/ewhite/DeepForest/justdiggit/train.csv")
    test.to_csv("/blue/ewhite/DeepForest/justdiggit/test.csv")

# Uncomment to regenerate each dataset
#Beloiu_2023()
#Siberia()
justdiggit()