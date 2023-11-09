from src.utilities import *
import random
import glob
from scipy.io import loadmat

def Beloiu_2023():
    annotations = read_Beloiu_2023()
    annotations["label"] = "Tree"
    print("There are {} annotations in {} images".format(annotations.shape[0], len(annotations.image_path.unique())))
    images = annotations.image_path.unique()
    random.shuffle(images)
    train_images = images[0:int(len(images)*0.9)]
    train = annotations[annotations.image_path.isin(train_images)]
    test = annotations[~(annotations.image_path.isin(train_images))]
    train.to_csv("/blue/ewhite/DeepForest/Beloiu_2023/images/train.csv")
    test.to_csv("/blue/ewhite/DeepForest/Beloiu_2023/images/test.csv")

def Siberia():
    annotations = read_Siberia()
    print("There are {} annotations in {} images".format(annotations.shape[0], len(annotations.image_path.unique())))
    images = annotations.image_path.unique()
    random.shuffle(images)
    train_images = images[0:int(len(images)*0.9)]
    train = annotations[annotations.image_path.isin(train_images)]
    test = annotations[~(annotations.image_path.isin(train_images))]
    train.to_csv("/blue/ewhite/DeepForest/Siberia/orthos/train.csv")
    test.to_csv("/blue/ewhite/DeepForest/Siberia/orthos/test.csv")

def justdiggit():
    annotations = read_justdiggit("/blue/ewhite/DeepForest/justdiggit-drone/label_sample/Annotations_trees_only.json")
    print("There are {} annotations in {} images".format(annotations.shape[0], len(annotations.image_path.unique())))
    images = annotations.image_path.unique()
    random.shuffle(images)
    train_images = images[0:int(len(images)*0.8)]
    train = annotations[annotations.image_path.isin(train_images)]
    test = annotations[~(annotations.image_path.isin(train_images))]

    train["label"] = "Tree"
    test["label"] = "Tree"

    train.to_csv("/blue/ewhite/DeepForest/justdiggit-drone/label_sample/train.csv")
    test.to_csv("/blue/ewhite/DeepForest/justdiggit-drone/label_sample/test.csv")

def ReForestTree():
    """This dataset used deepforest to generate predictions which were cleaned, no test data can be used"""
    annotations = pd.read_csv("/blue/ewhite/DeepForest/ReForestTree/mapping/final_dataset.csv")
    annotations["image_path"] = annotations["img_path"]
    print("There are {} annotations in {} images".format(annotations.shape[0], len(annotations.image_path.unique())))
    annotations.to_csv("/blue/ewhite/DeepForest/ReForestTree/images/train.csv")

def Treeformer():
    def convert_mat(path):
        f = loadmat(x)
        points = f["image_info"][0][0][0][0][0]
        df = pd.DataFrame(points,columns=["x","y"])
        df["label"] = "Tree"
        image_path = "_".join(os.path.splitext(os.path.basename(x))[0].split("_")[1:])
        image_path = "{}.jpg".format(image_path)
        image_dir = os.path.dirname(os.path.dirname(x))
        df["image_path"] = image_path
        return df

    test_gt = glob.glob("/blue/ewhite/DeepForest/TreeFormer/test_data/ground_truth/*.mat")
    test_ground_truth = []
    for x in test_gt:
        df = convert_mat(x)
        test_ground_truth.append(df)
    test_ground_truth = pd.concat(test_ground_truth)

    train_gt = glob.glob("/blue/ewhite/DeepForest/TreeFormer/train_data/ground_truth/*.mat")
    train_ground_truth = []
    for x in train_gt:
        df = convert_mat(x)
        train_ground_truth.append(df)
    train_ground_truth = pd.concat(train_ground_truth)
    
    val_gt = glob.glob("/blue/ewhite/DeepForest/TreeFormer/valid_data/ground_truth/*.mat")
    val_ground_truth = []
    for x in val_gt:
        df = convert_mat(x)
        val_ground_truth.append(df)
    val_ground_truth = pd.concat(val_ground_truth)

    test_ground_truth.to_csv("/blue/ewhite/DeepForest/TreeFormer/all_images/test.csv")
    train_ground_truth.to_csv("/blue/ewhite/DeepForest/TreeFormer/all_images/train.csv")
    val_ground_truth.to_csv("/blue/ewhite/DeepForest/TreeFormer/all_images/validation.csv")

def Ventura():
    raise ("This is not complete, need to account for across year repeats")
    """In the current conception, using all Ventura data and not comparing against the train-test split"""
    all_csvs = glob.glob("/blue/ewhite/DeepForest/Ventura_2022/urban-tree-detection-data/csv/*.csv")
    df = []
    for x in all_csvs:
        points = pd.read_csv(x)
        points["image_path"] = os.path.splitext(os.path.basename(x))[0]
        df.append(points)
    annotations = pd.concat(df)
    annotations["label"] = "Tree"

    images = annotations.image_path.unique()
    random.shuffle(images)
    train_images = images[0:int(len(images)*0.8)]
    train = annotations[annotations.image_path.isin(train_images)]
    test = annotations[~(annotations.image_path.isin(train_images))]

    train["label"] = "Tree"
    test["label"] = "Tree"
                         
    train.to_csv("/blue/ewhite/DeepForest/Ventura_2022/urban-tree-detection-data/images/train.csv")
    test.to_csv("/blue/ewhite/DeepForest/Ventura_2022/urban-tree-detection-data/images/test.csv")

# Uncomment to regenerate each dataset
#Beloiu_2023()
#Siberia()
#justdiggit()
#ReForestTree()
#Treeformer()
Ventura()
