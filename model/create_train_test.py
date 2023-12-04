from src.utilities import *
import random
import glob
from scipy.io import loadmat
import geopandas as gpd
import rasterio
from deepforest.utilities import shapefile_to_annotations
from deepforest.preprocess import split_raster
from shapely.geometry import Point
import CHM

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
    train_images = annotations.image_path.drop_duplicates().sample(frac=0.8).values()
    test_images = [x for x in images if x not in train_images]
    split_train_annotations = []
    for x in train_images:
        selected_annotations = split_train_annotations[split_train_annotations.image_path==x]
        split_annotations = split_raster(
            annotations_file=selected_annotations,
            path_to_raster=x,
            patch_size=600,
            save_dir="/blue/ewhite/DeepForest/Siberia/images")
    split_train_annotations = pd.concat(split_train_annotations)

    split_test_annotations = []
    for x in test_images:
        selected_annotations = split_test_annotations[split_test_annotations.image_path==x]
        split_annotations = split_raster(
            annotations_file=selected_annotations,
            path_to_raster=x,
            patch_size=600,
            save_dir="/blue/ewhite/DeepForest/Siberia/images")
    split_test_annotations = pd.concat(split_test_annotations)

    #Cut into pieces 
    split_train_annotations.to_csv("/blue/ewhite/DeepForest/Siberia/images/train.csv")
    split_test_annotations.to_csv("/blue/ewhite/DeepForest/Siberia/images/test.csv")

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
    """In the current conception, using all Ventura data and not comparing against the train-test split"""
    all_csvs = glob.glob("/blue/ewhite/DeepForest/Ventura_2022/urban-tree-detection-data/csv/*.csv")
    train_images = pd.read_csv("/blue/ewhite/DeepForest/Ventura_2022/urban-tree-detection-data/train.txt",header=None,sep=" ")[0].values
    test_images = pd.read_csv("/blue/ewhite/DeepForest/Ventura_2022/urban-tree-detection-data/test.txt",header=None,sep=" ")[0].values
    val_images = pd.read_csv("/blue/ewhite/DeepForest/Ventura_2022/urban-tree-detection-data/val.txt",header=None,sep=" ")[0].values

    df = []
    for x in all_csvs:
        points = pd.read_csv(x)
        points["image_path"] = os.path.splitext(os.path.basename(x))[0]
        df.append(points)
    annotations = pd.concat(df)
    annotations["label"] = "Tree"

    train = annotations[annotations.image_path.isin(train_images)]
    test = annotations[annotations.image_path.isin(test_images)]
    val = annotations[annotations.image_path.isin(val_images)]
              
    train.to_csv("/blue/ewhite/DeepForest/Ventura_2022/urban-tree-detection-data/images/train.csv")
    test.to_csv("/blue/ewhite/DeepForest/Ventura_2022/urban-tree-detection-data/images/test.csv")
    val.to_csv("/blue/ewhite/DeepForest/Ventura_2022/urban-tree-detection-data/images/val.csv")

def Cloutier2023():
    # Zone 3 is test, Zone 1 and 2 is train. Intentionally vary window size.
    drone_flights = glob.glob("/blue/ewhite/DeepForest/Cloutier2023/**/*.tif",recursive=True)
    zone1 = "/blue/ewhite/DeepForest/Cloutier2023/quebec_trees_dataset_2021-06-17/Z1_polygons.gpkg"
    zone2 = "/blue/ewhite/DeepForest/Cloutier2023/quebec_trees_dataset_2021-06-17/Z2_polygons.gpkg"
    zone3 = "/blue/ewhite/DeepForest/Cloutier2023/quebec_trees_dataset_2021-06-17/Z3_polygons.gpkg"
    
    train = []
    test = []
    for flight in drone_flights:
            train_zone1 = shapefile_to_annotations(zone1, rgb=flight)
            split_annotations_1 = split_raster(train_zone1, path_to_raster=flight, patch_size=1000, allow_empty=True, base_dir="/blue/ewhite/DeepForest/Cloutier2023/images/")
            train_zone2 = shapefile_to_annotations(zone2, rgb=flight)
            split_annotations_2 = split_raster(train_zone2, path_to_raster=flight, patch_size=2000, allow_empty=True, base_dir="/blue/ewhite/DeepForest/Cloutier2023/images/")
            test_zone3 = shapefile_to_annotations(zone3, rgb=flight)
            split_annotations_3 = split_raster(test_zone3, path_to_raster=flight, patch_size=1500, allow_empty=True, base_dir="/blue/ewhite/DeepForest/Cloutier2023/images/")
            train.append(split_annotations_1)
            train.append(split_annotations_2)
            test.append(split_annotations_3)
    train = pd.concat(train)
    test = pd.concat(test)

    test.to_csv("/blue/ewhite/DeepForest/Cloutier2023/images/test.csv")
    train.to_csv("/blue/ewhite/DeepForest/Cloutier2023/images/train.csv")

def HemmingSchroeder():
    annotations = gpd.read_file("/blue/ewhite/DeepForest/HemmingSchroeder/data/training/trees_2017_training_filtered_labeled.shp")
    # There a many small plots, each with a handful of annotations
    plots = annotations.sampleid.unique()
    train_plots = annotations.sampleid.drop_duplicates().sample(frac=0.9)

    train = annotations[annotations.sampleid.isin(train_plots)]
    test = annotations[~(annotations.sampleid.isin(train_plots))]

    #Need to generate images from NEON data archive base on plot locations
    rgb_pool = glob.glob("/orange/ewhite/NeonData/*/DP3.30010.001/**/Camera/**/*.tif", recursive=True)
    plot_locations = gpd.read_file("/blue/ewhite/DeepForest/HemmingSchroeder/data/training/training_sample_sites.shp")
    plot_locations = plot_locations[plot_locations.sample_id.isin(annotations.sampleid)]
    
    year_annotations = []
    for plot_location in plot_locations.geometry:
            # Get rgb path for all years
            rgb_paths = find_sensor_path(bounds = plot_location.bounds, lookup_pool=rgb_pool, all_years=True)
            sampleid = plot_locations[plot_locations.geometry==plot_location]["sample_id"].values[0]
            sample_annotations = annotations[annotations.sampleid==sampleid]

            for rgb_path in rgb_paths:
                year = year_from_tile(rgb_path)
                basename = "{}_{}".format(sampleid, year)
                year_annotation = sample_annotations.copy()
                year_annotation["image_path"] = basename
                crop(
                    bounds=plot_location.bounds,
                    sensor_path=rgb_path,
                    savedir="/blue/ewhite/DeepForest/HemmingSchroeder/data/training/images/",
                    basename=basename
                    )
                year_annotations.append(year_annotation)

    year_annotations = pd.concat(year_annotations)
    # Remove basenames with FullSite
    year_annotations = year_annotations[~(year_annotations.basename.str.contains("FullSite"))]
    train_annotations = year_annotations[year_annotations.sampleid.isin(train.sampleid)]
    test_annotations = year_annotations[year_annotations.sampleid.isin(test.sampleid)]

    train_annotations.to_csv("/blue/ewhite/DeepForest/HemmingSchroeder/train.csv")
    test_annotations.to_csv("/blue/ewhite/DeepForest/HemmingSchroeder/test.csv")

#def ForestGEO():
#   """Point data from within the NEON sites"""
#    OSBS = gpd.read_file("/orange/idtrees-collab/megaplot/OSBS_megaplot.shp")
#    SERC = gpd.read_file("/orange/idtrees-collab/megaplot/SERC_megaplot.shp")
#    HARV = gpd.read_file("/orange/idtrees-collab/megaplot/HARV_megaplot.shp")

#    #Split into iamges. 
#    shapefile_to_annotations()
   
   
                
def NEON_Trees():
    """Transform raw NEON data into clean shapefile   
    Args:
        config: DeepTreeAttention config dict, see config.yml
    """
    field = pd.read_csv("/blue/ewhite/DeepForest/NEON_Trees/vst_nov_2023.csv")
    field["individual"] = field["individualID"]
    field = field[~field.growthForm.isin(["liana","small shrub"])]
    field = field[~field.growthForm.isnull()]
    field = field[~field.plantStatus.isnull()]        
    field = field[field.plantStatus.str.contains("Live")]  
    field.shape
    field = field[field.stemDiameter > 10]
    
    # Recover canopy position from any eventID
    groups = field.groupby("individual")
    shaded_ids = []
    for name, group in groups:
        shaded = any([x in ["Full shade", "Mostly shaded"] for x in group.canopyPosition.values])
        if shaded:
            if any([x in ["Open grown", "Full sun"] for x in group.canopyPosition.values]):
                continue
            else:
                shaded_ids.append(group.individual.unique()[0])
        
    field = field[~(field.individual.isin(shaded_ids))]
    field = field[(field.height > 3) | (field.height.isnull())]

    # Most Recent Year
    field = field.groupby("individual").apply(lambda x: x.sort_values(["eventID"],ascending=False).head(1)).reset_index(drop=True)
    field = field[~(field.eventID.str.contains("2014"))]
    field = field[~field.height.isnull()]

    # Remove multibole
    field = field[~(field.individual.str.contains('[A-Z]$',regex=True))]

    # List of hand cleaned errors
    known_errors = ["NEON.PLA.D03.OSBS.03422","NEON.PLA.D03.OSBS.03422","NEON.PLA.D03.OSBS.03382", "NEON.PLA.D17.TEAK.01883"]
    field = field[~(field.individual.isin(known_errors))]
    field = field[~(field.plotID == "SOAP_054")]

    #Create shapefile
    field["geometry"] = [Point(x,y) for x,y in zip(field["itcEasting"], field["itcNorthing"])]
    shp = gpd.GeoDataFrame(field)

    # CHM Filter
    CHM_pool = glob.glob("/orange/ewhite/NeonData/**/CanopyHeightModelGtif/*.tif",recursive=True)
    rgb_pool = glob.glob("/orange/ewhite/NeonData/**/DP3.30010.001/**/Camera/**/*.tif",recursive=True)

    #shp = CHM.filter_CHM(shp, CHM_pool)

    # BLAN has some data in 18N UTM, reproject to 17N update columns
    BLAN_errors = shp[(shp.siteID == "BLAN") & (shp.utmZone == "18N")]
    BLAN_errors.set_crs(epsg=32618, inplace=True)
    BLAN_errors.to_crs(32617,inplace=True)
    BLAN_errors["utmZone"] = "17N"
    BLAN_errors["itcEasting"] = BLAN_errors.geometry.apply(lambda x: x.coords[0][0])
    BLAN_errors["itcNorthing"] = BLAN_errors.geometry.apply(lambda x: x.coords[0][1])

    # reupdate
    shp.loc[BLAN_errors.index] = BLAN_errors

    # Oak Right Lab has no AOP data
    shp = shp[~(shp.siteID.isin(["PUUM","ORNL"]))]

    # Create a unique subplot ID
    shp = shp[~(shp["subplotID"].isnull())]
    shp["subID"] = shp["plotID"] + "_" + shp["subplotID"].astype(int).astype("str") 

    # For each subplot crop image and gather annotations
    plot_shp = gpd.read_file("/blue/ewhite/DeepForest/NEON_Trees/All_Neon_TOS_Polygon_V4.shp")
    annotations = []
    for plot in shp.plotID:
        subplot_annotations = shp[shp.plotID==plot]
        bounds = plot_shp[plot_shp.plotID==plot]
        utmZone = bounds.utmZone.unique()[0] 
        if utmZone == "6N":
            epsg = 32606
        elif utmZone=="5N":
            epsg=32605
        else:
            epsg = "326{}".format(int(utmZone[:2]))
        bounds = bounds.to_crs(epsg).total_bounds
        try:
            sensor_path = find_sensor_path(bounds=list(bounds), lookup_pool=rgb_pool)
            crop(
                bounds=bounds,
                sensor_path=sensor_path,
                savedir="/blue/ewhite/DeepForest/NEON_Trees/images/",
                basename=plot)
        except:
            continue
        annotations.append(subplot_annotations)
    
    #Split into train and test
    annotations = pd.concat(annotations)
    subplots = annotations.subID.unique()

    return shp

# Uncomment to regenerate each dataset
#Beloiu_2023()
#Siberia()
#justdiggit()
#ReForestTree()
#Treeformer()
#Ventura()
#Cloutier2023()
#HemmingSchroeder()
#ForestGEO()
NEON_Trees()

