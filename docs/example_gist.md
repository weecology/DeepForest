from matplotlib import pyplot as plt
from deepforest import deepforest
from deepforest import utilities
from deepforest import get_data
from deepforest import preprocess

#convert hand annotations from xml into retinanet format
YELL_xml = get_data("2019_YELL_2_541000_4977000_image_crop.xml")
annotation = utilities.xml_to_annotations(YELL_xml)
annotation.head()

#Write converted dataframe to file. Saved alongside the images
annotation.to_csv("deepforest/data/eval_example.csv", index=False)

#Find data on path
YELL_test = get_data("2019_YELL_2_541000_4977000_image_crop.tiff")
crop_dir = "tests/data/"
cropped_annotations= preprocess.split_raster(path_to_raster=YELL_test,
                                 annotations_file="deepforest/data/eval_example.csv",
                                 base_dir=crop_dir,
                                 patch_size=400,
                                 patch_overlap=0.05)
#View output
cropped_annotations.head()

#Write window annotations file without a header row, same location as the "base_dir" above.
eval_annotations_file= crop_dir + "cropped_example.csv"
cropped_annotations.to_csv(eval_annotations_file,index=False, header=None)

test_model = deepforest.deepforest()
test_model.use_release()

mAP = test_model.evaluate_generator(annotations=eval_annotations_file)
print("Mean Average Precision is: {:.3f}".format(mAP))


#convert hand annotations from xml into retinanet format
YELL_xml = get_data("2019_YELL_2_528000_4978000_image_crop2.xml")
annotation = utilities.xml_to_annotations(YELL_xml)
annotation.head()

#Write converted dataframe to file. Saved alongside the images
annotation.to_csv("deepforest/data/train_example.csv", index=False)

#Find data on path
YELL_train = get_data("2019_YELL_2_528000_4978000_image_crop2.tiff")
crop_dir = "tests/data/"
train_annotations= preprocess.split_raster(path_to_raster=YELL_train,
                                 annotations_file="deepforest/data/train_example.csv",
                                 base_dir=crop_dir,
                                 patch_size=400,
                                 patch_overlap=0.05)
#View output
train_annotations.head()

#Write window annotations file without a header row, same location as the "base_dir" above.
annotations_file= crop_dir + "train_example.csv"
train_annotations.to_csv(annotations_file,index=False, header=None)

#Load the latest release
test_model = deepforest.deepforest()
test_model.use_release()

# Example run with short training
test_model.config["epochs"] = 2
test_model.config["save-snapshot"] = False
test_model.train(annotations=annotations_file, input_type="fit_generator")

mAP = test_model.evaluate_generator(annotations=eval_annotations_file)
print("Mean Average Precision is: {:.3f}".format(mAP))
