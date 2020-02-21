from deepforest import deepforest
from deepforest import preprocess
from deepforest import utilities
from deepforest import tfrecords
from deepforest import get_data

annotations = utilities.xml_to_annotations(get_data("SOAP_061.xml"))
annotations.image_path = annotations.image_path.str.replace(".tif",".png")
annotations_file = get_data("testfile_multi.csv")
annotations.to_csv(annotations_file,index=False,header=False)

test_model = deepforest.deepforest()
#test_model.use_release()
test_model.config["epochs"] = 1000
test_model.config["save-snapshot"] = False
test_model.config["steps"] = 1
#test_model.config["multi_gpu"] = 1
test_model.config["save_path"] = "/home/b.weinstein/logs/deepforest"
test_model.config["freeze_resnet"] = False
test_model.train(annotations=annotations_file, input_type="fit_generator")

# Test labels
labels = list(test_model.labels.values())
labels.sort()
target_labels = ["Dead","Alive"]
target_labels.sort()

assert labels == target_labels

image_path = get_data("SOAP_061.png")
image = test_model.predict_image(image_path= image_path, return_plot=True)
boxes = test_model.predict_image(image_path= image_path, return_plot=False)
assert not boxes.empty
boxes.label.value_counts()

test_model.evaluate_generator(annotations_file)