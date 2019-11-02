import os
import sys
import matplotlib.pyplot as plt
import skimage.io

# 1. Set directories
# Root directory of the project
ROOT_DIR = os.path.abspath("../bikeability_ConvNet/")
# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
# Import COCO config
sys.path.append(os.path.join(ROOT_DIR, "models/Mask_RCNN-master/samples/coco/"))  # To find local version
import coco
# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "models/Mask_RCNN-master/logs")
# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "models/Mask_RCNN-master/pretrained_models/mask_rcnn_coco.h5")
# Download COCO trained weights from Releases if needed
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)
# Directory of images to run detection on
IMAGE_DIR = os.path.join(ROOT_DIR, "data/test_gsv")


# 2. Configurations
class InferenceConfig(coco.CocoConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

config = InferenceConfig()
config.display()


# 3. Create Model and Load Trained Weights
# Create model object in inference mode.
model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)
# Load weights trained on MS-COCO
model.load_weights(COCO_MODEL_PATH, by_name=True)


# 4. COCO Class names
# Index of the class in the list is its ID. For example, to get ID of
# the teddy bear class, use: class_names.index('teddy bear')
class_names = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
               'bus', 'train', 'truck', 'boat', 'traffic light',
               'fire hydrant', 'stop sign', 'parking meter']


# 5. Run Object Detection
# Load a random image from the images folder
file_names = next(os.walk(IMAGE_DIR))[2]
with open("mask_results.csv", 'w') as file:
    for file_name in file_names:
        image = skimage.io.imread(os.path.join(IMAGE_DIR, file_name))
        # Run detection
        results = model.detect([image], verbose=1)
        # Save results
        r = results[0]
        file.write(file_name + ',' + r['rois'] + ',' + r['masks'] + ',' + r['class_ids'] + ',' + class_names + ',' + r['scores'])
        visualize.display_instances(file_name, image, r['rois'], r['masks'], r['class_ids'],
                                    class_names, r['scores'])
    