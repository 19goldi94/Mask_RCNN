import os
import sys
import json
import datetime
import numpy as np
import skimage.draw

"""
Usage: import the module (see Jupyter notebooks for examples), or run from
       the command line as such:
    # Train a new model starting from pre-trained COCO weights
    python postit.py train --dataset=/path/to/postit/dataset --weights=coco
    # Resume training a model that you had trained earlier
    python postit.py train --dataset=/path/to/postit/dataset --weights=last
    # 
    python postit.py detect --weights=/path/to/weights/file.h5 --image=<URL or path to file>
    
python postit.py train --dataset=C:/Users/chris/Downloads/PostIt_Segmentation/images --weights=C:/Users/chris/Downloads/PostIt_Segmentation/mask_rcnn_coco.h5

    
"""

# Root directory of the project
ROOT_DIR = os.path.abspath("../")
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")
COCO_WEIGHTS_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import model as modellib, utils


############################################################
#  Configurations
############################################################

class PostConfig(Config):
    """Configuration for the training
    Derived from the base Config class"""

    NAME = "postit"

    IMAGES_PER_GPU = 2

    NUM_CLASSES = 1 + 1  # Background + postit

    STEPS_PER_EPOCH = 100

    DETECTION_MIN_CONFIDENCE = 0.9


############################################################
#  Dataset
############################################################

class PostDataset(utils.Dataset):

    def load_postit(self, dataset_dir, subset):

        # add class
        self.add_class("postit", 1, "postit")

        # subset has to be train or val

        assert subset in ["train", "val"]

        #dataset_dir = os.path.abspath("../../Coco")

        dataset_dir = os.path.join(dataset_dir, subset)
        #dataset_dir = os.path.join(dataset_dir, "train")

        # load json file
        data_json = json.load(open(os.path.join(dataset_dir, "output.json")))
        print(data_json)
        # iterates over all images in the json file

        for i in data_json:
            polygons = [g['geometry'] for g in i['Label']['Post It']]

            image_path = i["Labeled Data"]
            image = skimage.io.imread(image_path)
            height, width = image.shape[:2]

            self.add_image(
                "postit",
                image_id=i['ID'],
                path=image_path,
                width=width, height=height,
                polygons=polygons
            )

    def load_mask(self, image_id):

        image_info = self.image_info[image_id]
        if image_info["source"] != "postit":
            return super(self.__class__, self).load_mask(image_id)

        ##convert x, y coordinates to mask
        imginfo = self.image_info[image_id]

        mask = np.zeros([imginfo["height"], imginfo["width"], len(imginfo["polygons"])],
                        dtype=np.uint8)

        for i, p in enumerate(imginfo["polygons"]):
            x_values = []
            y_values = []
            for x in p:
                x_values.append(x['x'])
                y_values.append(x['y'])

            rr, cc = skimage.draw.polygon(y_values, x_values)
            mask[rr, cc, i] = 1

        return mask.astype(np.bool), np.ones([mask.shape[-1]], dtype=np.int32)

    def image_reference(self, image_id):

        info = self.image_info[image_id]
        if info["source"] == "postit":
            return info["path"]
        else:
            super(self.__class__, self).image_reference(image_id)

def train(model):

    # Training Dataset
    dataset_train = PostDataset()
    dataset_train.load_postit(args.dataset, "train")
    dataset_train.prepare()

    # Validation dataset
    dataset_val = PostDataset()
    dataset_val.load_postit(args.dataset, "val")
    dataset_val.prepare()

    print("Training network heads")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=30,
                layers='heads')

def detectpostit(image, image_path=None):

    r = model.detect([image], verbose=1)[0]
    m = r['masks']
    m = np.sum(m * np.arange(1, m.shape[-1] + 1), -1)
    
    for i in range(len(m)):
        for j in range(len(m[0])):
            if m[i][j] > 5:
                m[i][j] = 255
            else:
                m[i][j] = 0

    import imageio
    file_name = "postit_{:%Y%m%dT%H%M%S}.png".format(datetime.datetime.now())
    imageio.imwrite(file_name, m)


############################################################
#  Training
############################################################

if __name__ == '__main__':
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Train Mask R-CNN to detect post its.')
    parser.add_argument("command",
                        metavar="<command>",
                        help="'train' or 'detect'")
    parser.add_argument('--dataset', required=False,
                        metavar="/path/to/postit/dataset/",
                        help='Directory of the dataset')
    parser.add_argument('--weights', required=True,
                        metavar="/path/to/weights.h5",
                        help="Path to weights .h5 file")
    parser.add_argument('--logs', required=False,
                        default=DEFAULT_LOGS_DIR,
                        metavar="/path/to/logs/",
                        help='Logs and checkpoints directory (default=logs/)')
    parser.add_argument('--image', required=False,
                        metavar="path or URL to image",
                        help='Post it image')
    args = parser.parse_args()

    # Validate arguments
    if args.command == "train":
        assert args.dataset, "Argument --dataset is required for training"
    elif args.command == "detect":
        assert args.image or args.video, \
            "Provide --image to detect post its"

    print("Weights: ", args.weights)
    print("Dataset: ", args.dataset)
    print("Logs: ", args.logs)

    # Configurations
    if args.command == "train":
        config = PostConfig()
    else:
        class InferenceConfig(PostConfig):
            # Set batch size to 1 since we'll be running inference on
            # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
            GPU_COUNT = 1
            IMAGES_PER_GPU = 1


        config = InferenceConfig()
    config.display()

    # Create model
    if args.command == "train":
        model = modellib.MaskRCNN(mode="training", config=config,
                                  model_dir=args.logs)
    else:
        model = modellib.MaskRCNN(mode="inference", config=config,
                                  model_dir=args.logs)
    if args.weights.lower() == "coco":
        weights_path = COCO_WEIGHTS_PATH
        # Download weights file
        if not os.path.exists(weights_path):
            utils.download_trained_weights(weights_path)
    # Select weights file to load
    elif args.weights.lower() == "last":
        # Find last trained weights
        weights_path = model.find_last()
    else:
        weights_path = args.weights

    # Load weights
    print("Loading weights ", weights_path)
    if args.weights.lower() == "coco":
        # Exclude the last layers because they require a matching
        # number of classes
        model.load_weights(weights_path, by_name=True, exclude=[
            "mrcnn_class_logits", "mrcnn_bbox_fc",
            "mrcnn_bbox", "mrcnn_mask"])
    else:
        model.load_weights(weights_path, by_name=True)

    # Train or evaluate
    if args.command == "train":
        train(model)
    elif args.command == "detect":
        detectpostit(model, image_path=args.image)
    else:
        print("'{}' is not recognized. "
              "Use 'train' or 'detect'".format(args.command))
