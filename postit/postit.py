
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
    python3 balloon.py train --dataset=/path/to/balloon/dataset --weights=coco
    # Resume training a model that you had trained earlier
    python3 balloon.py train --dataset=/path/to/balloon/dataset --weights=last
    # Train a new model starting from ImageNet weights
    python3 balloon.py train --dataset=/path/to/balloon/dataset --weights=imagenet
    # Apply color splash to an image
    python3 balloon.py splash --weights=/path/to/weights/file.h5 --image=<URL or path to file>
    # Apply color splash to video using the last weights you trained
    python3 balloon.py splash --weights=last --video=<URL or path to file>
"""

# Root directory of the project
ROOT_DIR = os.path.abspath("../")

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

    NUM_CLASSES = 1 + 1 # Background + postit

    STEPS_PER_EPOCH = 100

    DETECTION_MIN_CONFIDENCE = 0.9

############################################################
#  Dataset
############################################################

class PostDataset(utils.Dataset):

    def load_postit(self, dataset_dir, subset):
        """lead bollon dataset"""

        #add class
        self.add_class("postit", 1, "postit")

        #load annotations


############################################################
#  Training
############################################################