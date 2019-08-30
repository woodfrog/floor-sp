"""
Mask R-CNN
Configurations and data loading code for MS COCO.

Copyright (c) 2017 Matterport, Inc.
Licensed under the MIT License (see LICENSE for details)
Written by Waleed Abdulla

------------------------------------------------------------

Usage: import the module (see Jupyter notebooks for examples), or run from
       the command line as such:

    # Train a new model starting from pre-trained COCO weights
    python3 main.py train --dataset=/path/to/coco/ --model=coco

    # Train a new model starting from ImageNet weights
    python3 main.py train --dataset=/path/to/coco/ --model=imagenet

    # Continue training a model that you had trained earlier
    python3 main.py train --dataset=/path/to/coco/ --model=/path/to/weights.h5

    # Continue training the last model you trained
    python3 main.py train --dataset=/path/to/coco/ --model=last

    # Run COCO evaluatoin on the last model you trained
    python3 main.py evaluate --dataset=/path/to/coco/ --model=last
"""

import os
import pdb
import numpy as np
from PIL import Image, ImageDraw
import skimage.io
import json
# Download and install the Python COCO tools from https://github.com/waleedka/coco
# That's a fork from the original https://github.com/pdollar/coco with a bug
# fix for Python 3.
# I submitted a pull request https://github.com/cocodataset/cocoapi/pull/50
# If the PR is merged then use the original repo.
# Note: Edit PythonAPI/Makefile and replace "python" with "python3".


from config import Config
import model.utils as utils
import model.model as modellib
import scipy.misc

import torch

# Root directory of the project
ROOT_DIR = os.getcwd()

# Path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "pretrained/mask_rcnn_coco.pth")

# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")

# change the base dir accordingly
DATASET_BASE_DIR = '/local-scratch/cjc/floor-sp/data/'

############################################################
#  Configurations
############################################################


class LianjiaConfig(Config):
    # Give the configuration a recognizable name
    NAME = "Lianjia_dataset"

    # We use one GPU with 8GB memory, which can fit one image.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 16

    # Uncomment to train on 8 GPUs (default is 1)
    # GPU_COUNT = 8

    # Number of classes (including background)
    # need to make sure this, the number of rooms + bg
    NUM_CLASSES = 15


class InferenceConfig(LianjiaConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    # GPU_COUNT = 0 for CPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1


############################################################
#  Dataset
############################################################

class LianjiaDataset(utils.Dataset):
    def load_samples(self, phase, metadata_path):
        self.phase = phase
        # Add classes
        if phase != 'train' and phase != 'test':
            raise ValueError('Invalid phase {} for LianjiaDataset'.format(phase))

        with open(metadata_path, 'r') as f:
            metadata = json.load(f)

        for i_class, room_name in metadata['label_room_map'].items():
            self.add_class("Lianjia", i_class, room_name)

        base_dir = metadata['base_dir']

        # Add images
        if self.phase == 'train':
            train_list = list(sorted(os.listdir(os.path.join(base_dir, self.phase))))
            for im_idx, filename in enumerate(train_list):
                im_path = os.path.join(os.path.join(base_dir, self.phase, filename))
                for i in range(4):
                    self.add_image("Lianjia", False, image_id=8 * im_idx + 2 * i, path=im_path)
                    self.add_image("Lianjia", True, image_id=8 * im_idx + 2 * i + 1, path=im_path)

        elif self.phase == 'test':
            test_list = list(sorted(os.listdir(os.path.join(base_dir, self.phase))))
            for im_idx, filename in enumerate(test_list):
                im_path = os.path.join(os.path.join(base_dir, self.phase, filename))
                self.add_image("Lianjia", False, image_id=im_idx, path=im_path)

    def load_mask(self, image_id):
        """Load instance masks for the given image.

        Different datasets use different ways to store masks. This
        function converts the different mask format to one format
        in the form of a bitmap [height, width, instances].

        Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        info = self.image_info[image_id]
        data = np.load(info['path'], encoding='latin1').tolist()

        room_instances = data['room_instances_annot']

        if self.phase == 'train':
            rot = (image_id % 4) * 90

        # todo: load annotations
        masks, class_ids = [], []

        for room_annot in room_instances:
            # create mask
            mask = np.array(room_annot['mask'] * 255, dtype=np.uint8)
            class_id = room_annot['class']
            mask_im = Image.fromarray(mask)

            # apply augmentation
            if self.phase == 'train':
                mask_im = mask_im.rotate(rot)
                if info['flip']:
                    mask_im = mask_im.transpose(Image.FLIP_LEFT_RIGHT)

            masks.append(np.array(mask_im))
            class_ids.append(class_id)

        masks = np.stack(masks).astype('float').transpose(1, 2, 0)
        class_ids = np.array(class_ids).astype('int32')

        # For debugging purpose, visualizing the edge we drew in a single
        # scipy.misc.imsave('./mask-all-corner.jpg', mask_im_all_corner)
        # scipy.misc.imsave('./mask-all-edge.jpg', mask_im_all_edge)

        return masks, class_ids

    def image_reference(self, image_id):
        info = self.image_info[image_id]
        if info["source"] == "Lianjia":
            return info["Lianjia"]
        else:
            super(LianjiaDataset, self).image_reference(image_id)

    def load_image(self, image_id):
        info = self.image_info[image_id]
        data = np.load(info['path'], encoding='latin1').tolist()
        im = Image.fromarray(data['topview_image'])

        if self.phase == 'train':
            rot = (image_id % 4) * 90
            im = im.rotate(rot)

            if info['flip'] is True:
                im = im.transpose(Image.FLIP_LEFT_RIGHT)

        return np.array(im)

    def rotate_coords(self, image_shape, xy, angle):
        org_center = (image_shape - 1) / 2.
        rot_center = (image_shape - 1) / 2.
        org = xy - org_center
        a = np.deg2rad(angle)
        new = np.array([org[0] * np.cos(a) + org[1] * np.sin(a), -org[0] * np.sin(a) + org[1] * np.cos(a)])
        return new + rot_center


############################################################
#  Training
############################################################


if __name__ == '__main__':
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Train Mask R-CNN.')
    parser.add_argument("command",
                        metavar="<command>",
                        help="'train' or 'evaluate'")
    parser.add_argument('--model', required=False,
                        metavar="/path/to/weights.pth",
                        help="Path to weights .pth file")
    parser.add_argument('--logs', required=False,
                        default=DEFAULT_LOGS_DIR,
                        metavar="/path/to/logs/",
                        help='Logs and checkpoints directory (default=logs/)')
    parser.add_argument('--limit', required=False,
                        default=500,
                        metavar="<image count>",
                        help='Images to use for evaluation (default=500)')

    args = parser.parse_args()
    print("Command: ", args.command)
    print("Model: ", args.model)
    print("Logs: ", args.logs)

    # Configurations
    if args.command == "train":
        config = LianjiaConfig()
    else:
        class InferenceConfig(LianjiaConfig):
            # Set batch size to 1 since we'll be running inference on
            # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
            GPU_COUNT = 1
            IMAGES_PER_GPU = 1
            DETECTION_MIN_CONFIDENCE = 0


        config = InferenceConfig()
    config.display()

    # Create model
    if args.command == "train":
        model = modellib.MaskRCNN(config=config,
                                  model_dir=args.logs, input_channel=3)
    else:
        model = modellib.MaskRCNN(config=config,
                                  model_dir=args.logs, input_channel=3)
    if config.GPU_COUNT:
        model = model.cuda()

    # Select weights file to load
    if args.command == 'train' and args.model:
        if args.model.lower() == "coco":
            model_path = COCO_MODEL_PATH
        elif args.model.lower() == "last":
            # Find last trained weights
            model_path = model.find_last()[1]
        elif args.model.lower() == "imagenet":
            # Start from ImageNet trained weights
            model_path = config.IMAGENET_MODEL_PATH
        else:
            model_path = args.model
        # Load weights
        print("Loading weights for training from {}".format(model_path))
        model.load_pretrained_weights(model_path, extra_channels=0)
    else:
        model_path = ""

    # Train or evaluate
    if args.command == "train":
        # Training dataset. Use the training set and 35K from the
        # validation set, as as in the Mask RCNN paper.
        dataset_train = LianjiaDataset()
        dataset_train.load_samples("train",
                                   metadata_path=os.path.join(DATASET_BASE_DIR, '/data/lianjia_500/processed/room_metadata.json'))
        dataset_train.prepare()

        # # Validation dataset
        # dataset_val = BuildingsDataset()
        # dataset_val.load_buildings(args.dataset, "minival")
        # dataset_val.prepare()

        # *** This training schedule is an example. Update to your needs ***

        # Training - Stage 1
        # print("Training network heads")
        # model.train_model(dataset_train, dataset_train,
        #                   learning_rate=config.LEARNING_RATE,
        #                   epochs=20,
        #                   layers='heads',
        #                   config=config)

        # # Training - Stage 2
        # # Finetune layers from ResNet stage 4 and up
        # print("Fine tune Resnet stage 4 and up")
        # model.train_model(dataset_train, dataset_train,
        #                   learning_rate=config.LEARNING_RATE,
        #                   epochs=40,
        #                   layers='4+',
        #                   config=config)

        # Training - Stage 3
        # Fine tune all layers
        print("Fine tune all layers")
        model.train_model(dataset_train, dataset_train,
                          learning_rate=config.LEARNING_RATE / 2,
                          epochs=200,
                          layers='all',
                          config=config)
    else:
        print("'{}' is not recognized. "
              "Use 'train'".format(args.command))
