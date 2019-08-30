import os
import numpy as np

import main
import model.utils as utils
import model.model as modellib
import model.visualize as visualize
import torch
import json
import cv2

import pdb

# Root directory of the project
ROOT_DIR = os.getcwd()

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

class InferenceConfig(main.LianjiaConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    # GPU_COUNT = 0 for CPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1


config = InferenceConfig()
config.display()

# Create model object.
model = modellib.MaskRCNN(model_dir=MODEL_DIR, config=config, input_channel=3)
if config.GPU_COUNT:
    model = model.cuda()

# Load weights trained on MS-COCO
saved_model = './logs/lianjia_dataset20180920T2038/mask_rcnn_lianjia_dataset_0069.pth'
model.load_state_dict(torch.load(saved_model))

print('loaded weights from {}'.format(saved_model))

# Read in metadata
metadata_path = '/local-scratch/cjc/floor-sp/data/lianjia_500/processed/room_metadata.json'
with open(metadata_path, 'r') as f:
    metadata = json.load(f)
label_class_map = metadata['label_room_map']
label_class_map['5'] = 'none'

label_indices = sorted([int(s) for s in label_class_map.keys()])
class_names = ['BG', ] + [label_class_map[str(i)] for i in label_indices]

# Load a random image from the images folder
phase = 'test'
data_dir = os.path.join(metadata['base_dir'], phase)

# Set the corner data base path
CORNER_DATA_BASE = '/local-scratch/cjc/floor-sp/floor-sp/data/Lianjia_corner'
if not os.path.exists(CORNER_DATA_BASE):
    os.mkdir(CORNER_DATA_BASE)
CORNER_DATA_BASE = os.path.join(CORNER_DATA_BASE, phase)
if not os.path.exists(CORNER_DATA_BASE):
    os.mkdir(CORNER_DATA_BASE)

VIZ_DIR = './{}_viz'.format(phase)
if not os.path.exists(VIZ_DIR):
    os.makedirs(VIZ_DIR)

for file_idx, filename in enumerate(sorted(os.listdir(data_dir))):
    file_path = os.path.join(data_dir, filename)
    data = np.load(file_path, encoding='latin1').tolist()

    image = data['topview_image']
    room_annot = data['room_instances_annot']
    lines = data['line_coords']
    room_map = data['room_map']
    bg_idx = data['bg_idx']
    normal_image = data['topview_mean_normal']
    point_dict = data['point_dict']
    lines = data['lines']

    annot_image = np.zeros([image.shape[0], image.shape[1]], dtype=np.uint8)
    all_gt_masks = list()
    for i, room in enumerate(room_annot):
        mask = room['mask_large'].astype('uint8')
        all_gt_masks.append(mask)
        annot_image += mask * 200

    # Run detection
    results = model.detect([image])

    # Visualize results
    r = results[0]

    all_contours, all_masks, all_class_ids, _ = visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'],
                                                          class_names, scores=r['scores'], show_image=False,
                                                          save_path='{}/{}_maskrcnn.png'.format(VIZ_DIR, file_idx))


    cv2.imwrite('{}/{}_maskrcnn_input.png'.format(VIZ_DIR, file_idx), image)
    cv2.imwrite('{}/{}_maskrcnn_gt.png'.format(VIZ_DIR, file_idx), annot_image)

    room_mask_pred = np.zeros([image.shape[0], image.shape[0]])
    for room_i, (room_contour, room_mask) in enumerate(zip(all_contours, all_masks)):
        room_mask_pred = room_mask_pred + room_mask

    corner_data_sample = {
        'topview_image': image,
        'mean_normal': normal_image,
        'room_map': room_map,
        'room_mask_pred': room_mask_pred,
        'point_dict': point_dict,
        'lines': lines,
        'room_annot': room_annot,
    }

    filename = os.path.basename(file_path)
    save_path = os.path.join(CORNER_DATA_BASE, filename)

    np.save(save_path, corner_data_sample)
    print('finish processing {} sample No.{}'.format(phase, file_idx))
