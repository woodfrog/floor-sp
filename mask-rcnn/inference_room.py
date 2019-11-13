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


def find_matched_room_annot(room_mask_pred, room_annots):
    matched_annot = None
    for room_annot in room_annots:
        gt_mask = room_annot['mask'].astype(bool)
        pred_mask = room_mask_pred.astype(bool)
        joint_mask = np.logical_and(gt_mask, pred_mask)
        # print(joint_mask.sum() * 1.0 / gt_mask.sum())
        if joint_mask.sum() * 1.0 / gt_mask.sum() >= 0.5:
            matched_annot = room_annot
    return matched_annot


def get_all_corners(room_annots):
    all_corners = list()
    for room_annot in room_annots:
        all_corners += room_annot['room_corners']
    all_corners = list(set(all_corners))
    return all_corners


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

# Load Mask-RCNN weights trained on Lianjia dataset
saved_model = './logs/mask_rcnn_lianjia_dataset_0069.pth'
model.load_state_dict(torch.load(saved_model))

print('loaded weights from {}'.format(saved_model))

# Read in metdata of the dataset
metadata_path = '../data/public_100/processed/room_metadata.json'
with open(metadata_path, 'r') as f:
    metadata = json.load(f)
label_class_map = metadata['label_room_map']
label_class_map['5'] = 'none'

label_indices = sorted([int(s) for s in label_class_map.keys()])
class_names = ['BG', ] + [label_class_map[str(i)] for i in label_indices]

# Load a random image from the images folder
phase = 'test'
data_dir = os.path.join(metadata['base_dir'], phase)

# Set the base path for generating room dataset based on Mask-RCNN output
ROOM_DATA_BASE = '../floor-sp/data/dataset_room'
if not os.path.exists(ROOM_DATA_BASE):
    os.makedirs(ROOM_DATA_BASE)
ROOM_DATA_BASE = os.path.join(ROOM_DATA_BASE, phase)
if not os.path.exists(ROOM_DATA_BASE):
    os.makedirs(ROOM_DATA_BASE)

# dir for visualization results
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

    # generate hsv file for better visualizing inputs
    input_hsv = np.zeros(image.shape)
    input_hsv[:, :, 2] = np.clip(image[:, :, 0] * 6, 0, 255)
    input_hsv[:, :, 1] = normal_image[:, :, 2] * 127 + 127
    input_hsv[:, :, 0] = (np.arctan2(normal_image[:, :, 1], normal_image[:, :, 0]) / np.pi * 180 + 180) / 2
    input_hsv = input_hsv.astype(np.uint8)
    input_viz = cv2.cvtColor(input_hsv, cv2.COLOR_HSV2BGR)
    cv2.imwrite('hsv_viz/{}_hsv.png'.format(file_idx), input_viz)

    annot_image = np.zeros([image.shape[0], image.shape[1]], dtype=np.uint8)
    all_gt_masks = list()
    for i, room in enumerate(room_annot):
        mask = room['mask_large'].astype('uint8')
        all_gt_masks.append(mask)
        annot_image += mask * 200
    results = model.detect([image])

    # Visualize results
    r = results[0]
    
    # Control the color of generated masks by viz_colors. This is for generating consistent visualization 
    # between masks and final reconstructed rooms. 
    # read_path = '../floor-sp/results_associate/mode_room_corner_lr_0.0001_batch_size_16/processed_preds'
    # read_path = os.path.join(read_path, '{}_rooms_info.npy'.format(file_idx))
    # rooms_info = np.load(read_path).tolist()['rooms_info']
    # viz_colors = [info['viz_color'] for info in rooms_info]

    all_contours, all_masks, all_class_ids, room_viz_colors = visualize.display_instances(image, r['rois'], r['masks'],
                                                                                          r['class_ids'],
                                                                                          class_names,
                                                                                          scores=r['scores'],
                                                                                          show_image=False,
                                                                                          save_path='{}/{}_maskrcnn.png'.format(
                                                                                              VIZ_DIR, file_idx), flip=False, viz_colors=None)

    cv2.imwrite('{}/{}_maskrcnn_input.png'.format(VIZ_DIR, file_idx), np.flipud(image))
    cv2.imwrite('{}/{}_maskrcnn_gt.png'.format(VIZ_DIR, file_idx), annot_image)

    room_data = list()
    for room_i, (room_contour, room_mask, class_id, viz_color) in enumerate(zip(all_contours, all_masks, all_class_ids, room_viz_colors)):
        matched_room_annot = find_matched_room_annot(room_mask, room_annot)

        if matched_room_annot is None:
            print('The No.{} predicted contour does not have matched g.t. room instance'.format(room_i))
            if phase == 'train':
                continue
            else:
                matched_room_annot = {
                    'mask': None,
                    'room_corners': None,
                    'class_id': None
                }

        contour_image = np.zeros((image.shape[0], image.shape[1]))
        for vert in room_contour:
            cv2.circle(contour_image, (int(vert[1]), int(vert[0])), 2, 255)

        room_data_sample = {
            'topview_image': image,
            'mean_normal': normal_image,
            'pred_room_mask': room_mask,  # pred mask
            'gt_room_mask': matched_room_annot['mask'],
            'contour_image': contour_image,  # 256 x 256
            'contour': room_contour,
            'room_corners': matched_room_annot['room_corners'],
            'class_id': class_id,
            'viz_color': viz_color,
        }

        room_data.append(room_data_sample)
    all_corners = get_all_corners(room_annot)

    filename = os.path.basename(file_path)
    save_path = os.path.join(ROOM_DATA_BASE, filename)

    data_sample = {
        'class_names': class_names,
        'room_data': room_data,
        'all_corners': all_corners,
        'point_dict': point_dict,
        'lines': lines,
    }

    # # Debugging. For checking the difference between all_corners(merged from room corners) and all annotated points
    # points = list()
    # for key, item in point_dict.items():
    #     points.append((item['img_x'], item['img_y']))
    # import cv2
    # from scipy.misc import imsave
    # img = np.zeros([256, 256, 3])
    # for point in all_corners:
    #     cv2.circle(img, point, 2, (255, 0, 0), 2)
    # imsave('all_corners.png', img)
    # img = np.zeros([256, 256, 3])
    # for point in points:
    #     cv2.circle(img, point, 2, (255, 0, 0), 2)
    # imsave('points.png', img)

    np.save(save_path, data_sample)
    print('finish processing {} sample No.{}'.format(phase, file_idx))
