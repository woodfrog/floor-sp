import _init_paths
import os
import torch
from scipy.misc import imsave
from scipy.ndimage.morphology import binary_dilation
import numpy as np

from utils.config import Struct, load_config, compose_config_str
from utils.data_utils import get_single_corner_map, get_edge_map, get_direction_hist, get_room_heatmap, get_room_bbox
from utils.floorplan_utils.floorplan_misc import visualize_rooms_info


def room_corner_association(configs):
    viz_dir = os.path.join(configs.exp_dir, 'preds_visualization')
    save_dir = os.path.join(configs.exp_dir, 'processed_preds')
    if not os.path.exists(viz_dir):
        os.makedirs(viz_dir)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    data_dir = '/local-scratch/cjc/Lianjia-inverse-cad/FloorPlotter/data/Lianjia_room/test'
    corner_preds_dir = '/local-scratch/cjc/Lianjia-inverse-cad/FloorPlotter/results_corner/lr_0.0001_batch_size_4_augmentation_r_corner_edge/test_preds'

    assert len(os.listdir(data_dir)) == len(os.listdir(corner_preds_dir))

    for idx, filename in enumerate(sorted(os.listdir(data_dir))):
        file_path = os.path.join(data_dir, filename)
        with open(file_path, 'rb') as f:
            data = np.load(f).tolist()

        corner_preds_path = os.path.join(corner_preds_dir, '{}_corner_preds.npy'.format(idx))
        with open(corner_preds_path, 'rb') as f:
            corner_preds = np.load(corner_preds_path).tolist()

        room_data = data['room_data']
        room_labels = data['class_names']
        vectorized_corner_preds = corner_preds['vectorized_preds']
        corner_preds_heatmap = corner_preds['corner_heatmap']
        edge_preds = corner_preds['edge_preds']
        edge_heatmap = edge_preds[0]
        global_direction_hist = corner_preds['direction_hist']

        vectorized_rooms_info = list()
        # iterate through rooms of the scene
        for room_idx, room_info in enumerate(room_data):
            room_mask = room_info['pred_room_mask'].astype(np.float64)
            room_class_id = room_info['class_id']
            room_viz_color = room_info['viz_color']

            # Get a larger mask for selecting more corner candidates around every room proposal.
            large_mask = binary_dilation(room_info['pred_room_mask'].astype(np.float32),
                                         iterations=20).astype(np.float32)
            room_corners = list()
            for corner_pred in vectorized_corner_preds:
                corner = corner_pred['corner']
                if large_mask[corner[1], corner[0]] == 1:
                    corner_conf = corner_preds_heatmap[corner[1], corner[0]]
                    room_corner = {
                        'corner': corner,
                        'edge_dirs': corner_pred['edge_dirs'],
                        'corner_conf': corner_conf,
                        'binning': corner_pred['binning']
                    }
                    room_corners.append(room_corner)

            mask_size = room_info['pred_room_mask'].sum()
            if mask_size < 2000:
                expand_iter = 10
            else:
                expand_iter = 20
            expanded_mask = binary_dilation(room_info['pred_room_mask'].astype(np.float32),
                                            iterations=expand_iter).astype(np.float32)

            room_corner_heatmap = get_room_heatmap(room_corners, corner_preds_heatmap, mode='corner')
            # filter the edge map using expanded mask regions, rather than using bounding box
            room_edge_map = edge_heatmap * expanded_mask
            room_edge_preds = edge_preds * np.expand_dims(expanded_mask, axis=0)
            room_direction_hist = get_direction_hist(room_edge_preds)

            room_details = {
                'corners_info': room_corners,
                'contour': room_info['contour'],
                'mask': room_info['pred_room_mask'],
                'edge_map': room_edge_map,
                'room_corner_heatmap': room_corner_heatmap,
                'room_direction_histogram': room_direction_hist,
                'class_id': room_class_id,
                'viz_color': room_viz_color,
            }
            vectorized_rooms_info.append(room_details)

        # refine rooms info, computing the source/end nodes according to corner conf
        room_imgs = visualize_rooms_info(vectorized_rooms_info)

        sample_data = {
            'rooms_info': vectorized_rooms_info,
            'all_corners_info': vectorized_corner_preds,
            'density_img': room_data[0]['topview_image'][:, :, 0].astype(
                np.float64) / 255,
            'mean_normal': room_data[0]['mean_normal'],
            'direction_hist': global_direction_hist,
            'room_labels': room_labels,
        }

        save_path = os.path.join(save_dir, '{}_rooms_info.npy'.format(idx))
        with open(save_path, 'wb') as f:
            np.save(f, sample_data)

        # for debudding use
        viz_dir_rooms = os.path.join(viz_dir, 'rooms_{}'.format(idx))
        if not os.path.exists(viz_dir_rooms):
            os.makedirs(viz_dir_rooms)
        for room_idx, room_img in enumerate(room_imgs):
            room_save_path = os.path.join(viz_dir_rooms, '{}_room_corners.png'.format(room_idx))
            imsave(room_save_path, room_img)

        print('Finish processing sample NO.{}'.format(idx))


if __name__ == '__main__':
    config_dict = load_config(file_path='./configs/config_associate_module.yaml')
    configs = Struct(**config_dict)
    extra_option = configs.extra if hasattr(configs, 'extra') else None
    exp_dir = os.path.join(configs.exp_base_dir, 'heuristic_based_association')
    if not os.path.exists(exp_dir):
        os.makedirs(exp_dir)
    configs.exp_dir = exp_dir
    if configs.seed:
        torch.manual_seed(configs.seed)
        if configs.use_cuda:
            torch.cuda.manual_seed(configs.seed)
        np.random.seed(configs.seed)
        print('set random seed to {}'.format(configs.seed))

    print('Infer room-corner association using heuristics')
    room_corner_association(configs)
