import _init_paths
import os
import numpy as np
import cv2
from scipy.misc import imsave, imread
from utils.floorplan_utils.cores import solve_connections
from utils.floorplan_utils.merge import merge_room_graphs
from utils.floorplan_utils.visualize import draw_final_floorplan
import copy
import pdb

SAVE_DIR = './results_floorplan/final'
ROUND_1 = True


def run_roomwise_coorindate_descent(source_dir):
    for sample_idx in range(len(os.listdir(source_dir))):
        filename = '{}_rooms_info.npy'.format(sample_idx)
        file_path = os.path.join(source_dir, filename)

        with open(file_path, 'rb') as f:
            data = np.load(f)
        data = data.tolist()

        density_img = data['density_img']
        room_labels_map = data['room_labels']
        direction_hist = data['direction_hist']

        if ROUND_1:
            graph, recon_info = solve_connections(data['rooms_info'], sample_idx, density_img, direction_hist,
                                                  room_labels_map)
        else:
            round_1_dir = './results_floorplan/final/results_round_1'
            round_1_file = os.path.join(round_1_dir, '{}_recon.npy'.format(sample_idx))
            prev_recon = np.load(round_1_file).tolist()
            prev_dp_rooms = prev_recon['dp_room_edges']

            graph, recon_info = solve_connections(data['rooms_info'], sample_idx, density_img, direction_hist,
                                                  room_labels_map, round_1=False, prev_rooms=prev_dp_rooms)

        result_img = np.zeros([256, 256, 3])
        result_img += np.stack([density_img] * 3, axis=-1) * 255

        for corner, connections in graph.items():
            for to_corner in connections:
                cv2.line(result_img, corner, to_corner, (0, 255, 255), 1)

        for corner in graph.keys():
            if len(graph[corner]) != 0:
                cv2.circle(result_img, corner, 2, (255, 0, 0), 2)
                cv2.circle(result_img, corner, 1, (255, 255, 255), 1)
        viz_dir = os.path.join(SAVE_DIR, 'visualization')
        data_dir = os.path.join(SAVE_DIR, 'results')
        # pdb.set_trace()
        if not os.path.exists(viz_dir):
            os.makedirs(viz_dir)
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
        save_path_viz = os.path.join(viz_dir, '{}_rough_floorplan.png'.format(sample_idx))
        save_path_recon = os.path.join(data_dir, '{}_recon.npy'.format(sample_idx))
        imsave(save_path_viz, result_img)
        np.save(save_path_recon, recon_info)
        print('finish processing NO.{} sample'.format(sample_idx))


def merge(save_dir, rooms_info_dir):
    data_dir = '/local-scratch/cjc/Lianjia-inverse-cad/FloorPlotter/results_floorplan/final/results_round_2'
    save_final_dir = '/local-scratch/cjc/Lianjia-inverse-cad/FloorPlotter/results_floorplan/final/output_round_2'
    for sample_idx in range(len(os.listdir(data_dir))):
        if sample_idx != 38:
            continue
        filename = '{}_recon.npy'.format(sample_idx)
        file_path = os.path.join(data_dir, filename)
        recon_info = np.load(file_path).tolist()
        dp_room_edges = recon_info['dp_room_edges']
        density_img = recon_info['density_img']
        room_class_ids = recon_info['room_class_ids']
        room_viz_colors = recon_info['room_viz_colors']
        room_labels_map = recon_info['room_labels_map']

        rooms_info_path = os.path.join(rooms_info_dir, '{}_rooms_info.npy'.format(sample_idx))
        rooms_info_data = np.load(rooms_info_path).tolist()
        rooms_info = rooms_info_data['rooms_info']

        if 'failed_rooms' in recon_info and len(recon_info['failed_rooms']) != 0:
            room_class_ids = [room_class_ids[i] for i in range(len(room_class_ids)) if
                              i not in recon_info['failed_rooms']]
            room_viz_colors = [room_viz_colors[i] for i in range(len(room_viz_colors)) if
                               i not in recon_info['failed_rooms']]
            rooms_info = [rooms_info[i] for i in range(len(room_viz_colors)) if i not in recon_info['failed_rooms']]

        raw_room_edges = copy.deepcopy(dp_room_edges)
        global_graph, all_room_edges, all_room_masks, all_room_paths, removed_indices = merge_room_graphs(raw_room_edges)

        room_class_ids = [room_class_ids[i] for i in range(len(all_room_edges) + len(removed_indices)) if
                          i not in removed_indices]
        room_viz_colors = [room_viz_colors[i] for i in range(len(all_room_edges) + len(removed_indices)) if
                           i not in removed_indices]
        rooms_info = [rooms_info[i] for i in range(len(all_room_edges) + len(removed_indices)) if
                      i not in removed_indices]

        # This part is modified for drawing figures for the paper, will be changed back later. 
        hsv_path = '../mask-rcnn/hsv_viz/{}_hsv.png'.format(sample_idx)
        hsv_img = imread(hsv_path)
        for to_idx in range(7):
            floorplan_img = draw_final_floorplan(1000, hsv_img, all_room_edges, global_graph, room_class_ids,
                                                 room_labels_map,
                                                 # floorplan_img = draw_final_floorplan(1000, hsv_img, raw_room_edges, global_graph, room_class_ids, room_labels_map,
                                                 room_viz_colors, flip_y=True, to_idx=to_idx, rooms_info=rooms_info)
            imsave('supple/step_{}.png'.format(to_idx), floorplan_img)
        return
        result_img = np.zeros([256, 256, 3])
        result_img += np.stack([density_img] * 3, axis=-1) * 255

        for corner, connections in global_graph.items():
            for to_corner in connections:
                cv2.line(result_img, corner, to_corner, (0, 255, 255), 1)

        for corner in global_graph.keys():
            if len(global_graph[corner]) != 0:
                cv2.circle(result_img, corner, 2, (255, 0, 0), 2)
                cv2.circle(result_img, corner, 1, (255, 255, 255), 1)
        viz_dir = os.path.join(save_dir, 'visualization')

        if not os.path.exists(viz_dir):
            os.makedirs(viz_dir)

        save_path_viz = os.path.join(viz_dir, '{}_rough_floorplan_round_2.png'.format(sample_idx))
        save_path_floorplan = os.path.join(viz_dir, '{}_floorplan_round_2.png'.format(sample_idx))
        # save_path_output = os.path.join(save_final_dir, '{}_output'.format(sample_idx))
        output = {
            'global_graph': global_graph,
            'all_room_edges': all_room_edges,
            'all_room_masks': all_room_masks,
            'all_room_paths': all_room_paths,
            'floorplan_image': floorplan_img,
        }
        # np.save(save_path_output, output)
        imsave(save_path_viz, result_img)
        imsave(save_path_floorplan, floorplan_img)
        cv2.imwrite(save_path_floorplan, floorplan_img)
        print('finish processing NO.{} sample'.format(sample_idx))


if __name__ == '__main__':
    DIR = '/local-scratch/cjc/Lianjia-inverse-cad/FloorPlotter/results_associate/heuristic_based_association/processed_preds'
    run_roomwise_coorindate_descent(source_dir=DIR)

    # merge(save_dir='./results_floorplan/final', rooms_info_dir=DIR)
