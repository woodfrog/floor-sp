import _init_paths
import numpy as np
import cv2
import copy
from scipy import ndimage
from collections import defaultdict
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage.morphology import binary_erosion, binary_dilation
from utils.data_utils import compute_bin_idx, get_edge_pixels, get_room_bbox, draw_room_seg_from_edges
from utils.floorplan_utils.graph_algorithms import reselect_path, dfs_all_paths, dijkstra, check_on_edge_approx
from utils.floorplan_utils.floorplan_misc import cost_map_to_mat, mat_to_cost_map, get_intersection, len_edge, \
    unit_v_edge
from utils.floorplan_utils.merge import merge_room_graphs, _refine_room_path
from utils.floorplan_utils.cost import build_room_pixel_graph
from utils.floorplan_utils.source_sink import room_init_setup
import time
import pdb

# todo: clean up unnecessary parts later...
# todo: rename functions in this file in more proper way

"""
   Solve vectorized floorplan structures
   The whole floorplan is divided into rooms
"""


def solve_connections(rooms_info, idx, density_img, direction_hist, room_labels_map, round_1=True, prev_rooms=None):
    rooms_info = _refine_rooms_info(rooms_info)
    dominant_direction_rads = get_global_dominant_direction(direction_hist)

    inter_region_cost = np.zeros(density_img.shape)

    inter_edge_cost = np.zeros(density_img.shape)
    inter_corner_cost = np.zeros(density_img.shape)

    if not round_1:
        assert prev_rooms is not None
        assert len(rooms_info) == len(prev_rooms)

    room_recons = list()
    all_room_edges = list()
    all_raw_room_edges = list()
    all_room_recon_edge = list()
    all_room_directions = list()
    room_intra_energies = list()
    failed_rooms = list()
    for room_idx, room_info in enumerate(rooms_info):
        # if it's not the first round, we also consider consistency terms from previous round's results
        if not round_1:
            prev_other_rooms = prev_rooms[room_idx + 1:]
            prev_inter_edge = np.zeros(density_img.shape)
            prev_inter_corner = np.zeros(density_img.shape)
            for prev_room_edges in prev_other_rooms:
                prev_room_mask, prev_edge_mask, prev_corner_mask = draw_room_seg_from_edges(prev_room_edges,
                                                                                            im_size=density_img.shape[
                                                                                                0])
                prev_inter_edge += prev_edge_mask
                prev_inter_corner += prev_corner_mask
            combined_inter_edge = inter_edge_cost + prev_inter_edge
            combined_inter_edge[np.where(combined_inter_edge >= 1)] = 1
            combined_inter_corner = inter_corner_cost + prev_inter_corner
            combined_inter_corner[np.where(combined_inter_corner >= 1)] = 1

        if round_1:
            inter_edge_term = inter_edge_cost
            inter_corner_term = inter_corner_cost
        else:
            inter_edge_term = combined_inter_edge
            inter_corner_term = combined_inter_corner

        room_recon, room_edges, room_path, room_directions, failed, room_intra_energy, raw_room_edges = _get_room_connections_pixel_grpah(
            room_info,
            density_img,
            inter_region_cost,
            inter_edge_term,
            inter_corner_term,
            dominant_direction_rads,
            all_room_recon_edge,
            all_room_directions,
            room_idx, idx)
        if not failed:
            recon_room_mask, recon_edge_mask, recon_corner_mask = draw_room_seg_from_edges(room_edges,
                                                                                           im_size=density_img.shape[0])
            all_room_recon_edge.append(recon_edge_mask)
            all_room_directions.append(room_directions)
            inter_region_cost += recon_room_mask
            inter_edge_cost += recon_edge_mask
            inter_corner_cost += recon_corner_mask
            inter_edge_cost[np.where(inter_edge_cost >= 1)] = 1
            inter_corner_cost[np.where(inter_corner_cost >= 1)] = 1
            room_recons.append(room_recon)
            all_room_edges.append(room_edges)
            room_intra_energies.append(room_intra_energy)
            all_raw_room_edges.append(raw_room_edges)  # for computing the exact energy
        else:
            failed_rooms.append(room_idx)

    dp_room_edges = copy.deepcopy(all_room_edges)
    global_graph, room_edges, _, _, _ = merge_room_graphs(all_room_edges)
    room_class_ids = [info['class_id'] for info in rooms_info]
    room_viz_colors = [info['viz_color'] for info in rooms_info]

    global_energy = compute_global_energy(room_intra_energies, all_raw_room_edges, lambda_model=1.0, lambda_consis=0.2, im_size=256)

    recon_info = {
        'density_img': density_img,
        'dominant_directions': dominant_direction_rads,
        'dp_room_edges': dp_room_edges,
        'global_graph': global_graph,
        'room_edges': room_edges,
        'room_class_ids': room_class_ids,
        'room_viz_colors': room_viz_colors,
        'room_labels_map': room_labels_map,
        'failed_rooms': failed_rooms,
        'global_energy': global_energy,
    }

    return global_graph, recon_info


def get_global_dominant_direction(direction_hist):
    refined_hist = _refine_direction_hist(direction_hist)

    top_8_bins = np.argsort(refined_hist)[-8:][::-1]
    sum_hist = np.sum(refined_hist)

    picked_bins = list()

    for bin in top_8_bins:
        if bin in picked_bins or refined_hist[bin] < 20:
            continue
        picked_bins.append(bin)
        if bin >= 6:
            picked_bins.append(bin - 6)
        else:
            picked_bins.append(bin + 6)

    rads = [x * np.pi / 12 for x in picked_bins]
    if len(rads) > 8:
        global_dominant_directions = [x for x in rads[:8]]
    else:
        global_dominant_directions = [x for x in rads]

    return global_dominant_directions


"""
    Merge room graphs guided by global corner registration, merge every set of colinear edges
"""

"""
    End Room Graph Merging
"""


def _refine_rooms_info(rooms_info):
    """
        Adjust room masks and edgeness maps
    """
    global_mask = np.zeros(rooms_info[0]['mask'].shape, dtype=np.float32)
    mask_size = [room_info['mask'].sum() for room_info in rooms_info]
    orders = np.argsort(mask_size)

    # sort to put larger rooms first
    rooms_info = [rooms_info[x] for x in list(orders)]

    for room_info in rooms_info:
        room_mask = room_info['mask'].astype(np.float32)
        room_mask[np.where(global_mask != 0)] = 0
        expanded_room_mask = gaussian_filter(room_mask, 5)
        expanded_room_mask[np.where(expanded_room_mask > 0.1)] = 1
        expanded_room_mask[np.where(expanded_room_mask <= 0.1)] = 0
        global_mask += expanded_room_mask
        # expand and binarize edge map
        room_info['mask'] = room_mask
        room_edge_map = room_info['edge_map']
        room_edge_map[np.where(room_edge_map > 0.5)] = 1
        room_edge_map = gaussian_filter(room_edge_map, 3)
        room_edge_map[np.where(room_edge_map > 0.5)] = 1
        room_info['edge_map'] = room_edge_map

    # filter room instances that are completely overlapped with others
    rooms_info = [room_info for room_info in rooms_info if room_info['mask'].sum() > 50]

    for room_info in rooms_info:
        room_mask = room_info['mask']
        shrinked_room_mask = binary_erosion(room_mask, iterations=2).astype(room_mask.dtype)
        if shrinked_room_mask.sum() > 50:
            room_mask = shrinked_room_mask
        room_info['mask'] = room_mask

    return rooms_info


"""
Corner-to-corner based graph, together DFS-based search algorithm for finding optimal path
"""

"""
Dense Graph, together with Shortest Path algorithms for finding optimal path.
"""


def _get_room_connections_pixel_grpah(room_info, density_img, inter_region, inter_edge, inter_corner, global_directions,
                                      all_room_recon_edge, all_room_directions, room_idx, global_idx):
    corners_info = room_info['corners_info']
    mask = room_info['mask']
    contour = room_info['contour']
    room_corner_heatmap = room_info['room_corner_heatmap']
    room_edge_map = room_info['edge_map']
    room_direction_hist = room_info['room_direction_histogram']
    # build the graph, define the distance between different corners
    room_edge_map[np.where(mask == 1)] = 0
    room_edge_map[np.where(room_edge_map > 0.5)] = 1
    room_edge_map = np.clip(room_edge_map, 0, 1)

    # for debugging use
    import cv2
    from scipy.misc import imsave, imread
    debug_img = np.zeros([256, 256, 3])
    debug_img += np.stack([mask] * 3, axis=-1).astype(np.float32) * 255
    for corner_idx, corner_info in enumerate(corners_info):
        cv2.circle(debug_img, corner_info['corner'], 1, (255, 0, 0), 2)
        cv2.putText(debug_img, '{}'.format(corner_idx), corner_info['corner'], cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (0, 255, 255), 1)

    debug_img += np.stack([density_img] * 3, -1) * 255
    # debug_img += np.stack([room_edge_map] * 3, -1) * 127
    # debug_img += np.stack([inter_edge] * 3, -1) * 127
    debug_img = np.clip(debug_img, 0, 255)

    imsave('./room_results/test/{}_{}_corners.png'.format(global_idx, room_idx), debug_img)

    im_size = mask.shape[0]
    room_corners = [info['corner'] for info in corners_info]

    # get room directions from global directions
    room_directions, directions_ratios = get_room_directions(global_directions, room_direction_hist)

    if 0.0 not in room_directions or directions_ratios[room_directions.index(0.0)] < 0.25:
        room_bbox = get_room_bbox(contour, room_corners, im_size, extra_margin=5 * 2)
    else:
        room_bbox = get_room_bbox(contour, room_corners, im_size, extra_margin=5)

    refined_room_directions = copy.deepcopy(room_directions)
    for prev_room_recon_edge, prev_room_directions in zip(all_room_recon_edge, all_room_directions):
        if prev_room_recon_edge[room_bbox[1]:room_bbox[1] + room_bbox[3],
           room_bbox[0]:room_bbox[0] + room_bbox[2]].sum() > 0:
            for direction in prev_room_directions:
                if direction not in refined_room_directions:
                    refined_room_directions.append(direction)

    # collect all pixel nodes
    pixel_nodes = list()
    for y in range(room_bbox[1], room_bbox[1] + room_bbox[3] + 1):
        for x in range(room_bbox[0], room_bbox[0] + room_bbox[2] + 1):
            if (x, y) not in room_corners and (mask[y][x] == 0 or room_edge_map[y][x] == 1 or inter_edge[y][x] >= 1):
                pixel_nodes.append((x, y))

    all_nodes = room_corners + pixel_nodes  # keep indices for corners the same

    tried_sources = list()

    failed = False

    while True:
        # get a proper source and sink node for per-room estimation, this is a very important step
        try:
            source, end, starting_line, tried_sources = room_init_setup(all_nodes, corners_info,
                                                                        refined_room_directions,
                                                                        room_bbox, mask, room_edge_map, inter_region,
                                                                        inter_edge, tried_sources)
        except RuntimeError:
            failed = True
            print('Fail to reconstruct room {}'.format(room_idx))
            break

        start_time = time.time()
        room_graph, all_nodes, cost_map, starting_line, intra_map = build_room_pixel_graph(all_nodes,
                                                                                           refined_room_directions,
                                                                                           starting_line,
                                                                                           inter_region,
                                                                                           inter_edge,
                                                                                           inter_corner,
                                                                                           mask,
                                                                                           room_edge_map,
                                                                                           room_corner_heatmap,
                                                                                           room_bbox,
                                                                                           density_img)

        shortest_path, dists = dijkstra(room_graph, source, end, all_nodes)
        end_time = time.time()
        print('Finish scene {} room {}, Time: {}'.format(global_idx, room_idx, end_time - start_time))

        path = list(reversed(shortest_path))

        if len(path) == 1:
            print('Failed to find a proper reconstruction, restart with new source and sink')
            tried_sources.append(source)
            continue

        room_nodes = [all_nodes[x] for x in path]
        room_edges = [(node, room_nodes[idx + 1]) for idx, node in enumerate(room_nodes[:-1])]
        room_edges.append((room_nodes[-1], room_nodes[0]))
        try:
            recon_mask, _, _ = draw_room_seg_from_edges(room_edges, im_size)
        except AssertionError:
            failed = True
            print('Failed to reconstruct room {}'.format(room_idx))
            break

        mask_covered = recon_mask[np.where(mask == 1)].sum()
        if mask_covered / mask.sum() > 0.5:
            print('Reconstruction is successful for room {} of sample {}'.format(room_idx, global_idx))
            break
        else:
            tried_sources.append(source)
            print('Reconstruction failed to cover the mask, restart...')

    if failed is True:
        return None, None, None, None, failed, None, None

    room_path = [all_nodes[node_i] for node_i in path]
    # compute the exact energy and the raw room edges
    raw_edges = list()
    intra_energies = list()
    for node_idx, path_node in enumerate(room_path):
        next_idx = node_idx + 1 if node_idx < len(room_path) - 1 else 0
        room_edge = (path_node, room_path[next_idx])
        raw_edges.append(room_edge)
        if node_idx != len(room_path) - 1:  # skip the start-line, which is not in the graph
            edge_energy = intra_map[room_edge[0]][room_edge[1]]
            corner_energy = (1 - room_corner_heatmap[room_edge[1][1], room_edge[1][0]]) * 0.2
            intra_energies.append(edge_energy + corner_energy)

    room_path = _refine_room_path(room_path)
    room_edges = list()
    for node_idx, path_node in enumerate(room_path):
        next_idx = node_idx + 1 if node_idx < len(room_path) - 1 else 0
        room_edge = (path_node, room_path[next_idx])
        room_edges.append(room_edge)

    room_intra_energy = np.sum(intra_energies)

    cost_map_mat = cost_map_to_mat(cost_map, im_size=room_corner_heatmap.shape[0])

    room_recon = {
        'room_path': room_path,
        'corner_heatmap': room_corner_heatmap,
        'cost_map_mat': cost_map_mat,
    }

    #####
    # # For visualization and debugging purpose
    #####

    for node_i, node in enumerate(room_path):
        if node_i == 0 or node_i == len(room_path) - 1:
            cv2.circle(debug_img, node, 2, (0, 0, 255), 2)
            cv2.circle(debug_img, node, 1, (255, 255, 255), 1)
        else:
            cv2.circle(debug_img, node, 2, (255, 0, 0), 2)
            cv2.circle(debug_img, node, 1, (255, 255, 255), 1)

    for corner, to_corner in room_edges:
        cv2.line(debug_img, corner, to_corner, (0, 255, 255), 1)

    # put room bbox
    cv2.rectangle(debug_img, (room_bbox[0], room_bbox[1]), (room_bbox[0] + room_bbox[2], room_bbox[1] + room_bbox[3]),
                  (0, 255, 0))

    # put texts for path
    path_str = '-'.join([str(all_nodes.index(x)) for x in room_path])
    cv2.putText(debug_img, path_str, (20, 20), 1, 0.5, (255, 255, 255))

    imsave('./room_results/test/{}_{}_results_test.png'.format(global_idx, room_idx), debug_img)
    #####
    # # END. For visualization and debugging purpose
    #####

    return room_recon, room_edges, room_path, room_directions, failed, room_intra_energy, raw_edges


def get_room_directions(global_directions, room_direction_hist):
    # filter global directions using per room votes
    refined_room_hist = _refine_direction_hist(room_direction_hist)
    sum_room_hist = np.sum(refined_room_hist)
    room_directions = [x for x in global_directions if
                       refined_room_hist[int(np.round(x / np.pi * 12))] >= 20]
    adjusted_room_dirs = list()
    for dir in room_directions:
        if dir in adjusted_room_dirs:
            continue
        adjusted_room_dirs.append(dir)
        bin_idx = int(np.round(dir / np.pi * 12))
        if bin_idx >= 6:
            adjusted_room_dirs.append((bin_idx - 6) * np.pi / 12)
        else:
            adjusted_room_dirs.append((bin_idx + 6) * np.pi / 12)
    if len(adjusted_room_dirs) > 4:
        room_directions = adjusted_room_dirs[:4]
    else:
        room_directions = adjusted_room_dirs
    direction_votes = [refined_room_hist[int(np.round(x / np.pi * 12))] for x in room_directions]
    direction_ratios = [x / np.sum(direction_votes) for x in direction_votes]

    if 0.0 in room_directions:
        index_0 = room_directions.index(0.0)
        index_90 = room_directions.index(np.pi / 2)
        if direction_ratios[index_0] + direction_ratios[index_90] > 0.95:
            room_directions = room_directions[:2]
            direction_ratios = direction_ratios[:2]

    return room_directions, direction_ratios


def _refine_direction_hist(hist):
    num_bins = len(hist)
    refined_hist = [0] * 12  # interval is 15 degree
    for direction, count in hist.items():
        if direction >= num_bins / 2:
            new_bin = (direction - num_bins / 2) * 1.0 / 1.5
        else:
            new_bin = (direction) * 1.0 / 1.5
        if new_bin == 12:
            new_bin = 0
        if new_bin == int(new_bin):
            refined_hist[int(new_bin)] += count
        else:
            refined_hist[int(np.floor(new_bin))] += count / 2.0
            ceil = np.ceil(new_bin)
            if ceil == 12:
                ceil = 0
            refined_hist[int(ceil)] += count / 2.0
    return refined_hist


def compute_global_energy(intra_energies, dp_room_edges, lambda_model, lambda_consis, im_size):
    edge_map = np.zeros([im_size, im_size])
    for room_edges in dp_room_edges:
        for edge in room_edges:
            cv2.line(edge_map, edge[0], edge[1], 1, thickness=1)
            edge_map[edge[1][1], edge[1][0]] -= 1
            edge_map[edge[0][1], edge[0][0]] -= 1
    edge_map[np.where(edge_map >= 1)] == 1
    edge_pixels = edge_map.sum()

    corner_map = np.zeros([im_size, im_size])
    for room_edges in dp_room_edges:
        for edge in room_edges:
            corner_map[edge[1][1], edge[1][0]] = 1
    corner_map[np.where(corner_map >= 1)] == 1
    corner_pixels = corner_map.sum()

    E_consis = edge_pixels * 0.5 + corner_pixels
    E_model = np.sum([len(room_edges) for room_edges in dp_room_edges])
    global_energy = np.sum(intra_energies) + lambda_consis * E_consis + lambda_model * E_model

    return global_energy











