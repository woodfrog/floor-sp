import numpy as np
from collections import defaultdict
from utils.data_utils import compute_bin_idx, get_edge_pixels
from utils.floorplan_utils.graph_algorithms import reselect_path, dfs_all_paths, dijkstra
from scipy.sparse import coo_matrix
from scipy import ndimage
import cv2
import pdb

def _build_room_graph(corners_info, mask, contour, source, end):
    """
     Construct the graph for a room accoring to corners, source index, end index, room masks,
     and room contour
    """
    # should return a graph(dict), keys are corners,
    # values are a list with distance to all other corners
    graph = dict()
    for corner_idx, corner_info in enumerate(corners_info):
        graph[corner_idx] = dict()
        for other_idx, other_info in enumerate(corners_info):
            if corner_idx == other_idx:
                continue
            if other_idx in graph and corner_idx in graph[other_idx]:  # symmetric property
                graph[corner_idx][other_idx] = graph[other_idx][corner_idx]
                continue
            try:
                dist_info = _compute_corner_distance(corner_info, other_info, corners_info[source], corners_info[end],
                                                     mask,
                                                     contour)
            except:
                pdb.set_trace()
            graph[corner_idx][other_idx] = dist_info

    graph[source][end] = np.inf

    return graph


def _get_room_connections_corner_grpah(room_info, density_img, room_idx, global_idx):
    corners_info = room_info['corners_info']
    mask = room_info['mask']
    contour = room_info['contour']
    source_idx = room_info['max_corner_idx']
    end_idx = room_info['adj_corner_idx']
    graph_weights = room_info['graph_weights']
    # build the graph, define the distance between different corners

    # for debugging use
    import cv2
    from scipy.misc import imsave
    debug_img = np.zeros([256, 256, 3])
    debug_img += np.stack([mask] * 3, axis=-1).astype(np.float32) * 255
    result_img = np.copy(debug_img)
    for corner_idx, corner_info in enumerate(corners_info):
        cv2.circle(debug_img, corner_info['corner'], 2, (255, 0, 0), 2)
        cv2.circle(result_img, corner_info['corner'], 2, (255, 0, 0), 2)
        cv2.putText(debug_img, '{}'.format(corner_idx), corner_info['corner'], cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (0, 255, 255), 1)

        for bin_idx, edge_conf in enumerate(corner_info['binning'].tolist()):
            if edge_conf > 0.1:
                unit_vec = (np.cos(bin_idx * 10 / 180 * np.pi), np.sin(bin_idx * 10 / 180 * np.pi))
                end_point = (
                    int(corner_info['corner'][0] + unit_vec[0] * 10),
                    int(corner_info['corner'][1] - unit_vec[1] * 10))
                cv2.line(debug_img, corner_info['corner'], end_point, (255, 255, 255), 1)

    imsave('./debug/{}_{}_corners.png'.format(global_idx, room_idx), debug_img)
    # imsave('./debug/{}_{}_density.png'.format(global_idx, room_idx), density_img)

    if end_idx is None:
        room_connections = defaultdict(list)
    else:
        # #build corner detection based graphs
        heuristic_room_graph = _build_room_graph(corners_info, mask, contour, source_idx, end_idx)
        final_graph = _refine_predicted_graph_weights(graph_weights, heuristic_room_graph, corners_info, source_idx,
                                                      end_idx)
        room_corners = [info['corner'] for info in corners_info]

        # solve the shortest path given source and end using Dijkstra's algorithm
        # shortest_path, dists = _dijkstra(final_graph, source_idx, end_idx)
        # path = list(reversed(shortest_path))

        # if global_idx == 10 and room_idx == 2:

        # if len(shortest_path) < 0.8 * len(final_graph):  # the path is too short
        trial_num = 0
        reselected_path = None
        while reselected_path is None:
            all_paths, all_lens = dfs_all_paths(final_graph, room_corners, source_idx, end_idx, trial_num)
            reselected_path = reselect_path(all_paths, all_lens, len(final_graph), trial_num)
            print('search trial No.{}'.format(trial_num))
            trial_num += 1
            if trial_num >= 3:
                pdb.set_trace()
        path = reselected_path
        room_connections = defaultdict(list)
        # construct room connections according to shortest path
        for idx, corner_idx in enumerate(path):
            next_idx = idx + 1 if idx < len(path) - 1 else 0
            corner = corners_info[corner_idx]['corner']
            to_corner = corners_info[path[next_idx]]['corner']
            room_connections[corner].append(to_corner)
            room_connections[to_corner].append(corner)

    for corner, to_corners in room_connections.items():
        for to_corner in to_corners:
            cv2.line(result_img, corner, to_corner, (0, 255, 255), 2)

    path_str = '-'.join([str(node) for node in path])
    cv2.putText(result_img, path_str, (20, 20), 1, 1, (255, 255, 255))
    imsave('./debug/{}_{}_results.png'.format(global_idx, room_idx), result_img)

    return room_connections


# def _compute_pixel_pixel_cost(p1, p2, mask, density_img):
#     all_pixels = get_edge_pixels(p1, p2)
#     cost = 0.1
#     for pixel in all_pixels:
#         if mask[pixel[1]][pixel[0]] == 1:
#             cost += 1
#     return cost


"""
#   Using a bunch of heuristics to define distance between two corners on the 2d space
    *** NOT used for now, we use dense DP for reconstructing per-room structure ***
"""


def _compute_corner_distance(corner_info_1, corner_info_2, corner_info_source, corner_info_end, mask, contour):
    corner_1 = corner_info_1['corner']
    corner_2 = corner_info_2['corner']
    corner_source = corner_info_source['corner']
    corner_end = corner_info_end['corner']
    all_pixels = get_edge_pixels(corner_1, corner_2)

    length = all_pixels.shape[0]
    # lambda_mask = 10.0
    # lambda_binning = 2.0
    crossing_thresh = 5.0

    # blurred_mask = gaussian_filter(mask.astype(np.float64), sigma=2)

    # todo: define the distance between two corners, using mask and density(wall evidence)
    num_mask = 0
    cross_source = False
    cross_end = False
    for pixel in all_pixels:
        # check whether the connection between two corners crossing the source, if yes, cut this connection
        if corner_1 != corner_source and corner_2 != corner_source:
            dist_to_source = np.sqrt((corner_source[0] - pixel[0]) ** 2 + (corner_source[1] - pixel[1]) ** 2)
            if dist_to_source <= crossing_thresh:
                cross_source = True
                break
        if corner_1 != corner_end and corner_2 != corner_end:
            dist_to_end = np.sqrt((corner_end[0] - pixel[0]) ** 2 + (corner_end[1] - pixel[1]) ** 2)
            if dist_to_end <= crossing_thresh:
                cross_end = True
                break

        mask_value = mask[pixel[1], pixel[0]]
        # on_contour = _check_on_contour(pixel, contour, thresh=1)

        if mask_value != 0:
            # if not on_contour:
            num_mask += 1

    # mask_ratio = num_mask * 1.0 / length
    # if cross_source or cross_end:
    #     dist = np.inf
    # else:
    #     binning_score = _compute_binning_score(corner_info_1, corner_info_2)
    #     dist = (num_mask / length) * lambda_mask + (1 - binning_score) * lambda_binning

    return (cross_source, cross_end, num_mask, length)


def _check_on_contour(corner, contour, thresh):
    on_contour = False
    for vert in contour:
        y, x = vert
        dist = np.sqrt((corner[0] - x) ** 2 + (corner[1] - y) ** 2)
        if dist <= thresh:
            on_contour = True
            break
    return on_contour


def _compute_binning_score(corner_info_1, corner_info_2):
    corner_1 = corner_info_1['corner']
    corner_2 = corner_info_2['corner']

    num_bins = corner_info_1['binning'].shape[0]
    vec_12 = (corner_2[0] - corner_1[0], -(corner_2[1] - corner_1[1]))
    bin_12 = compute_bin_idx(vec_12, num_bins)
    vec_21 = (corner_1[0] - corner_2[0], -(corner_1[1] - corner_2[1]))
    bin_21 = compute_bin_idx(vec_21, num_bins)

    score_12 = _match_binning(corner_info_1['edge_dirs'], bin_12, num_bins)
    score_21 = _match_binning(corner_info_2['edge_dirs'], bin_21, num_bins)

    score = (score_12 + score_21) / 2

    return score


def _match_binning(pred_edge_dirs, bin, num_bins):
    score = 0.0
    for edge_dir in pred_edge_dirs:
        prev_bin = bin - 1 if bin > 0 else num_bins - 1
        next_bin = bin + 1 if bin < num_bins - 1 else 0
        if edge_dir == bin or edge_dir == prev_bin or edge_dir == next_bin:
            score = 1.0
    return score


###
#   END - heuristics for defining distance between corners
###


"""
    Functions for converting floorplan dense cost map to sparse matrix, and backwards.
"""

def cost_map_to_mat(cost_map, im_size):
    rows = list()
    cols = list()
    data = list()
    for node, connections in cost_map.items():
        node_value = node[0] * im_size + node[1]
        for to_node, dist in connections.items():
            to_node_value = to_node[0] * im_size + to_node[1]
            rows.append(node_value)
            cols.append(to_node_value)
            data.append(dist)

    sparse_cost_mat = coo_matrix((data, (rows, cols)), shape=(im_size**2, im_size**2))

    return sparse_cost_mat


def mat_to_cost_map(mat, im_size):
    cost_map = defaultdict(dict)
    for row, col, data in zip(mat.row, mat.col, mat.data):
        row_x = row // im_size
        row_y = row % im_size
        col_x = col // im_size
        col_y = col % im_size
        node = (row_x, row_y)
        to_node = (col_x, col_y)
        cost_map[node][to_node] = data
        cost_map[to_node][node] = data
    return cost_map

"""
    END Functions for converting floorplan dense cost map to sparse matrix, and backwards.
"""


def len_edge(e):
    return np.sqrt((e[1][0] - e[0][0]) ** 2 + (e[1][1] - e[0][1]) ** 2)


def unit_v_edge(e):
    len_e = len_edge(e)
    assert len_e != 0
    unit_v = ((e[1][0] - e[0][0]) / len_e, (e[1][1] - e[0][1]) / len_e)
    return unit_v


def get_intersection(p0, p1, p2, p3):
    """
        reference: StackOverflow https://stackoverflow.com/questions/563198/how-do-you-detect-where-two-line-segments-intersect#565282
    """
    s1_x = p1[0] - p0[0]
    s1_y = p1[1] - p0[1]
    s2_x = p3[0] - p2[0]
    s2_y = p3[1] - p2[1]

    s = (-s1_y * (p0[0] - p2[0]) + s1_x * (p0[1] - p2[1])) / (-s2_x * s1_y + s1_x * s2_y);
    t = (s2_x * (p0[1] - p2[1]) - s2_y * (p0[0] - p2[0])) / (-s2_x * s1_y + s1_x * s2_y);

    if 1 >= s >= 0 and 1 >= t >= 0:
        i_x = p0[0] + (t * s1_x)
        i_y = p0[1] + (t * s1_y)
        return (i_x, i_y)
    else:
        return None


def visualize_rooms_info(rooms_info):
    """
    Refine information of rooms for every scene, make preparation for graph-based algorithms.
    Defining starting and ending corner for every room.
    :param rooms_info:
    :return:
    """
    # debugging use: visualize the corners for every single room
    room_imgs = list()
    # iterate through rooms
    for room_info in rooms_info:
        room_img = np.zeros([256, 256, 3])
        mask = room_info['mask']
        room_img = room_img + np.stack([mask] * 3, -1) * 255
        corner_confs = list()
        # processing every room
        for corner_info in room_info['corners_info']:
            corner_confs.append(corner_info['corner_conf'])
            cv2.circle(room_img, corner_info['corner'], 2, (255, 0, 0), 2)
            for edge_dir in corner_info['edge_dirs']:
                binning = corner_info['binning']
                if binning[edge_dir] > .5:
                    unit_vec = (np.cos(edge_dir * 10 / 180 * np.pi), np.sin(edge_dir * 10 / 180 * np.pi))
                    end_point = (
                        int(corner_info['corner'][0] + unit_vec[0] * 10),
                        int(corner_info['corner'][1] - unit_vec[1] * 10))
                    cv2.line(room_img, corner_info['corner'], end_point, (255, 255, 255), 1)

        room_imgs.append(room_img)

    return room_imgs


def vectorization(preds):
    corner_conf = preds[0]
    corner_preds = nms_2d(corner_conf)  # a list of corner predictions

    vectorized_preds = list()
    for corner_pred in corner_preds:
        edge_dirs = process_edge_binning(preds[1:, corner_pred[1], corner_pred[0]])
        vectorized_preds.append({
            'corner': corner_pred,
            'edge_dirs': edge_dirs
        })

    return vectorized_preds


def get_corner_dir_map(preds, im_size):
    vectorized_preds = vectorization(preds)
    corner_dir_map = draw_corner_dir_map(vectorized_preds, im_size)
    return vectorized_preds, corner_dir_map


def get_room_shape(room_preds, room_contour):
    vectorized_preds = vectorization(room_preds)
    corner_sequence = list()
    for vert in room_contour:
        matched_corner = _find_matched_corner(vert, [x['corner'] for x in vectorized_preds])
        if matched_corner is not None:
            if len(corner_sequence) == 0 or matched_corner != corner_sequence[-1]:
                corner_sequence.append(matched_corner)

    shape_img = np.zeros([room_preds.shape[1], room_preds.shape[2], 3])
    for idx, corner_id in enumerate(corner_sequence):
        cv2.circle(shape_img, vectorized_preds[corner_id]['corner'], 2, (255, 0, 0), 2)
        next_idx = 0 if idx == len(corner_sequence) - 1 else idx + 1
        cv2.line(shape_img, vectorized_preds[corner_id]['corner'],
                 vectorized_preds[corner_sequence[next_idx]]['corner'], (255, 255, 255), 1)

    return shape_img


def _find_matched_corner(vert, corners, thresh=10):
    y, x = vert.tolist()
    matched = None
    for idx, corner in enumerate(corners):
        if (x - corner[0]) ** 2 + (y - corner[1]) ** 2 <= thresh ** 2:
            matched = idx
            break
    return matched


def draw_corner_dir_map(room_preds, im_size):
    room_map = np.zeros([im_size, im_size, 3])
    for pred in room_preds:
        corner = pred['corner']
        dirs = pred['edge_dirs']
        cv2.circle(room_map, (corner[0], corner[1]), 2, (255, 0, 0), 2)
        for edge_dir in dirs:
            unit_vec = (np.cos(edge_dir * 10 / 180 * np.pi), np.sin(edge_dir * 10 / 180 * np.pi))
            end_point = (int(corner[0] + unit_vec[0] * 10), int(corner[1] - unit_vec[1] * 10))
            cv2.line(room_map, corner, end_point, (255, 255, 255), 1)
    return room_map


def process_edge_binning(binning, thresh=.5):
    picked_bins = np.where(binning > thresh)[0].tolist()
    return picked_bins


####
#   Non-Maximum Suppression
####

def nms_2d_naive(x, thresh=.5, nms_dist=15):
    """
    Given a corner heatmap, apply non-maximum suppresion and get the vectorized corner predictions
    :param x: corner heatmap, values are between 0 and 1
    """
    all_yx = np.where(x > thresh)
    all_preds = [(x, y) for x, y in zip(all_yx[1], all_yx[0])]

    final_preds = list()
    while len(all_preds) > 0:
        pred_init = all_preds[0]
        picked = [0, ]
        xs = [pred_init[0], ]
        ys = [pred_init[1], ]
        for idx, pred in enumerate(all_preds[1:]):
            dist = np.sqrt((pred[0] - pred_init[0]) ** 2 + (pred[1] - pred_init[1]) ** 2)
            if dist <= nms_dist:
                picked.append(idx + 1)
                xs.append(pred[0])
                ys.append(pred[1])

        final_preds.append((int(np.round(np.mean(xs))), int(np.round(np.mean(ys)))))
        all_preds = [all_preds[i] for i in range(len(all_preds)) if i not in picked]

    return final_preds


def nms_2d(x, thresh=.5):
    binary_im = x > thresh
    labeled, n_components = ndimage.label(binary_im)

    final_preds = list()
    for idx in range(1, n_components + 1):
        yx = np.where(labeled == idx)
        avg_y = np.mean(yx[0])
        avg_x = np.mean(yx[1])
        corner = (int(np.round(avg_x)), int(np.round(avg_y)))
        final_preds.append(corner)

    return final_preds


####
#   End NMS
####
