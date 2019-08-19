import numpy as np
from utils.floorplan_utils.floorplan_misc import len_edge, unit_v_edge
from utils.data_utils import compute_bin_idx, get_edge_pixels

import pdb


def room_init_setup(all_nodes, corners_info, directions_rad, room_bbox, mask, edgeness, inter_region, inter_edge, tried_sources):
    valid = False
    while not valid:
        source, sink, starting_line = _repick_source_sink(corners_info, all_nodes, directions_rad, room_bbox, tried_sources, mask, edgeness, inter_region, inter_edge)
        if sink is None:
            tried_sources.append(source)
            continue
        else:
            tried_sources.append(source)
            valid = True
    if valid:
        return source, sink, starting_line, tried_sources
    else:
        raise RuntimeError('Cannot find proper source and sink node')


def _repick_source_sink(corners_info, all_nodes, directions, room_bbox, picked_sources, mask, edgeness, inter_region, inter_edge):
    if len(corners_info) == len(picked_sources):
        raise RuntimeError('Cannot find proper source and sink nodes')
    corner_confs = list()
    for corner_info in corners_info:
        corner_confs.append(corner_info['corner_conf'])
    for idx in picked_sources:
        corner_confs[idx] = 0

    max_corner_idx = np.argmax(corner_confs)
    max_corner = corners_info[max_corner_idx]['corner']
    if mask[max_corner[1], max_corner[0]] == 1 or inter_region[max_corner[1], max_corner[0]] >= 1:
    # if mask[max_corner[1], max_corner[0]] == 1:
        return max_corner_idx, None, None

    adj_corner_idx, starting_line = _find_adj_corner(max_corner_idx, corners_info, mask, edgeness, inter_region, inter_edge, all_nodes, directions, room_bbox)

    return max_corner_idx, adj_corner_idx, starting_line


def _find_adj_corner(corner_idx, corners_info, mask, edgeness, inter_region, inter_edge, all_nodes, directions, room_bbox):
    room_corners = [info['corner'] for info in corners_info]
    source_c = room_corners[corner_idx]
    dists = [np.sqrt((source_c[0] - c[0]) ** 2 + (source_c[1] - c[1]) ** 2) for c in room_corners]
    order = np.argsort(dists).tolist()

    best_adj_idx = None
    for adj_idx in order:
        if adj_idx == corner_idx:
            continue
        adj_corner = room_corners[adj_idx]
        if mask[adj_corner[1], adj_corner[0]] == 1 or inter_region[adj_corner[1], adj_corner[0]] >= 1:
        # if mask[adj_corner[1], adj_corner[0]] == 1:
            continue
        adjust_adj_idx = _adjust_sink_node(corner_idx, adj_idx, directions, all_nodes)
        if adjust_adj_idx is None:
            continue
        adj_corner = all_nodes[adjust_adj_idx]
        starting_line, cross_mask = _compute_starting_line(source_c, adj_corner, room_bbox, mask)
        if not cross_mask:
            continue

        unit_v = unit_v_edge((source_c, adj_corner))
        len_v = len_edge((source_c, adj_corner))
        ext_source = (source_c[0] - int(unit_v[0] * len_v / 5.0), source_c[1] - int(unit_v[1] * len_v / 5.0))
        ext_adj = (adj_corner[0] + int(unit_v[0] * len_v / 5.0), adj_corner[1] + int(unit_v[1] * len_v / 5.0))
        all_pixels = get_edge_pixels(ext_source, ext_adj)
        length = len(all_pixels)
        count_on_edge = 0
        count_on_mask = 0
        for pixel in all_pixels:
            if not (room_bbox[0] <= pixel[0] <= room_bbox[0] + room_bbox[2] and room_bbox[1] <= pixel[1] <= room_bbox[1] + room_bbox[3]):
                length -= 1
                continue
            if edgeness[pixel[1], pixel[0]] == 1 or inter_edge[pixel[1], pixel[0]] >= 1:
                count_on_edge += 1
            if mask[pixel[1], pixel[0]] == 1 or inter_region[pixel[1], pixel[0]] >= 1:
            # if mask[pixel[1], pixel[0]] == 1:
                count_on_mask += 1

        if count_on_mask > 5 or count_on_edge * 1.0 / (length-len_v * 2 / 5) <= 0.1:
            continue
        best_adj_idx = all_nodes.index(adj_corner)
        break

    if best_adj_idx is not None:
        return best_adj_idx, starting_line
    else:
        return None, None


def _adjust_sink_node(source, end, directions_rad, all_nodes):
    expanded_rad = list()
    for rad in directions_rad:
        expanded_rad.append(rad)
        expanded_rad.append(rad + np.pi)

    expanded_rad_cos = [np.cos(x) for x in expanded_rad]
    source_c = all_nodes[source]
    end_c = all_nodes[end]
    vec = (end_c[0] - source_c[0], -(end_c[1] - source_c[1]))
    vec_len = np.sqrt(vec[0] ** 2 + vec[1] ** 2)
    cos = vec[0] / vec_len

    cos_diff = [abs(x - cos) for x in expanded_rad_cos]
    min_rad_idx = np.argmin(cos_diff)
    adjusted_cos = expanded_rad_cos[min_rad_idx]

    rad = expanded_rad[min_rad_idx]
    if vec[1] < 0 and 0 < rad < np.pi or vec[1] > 0 and rad > np.pi:
        rad = 2 * np.pi - rad
        if rad not in expanded_rad:
            return None

    if np.abs(adjusted_cos) < 1e-5:  # 90 and 270 degrees are special cases
        adjusted_sin = 1.0 if vec[1] > 0 else -1.0
    else:
        adjusted_sin = np.sin(rad)
    new_end_x = int(np.round(source_c[0] + vec_len * adjusted_cos))
    new_end_y = int(
        np.round(source_c[1] - vec_len * adjusted_sin))  # the minus here is important! Image frame system is different
    new_end = (new_end_x, new_end_y)

    if new_end in all_nodes:
        new_end_idx = all_nodes.index(new_end)
        return new_end_idx
    else:
        return None



'''
    Starting line related stuffs for preventing trivial solutions in pixel-based graph
'''


def _compute_starting_line(source_c, end_c, bbox, mask):
    center_x = (source_c[0] + end_c[0]) / 2
    center_y = (source_c[1] + end_c[1]) / 2
    length = np.sqrt((end_c[1] - source_c[1]) ** 2 + (end_c[0] - source_c[0]) ** 2)
    norm_vec = (-(end_c[1] - source_c[1]) / length, (end_c[0] - source_c[0]) / length)

    i = 1
    end_1 = (center_x, center_y)
    cross_mask = False

    while True:
        end_1_x = int(np.round(end_1[0] + norm_vec[0] * i))
        end_1_y = int(np.round(end_1[1] + norm_vec[1] * i))

        if end_1_x >= bbox[0] + bbox[2] or end_1_x <= bbox[0] or end_1_y >= bbox[1] + bbox[3] or end_1_y <= bbox[1]:
            break
        if mask[end_1_y][end_1_x] == 1:
            cross_mask = True
            break
        i += 1
    end_1 = (end_1_x, end_1_y)

    i = -1
    end_2 = (center_x, center_y)
    while True:
        end_2_x = int(np.round(end_2[0] + norm_vec[0] * i))
        end_2_y = int(np.round(end_2[1] + norm_vec[1] * i))

        if end_2_x >= bbox[0] + bbox[2] or end_2_x <= bbox[0] or end_2_y >= bbox[1] + bbox[3] or end_2_y <= bbox[1]:
            break
        if mask[end_2_y][end_2_x] == 1:
            cross_mask = True
            break
        i -= 1
    end_2 = (end_2_x, end_2_y)

    return (end_1, end_2), cross_mask

