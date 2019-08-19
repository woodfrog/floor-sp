import os
import numpy as np
import copy
from scipy.ndimage.filters import gaussian_filter, convolve
from scipy.ndimage.morphology import binary_dilation
from scipy.misc import imrotate
from scipy import ndimage
import cv2
import pdb


def generate_corner_annot(corner_dict, lines):
    corner_annots = compute_corner_bins(corner_dict, lines, num_bins=36)
    corner_gt_map = draw_gt_corner_map(corner_annots, num_bins=36, im_size=256)
    return corner_annots, corner_gt_map


def draw_gt_corner_map(corner_annots, num_bins, im_size):
    """
        Draw global corner map
    """
    # gt_maps = [np.zeros([im_size, im_size]) for _ in range(num_bins + 1)]
    gt_map = np.zeros([im_size, im_size, 1 + num_bins])
    neighbour_kernel_corner = disk(7)
    neighbour_kernel_dir = disk(7)
    for corner, corner_annot in corner_annots.items():
        x, y = corner_annot['x'], corner_annot['y']
        if x < 0 or x > 255 or y < 0 or y > 255:
            # due to the augmentation, we can sometimes have corners out of the image boundary
            continue
        gt_map[y, x, 0] = 1
        for idx in corner_annot['connections']:
            gt_map[y, x, 1 + idx] = 1
    for idx in range(gt_map.shape[2]):
        if idx == 0:
            gt_map[:, :, idx] = convolve(gt_map[:, :, idx], neighbour_kernel_corner)
        else:
            gt_map[:, :, idx] = convolve(gt_map[:, :, idx], neighbour_kernel_dir)

    gt_map = (gt_map > 0.5).astype(np.float64)

    gt_map = np.transpose(gt_map, [2, 0, 1])
    return gt_map


def compute_corner_bins(corner_dict, lines, num_bins=36):
    corner_bin_dict = dict()
    for corner, corner_data in corner_dict.items():
        if corner in corner_bin_dict:
            print('duplicate corners ...')
            pdb.set_trace()
            continue
        corner_bin_dict[corner] = dict()
        corner_bin_dict[corner]['x'] = corner_data['img_x']
        corner_bin_dict[corner]['y'] = corner_data['img_y']
        corner_bin_dict[corner]['connections'] = list()
        corner_bin_dict[corner]['bin_map'] = dict()
        skip_num = 0
        for line in lines:
            end_points = copy.copy(line['points'])  # make a copy of the list
            if corner in end_points:
                end_points.remove(corner)
                assert len(end_points) == 1
                if end_points[0] == corner:
                    skip_num += 1
                    continue  # skip a line with two same endpoints
                other_data = corner_dict[end_points[0]]
                other_x = other_data['img_x']
                other_y = other_data['img_y']
                if other_x == corner_bin_dict[corner]['x'] and other_y == corner_bin_dict[corner]['y']:
                    continue  # skip two corners with the same coordinates
                vec_to_other = (
                    other_x - corner_bin_dict[corner]['x'], -(other_y - corner_bin_dict[corner]['y']))  # flip y axis
                bin_idx = compute_bin_idx(vec_to_other, num_bins)
                corner_bin_dict[corner]['connections'].append(bin_idx)
                corner_bin_dict[corner]['bin_map'][(other_x, other_y)] = bin_idx
            # assert len(corner_bin_dict[corner]['connections']) == len(corner_data['lines']) - skip_num
    return corner_bin_dict


def compute_bin_idx(vec, num_bins):
    assert not (vec[0] == 0 and vec[1] == 0)
    cos = vec[0] / np.sqrt(vec[0] ** 2 + vec[1] ** 2)
    cos = np.clip(cos, -1.0, 1.0)
    theta = np.arccos(cos)
    if vec[1] < 0:
        theta = 2 * np.pi - theta
    bin_idx = int(np.round(theta / (np.pi * 2 / (num_bins * 1.0))))
    if bin_idx == num_bins:
        bin_idx = 0
    return bin_idx


def get_room_gt_map(room_corners, im_size, num_bins=36):
    """
        Draw per-room corner map
    """
    gt_map = np.zeros([im_size, im_size, 1 + num_bins])
    for idx, corner in enumerate(room_corners):
        x, y = corner
        gt_map[y, x, 0] = 1
        # generate binning map
        prev_corner = room_corners[idx - 1]
        next_corner = room_corners[idx + 1] if idx != len(room_corners) - 1 else room_corners[0]
        for other_corner in [prev_corner, next_corner]:
            other_x, other_y = other_corner
            vec_to_other = (other_x - x, -(other_y - y))
            bin_idx = compute_bin_idx(vec_to_other, num_bins)
            gt_map[y, x, 1 + bin_idx] = 1

    for i in range(1 + num_bins):
        gt_map[:, :, i] = gaussian_filter(gt_map[:, :, i], sigma=2)
    if gt_map.max() > 0:
        gt_map = gt_map * (1 / gt_map.max())
    gt_map = np.transpose(gt_map, [2, 0, 1])

    return gt_map


def get_room_edge_map(room_corners, im_size, num_bins=36):
    edge_map = np.zeros([im_size, im_size, 1 + num_bins])
    for idx, corner in enumerate(room_corners):
        next_idx = idx + 1 if idx != len(room_corners) - 1 else 0
        next_c = room_corners[next_idx]
        all_pixels = get_edge_pixels(corner, next_c)

        if corner[0] == next_c[0] and corner[1] == next_c[1]:
            continue

        vec_to_next = (next_c[0] - corner[0], -(next_c[1] - corner[1]))
        bin_to_next = compute_bin_idx(vec_to_next, num_bins)
        vec_back = (corner[0] - next_c[0], -(corner[1] - next_c[1]))
        bin_back = compute_bin_idx(vec_back, num_bins)

        if 0 <= corner[1] <= 255 and 0 <= corner[0] <= 255:
            edge_map[corner[1], corner[0], 0] = 1
            edge_map[corner[1], corner[0], 1 + bin_to_next] = 1
        if 0 <= next_c[1] <= 255 and 0 <= next_c[0] <= 255:
            edge_map[next_c[1], next_c[0], 1 + bin_back] = 1
        for pixel in all_pixels[1:-1]:
            if 0 <= pixel[1] <= 255 and 0 <= pixel[0] <= 255:
                edge_map[pixel[1], pixel[0], 0] = 1
                edge_map[pixel[1], pixel[0], 1 + bin_to_next] = 1
                edge_map[pixel[1], pixel[0], 1 + bin_back] = 1
    for i in range(1 + num_bins):
        edge_map[:, :, i] = binary_dilation(edge_map[:, :, i], iterations=2)
    edge_map = np.transpose(edge_map, [2, 0, 1])
    edge_map = (edge_map > 0.1).astype(np.float64)
    return edge_map


def get_single_corner_map(corner, corner_connections, im_size):
    map = np.zeros([im_size, im_size])
    cv2.circle(map, corner, 4, 255, 3)

    for connection in corner_connections:
        vec_x = np.cos(connection * 10 / 180 * np.pi)
        vec_y = np.sin(connection * 10 / 180 * np.pi)
        end_x = int(np.round(corner[0] + 255 * vec_x))
        end_y = int(np.round(corner[1] - 255 * vec_y))
        end_x = np.clip(end_x, 0, 255)
        end_y = np.clip(end_y, 0, 255)
        cv2.line(map, corner, (end_x, end_y), (255, 0, 0), 2)

    map = map / 255.0
    return map


def get_edge_map(corner1, corner2, im_size):
    map = np.zeros([im_size, im_size])
    cv2.circle(map, corner1, 4, 255, 3)
    cv2.circle(map, corner2, 4, 255, 3)
    cv2.line(map, corner1, corner2, 255, 4)
    map = map / 255.0
    return map


def generate_edge_annot(all_room_corners, im_size, num_bins=36):
    global_edge_map = np.zeros([1+num_bins, im_size, im_size])
    for room_corners in all_room_corners:
        room_edge_map = get_room_edge_map(room_corners, im_size, num_bins=num_bins)
        global_edge_map += room_edge_map
    global_edge_map = np.clip(global_edge_map, 0, 1)
    return global_edge_map

# -----------------------------------------------------
# -- Augmentation
# -----------------------------------------------------


def corner_rotate(x, y, rad, im_size):
    rot_center = (im_size // 2 - 0.5, im_size // 2 - 0.5)
    rot_x = rot_center[0] + (x - rot_center[0]) * np.cos(rad) - (y - rot_center[1]) * np.sin(rad)
    rot_y = rot_center[1] + (x - rot_center[0]) * np.sin(rad) + (y - rot_center[1]) * np.cos(rad)
    return rot_x, rot_y


def augment_rotation(topview_img, mean_normal, room_masks_map, corner_dict):
    topview_img = np.transpose(topview_img, [1, 2, 0])
    mean_normal = np.transpose(mean_normal, [1, 2, 0])

    degree = np.random.uniform(low=0, high=360, size=(1,))  # clock-wise rotation
    degree = degree // 10 * 10  # make the degree a multiplier of 10
    rad = degree / 180 * np.pi
    im_size = topview_img.shape[0]

    for corner_id in corner_dict:
        x = corner_dict[corner_id]['img_x']
        y = corner_dict[corner_id]['img_y']
        rot_x, rot_y = corner_rotate(x, y, rad, im_size)
        corner_dict[corner_id]['img_x_old'] = corner_dict[corner_id]['img_x']
        corner_dict[corner_id]['img_y_old'] = corner_dict[corner_id]['img_y']
        corner_dict[corner_id]['img_x'] = int(np.round(rot_x))
        corner_dict[corner_id]['img_y'] = int(np.round(rot_y))
        corner_dict[corner_id]['rotated'] = degree

    rot_topview_img = imrotate(topview_img, angle=-degree)  # imrotate does counter-clockwise by default

    # FIXME: the rotation of surface normal
    rot_mean = imrotate(mean_normal, angle=-degree).astype(np.float64) / 255.0
    rot_room_masks_pred = imrotate(room_masks_map, angle=-degree).astype(np.float64) / 255.0
    new_normal_x = rot_mean[:, :, 0] * np.cos(rad) - rot_mean[:, :, 1] * np.sin(rad)
    new_normal_y = rot_mean[:, :, 0] * np.sin(rad) + rot_mean[:, :, 1] * np.cos(rad)
    rot_mean[:, :, 0] = new_normal_x
    rot_mean[:, :, 1] = new_normal_y

    return np.transpose(rot_topview_img, [2, 0, 1]), np.transpose(rot_mean, [2, 0, 1]), rot_room_masks_pred, corner_dict


def corner_random_jittering(corner, radius=5):
    jittering = np.random.normal(0, scale=np.sqrt(5), size=(2,))
    jittering = np.round(np.clip(jittering, -radius, radius)).astype(np.int32)
    new_corner = (corner[0] + jittering[0], corner[1] + jittering[1])
    return new_corner


def connections_random_dropping(connections, keep_prob=0.75):
    new_connections = dict()
    for corner, edge_dirs in connections.items():
        new_dirs = list()
        for edge_dir in edge_dirs:
            if np.random.uniform(0, 1) <= keep_prob:
                new_dirs.append(edge_dir)
        new_connections[corner] = new_dirs
    return new_connections


# -----------------------------------------------------
# -- END Augmentation
# -----------------------------------------------------

def disk(k):
    ax = np.arange(-k // 2 + 1., k // 2 + 1.)
    xx, yy = np.meshgrid(ax, ax)

    kernel = (np.sqrt(pow(xx, 2) + pow(yy, 2)) <= (k - 1) / 2).astype(np.float32)

    return kernel


def get_edge_pixels(point1, point2):
    ends = np.array([point1, point2])
    d0, d1 = np.abs(np.diff(ends, axis=0))[0]
    if d0 > d1:
        return np.c_[np.linspace(ends[0, 0], ends[1, 0], d0 + 1, dtype=np.int32),
                     np.round(np.linspace(ends[0, 1], ends[1, 1], d0 + 1))
                         .astype(np.int32)]
    else:
        return np.c_[np.round(np.linspace(ends[0, 0], ends[1, 0], d1 + 1))
                         .astype(np.int32),
                     np.linspace(ends[0, 1], ends[1, 1], d1 + 1, dtype=np.int32)]


# -----------------------------------------------------
# -- Functions for processing per-room data
# -----------------------------------------------------

def get_direction_hist(edge_preds):
    edge_pixels = np.where(edge_preds[0] > 0.5)
    num_bins = edge_preds.shape[0] - 1
    direction_confs = np.transpose(edge_preds[1:, :, :], [1, 2, 0])[edge_pixels]

    direction_histogram = dict([x, 0] for x in range(num_bins))
    for i in range(direction_confs.shape[0]):
        top_2_directions = np.argsort(direction_confs[i])[-2:]
        for direction in top_2_directions:
            direction_histogram[direction] += 1
    return direction_histogram


def get_room_heatmap(room_corners_info, preds_heatmap, mode):
    room_heatmap = np.zeros(preds_heatmap.shape)
    if mode == 'corner':
        for room_corner_info in room_corners_info:
            corner = room_corner_info['corner']
            y_min = max(corner[1] - 5, 0)
            x_min = max(corner[0] - 5, 0)
            y_max = min(corner[1] + 5, preds_heatmap.shape[0])
            x_max = min(corner[0] + 5, preds_heatmap.shape[0])
            corner_patch = preds_heatmap[y_min: y_max, x_min: x_max]
            room_heatmap[y_min: y_max, x_min: x_max] = corner_patch
    elif mode == 'edge':
        room_corners = [info['corner'] for info in room_corners_info]
        room_bbox = get_room_bbox([], room_corners, preds_heatmap.shape[-1], extra_margin=5)
        if len(preds_heatmap.shape) > 2:
            room_heatmap[:, room_bbox[1]:room_bbox[1] + room_bbox[3],
                           room_bbox[0]:room_bbox[0] + room_bbox[2]] = preds_heatmap[:, room_bbox[1]:room_bbox[1] + room_bbox[3],
                           room_bbox[0]:room_bbox[0] + room_bbox[2]]
        else:
            room_heatmap[room_bbox[1]:room_bbox[1] + room_bbox[3],
                           room_bbox[0]:room_bbox[0] + room_bbox[2]] = preds_heatmap[room_bbox[1]:room_bbox[1]+room_bbox[3], room_bbox[0]:room_bbox[0]+room_bbox[2]]
    else:
        raise ValueError('Invalid mode {}'.format(mode))
    return room_heatmap


def get_room_bbox(contour, room_corners, im_size, extra_margin):
    all_x = list()
    all_y = list()
    for vert in contour:
        all_x.append(int(vert[1]))
        all_y.append(int(vert[0]))
    for corner in room_corners:
        all_x.append(corner[0])
        all_y.append(corner[1])

    width = np.max(all_x) - np.min(all_x)  
    height = np.max(all_y) - np.min(all_y)

    if width > im_size / 2 or height > im_size / 2: 
        margin_width = 2 * extra_margin
        margin_height = 2 * extra_margin
    if width > im_size * 2 / 3 or height > im_size * 2.0 / 3:
        margin_width = 3 * extra_margin
        margin_height = 3 * extra_margin
    else:
        margin_width = extra_margin
        margin_height = extra_margin
    min_x = max(np.min(all_x) - margin_width, 0)
    max_x = min(np.max(all_x) + margin_width, im_size - 1)
    min_y = max(np.min(all_y) - margin_height, 0)
    max_y = min(np.max(all_y) + margin_height, im_size - 1)
    bbox = (min_x, min_y, max_x - min_x, max_y - min_y)
    return bbox

# -----------------------------------------------------
# -- End.  Functions for processing per-room data
# -----------------------------------------------------


def draw_room_seg_from_edges(edges, im_size):
    edge_map = np.zeros([im_size, im_size])
    for edge in edges:
        cv2.line(edge_map, edge[0], edge[1], 1, 3)
    reverse_edge_map = 1 - edge_map
    label, num_features = ndimage.label(reverse_edge_map)
    assert num_features == 2
    num_label1 = (label == 1).sum()
    num_label2 = (label == 2).sum()
    room_label = 1 if num_label2 > num_label1 else 2
    room_map = np.zeros([im_size, im_size])
    room_map[np.where(label == room_label)] = 1
    edge_map[np.where(room_map == 1)] = 0

    corner_map = np.zeros([im_size, im_size])
    for edge in edges:
        corner = edge[0]
        cv2.circle(corner_map, corner, 1, 1, 1)
    corner_map = binary_dilation(corner_map).astype(np.float32)

    return room_map, edge_map, corner_map
