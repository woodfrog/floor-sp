import cv2
import numpy as np
from scipy import ndimage
import pdb
from scipy.ndimage.morphology import binary_erosion
from scipy.misc import imresize

def _convert_color(color):
    converted_color = [color[0] * 255, color[1] * 255, color[2] * 255]
    converted_color = [int(x) for x in converted_color]
    converted_color = [max(0, x - 40) for x in converted_color]
    return converted_color


def draw_final_floorplan(scale, input_img, all_room_edges, global_graph, room_ids, room_label_map, room_colors, flip_y=True, to_idx=None, rooms_info=None):
    all_room_edges = all_room_edges[:to_idx]
    room_colors = room_colors[:to_idx]

    from collections import defaultdict
    global_graph = defaultdict(list)
    for room_edges in all_room_edges:
        for edge in room_edges:
            c1, c2 = edge
            global_graph[c1] += [c2, ]
            global_graph[c2] += [c1, ]

    for corner in global_graph:
        global_graph[corner] = set(global_graph[corner])

    colorMap = [_convert_color(color) for color in room_colors]
    colorMap += [(0, 0, 0) for _ in range(3)] + [(65, 65, 65)] + [(0, 0, 0)]
    colorMap = np.asarray(colorMap)

    borderColorMap = [(128, 192, 64), (192, 64, 64), (192, 128, 64), (0, 128, 192), (0, 128, 192), (0, 128, 192),
                      (128, 64, 160), (128, 192, 64), (192, 64, 0), (255, 255, 255)]
    borderColorMap += [(0, 0, 0) for i in range(3)] + [(130, 130, 130)] + [(0, 0, 0)]
    borderColorMap = np.asarray(borderColorMap)

    borderColorMap = np.zeros_like(borderColorMap, dtype=np.uint8)

    colorMap = np.concatenate([np.full(shape=(1, 3), fill_value=0), colorMap, borderColorMap], axis=0).astype(
        np.uint8)
    # when using opencv, we need to flip, from RGB to BGR
    colorMap = colorMap[:, ::-1]

    alpha_channels = np.ones(colorMap.shape[0], dtype=np.uint8)
    alpha_channels = alpha_channels * 255
    alpha_channels[1:len(all_room_edges)+1] = 100

    colorMap = np.concatenate([colorMap, np.expand_dims(alpha_channels, axis=-1)], axis=-1)

    room_segmentation = np.zeros((scale, scale), dtype=np.int32)

    wallLineWidth = 3
    rooms = []
    for room_idx, room_edges in enumerate(all_room_edges):
        canvas = np.zeros((scale, scale), dtype=np.int32)
        for edge in room_edges:
            if flip_y:
                c_1 = (edge[0][0], 256 - edge[0][1])
                c_2 = (edge[1][0], 256 - edge[1][1])
                edge = (c_1, c_2)
            scaled_edge = rescale_edge(edge, 256, scale)
            cv2.line(canvas, scaled_edge[0], scaled_edge[1], color=1, thickness=wallLineWidth)
        canvas = 1 - canvas
        label, num_features = ndimage.label(canvas)
        assert num_features == 2
        num_label1 = (label == 1).sum()
        num_label2 = (label == 2).sum()
        room_label = 1 if num_label2 > num_label1 else 2
        mask = np.zeros([scale, scale], dtype=np.int32)
        mask[np.where(label == room_label)] = 1
        mask = cv2.dilate(mask.astype(np.uint8), np.ones((3, 3)), iterations=3)
        rooms.append((mask, room_idx + 1))

    for mask, label in rooms:
        room_segmentation[np.where(mask == 1)] = label

    image = colorMap[room_segmentation.reshape(-1)].reshape((scale, scale, 4))

    ## Draw density image on the floorplan visualization
    input_img = np.flipud(input_img)
    if len(input_img.shape) == 2:
        input_img = np.stack([input_img]*3, axis=-1)
    input_img = np.concatenate([input_img, np.ones([input_img.shape[0], input_img.shape[1], 1])], axis=-1).astype(np.uint8)
    input_img[:, :, -1] = 255
    input_img = imresize(input_img, [scale, scale, 4])
    # rooms_info = sorted(rooms_info, key=lambda x: x['mask'].sum())
    # pred_mask = rooms_info[to_idx]['mask']
    # pred_mask = binary_erosion(pred_mask, iterations=2).astype(np.uint8)
    # pred_mask = imresize(pred_mask, [scale, scale])
    # input_img += np.stack([pred_mask * 255] * 4, axis=-1)
    image = np.clip(image + input_img, 0, 255)

    pointColor = tuple((np.array([0.9, 0.3, 0.3, 1]) * 255).astype(np.uint8).tolist())

    # draw points
    for point in global_graph.keys():
        if flip_y:
            point = (point[0], 256 - point[1])
        scaled_point = rescale_corner(point, 256, scale)
        cv2.circle(image, scaled_point, color=pointColor, radius=8, thickness=-1)
        cv2.circle(image, scaled_point, color=(255, 255, 255, 255), radius=4, thickness=-1)

    # draw walls (edges)
    wall_records = dict()
    for point, connections in global_graph.items():
        if flip_y:
            point = (point[0], 256 - point[1])
        scaled_point = rescale_corner(point, 256, scale)
        for to_point in connections:
            if flip_y:
                to_point = (to_point[0], 256 - to_point[1])
            scaled_to_point = rescale_corner(to_point, 256, scale)
            wall = (scaled_point, scaled_to_point)
            if wall in wall_records:
                continue
            cv2.line(image, scaled_point, scaled_to_point, color=(0, 0, 0, 255), thickness=wallLineWidth)
            wall_records[(point, to_point)] = 1
            wall_records[(to_point, point)] = 1

    return image


def rescale_edge(e, in_scale, out_scale):
    new_c0 = rescale_corner(e[0], in_scale, out_scale)
    new_c1 = rescale_corner(e[1], in_scale, out_scale)
    new_e = (new_c0, new_c1)
    return new_e


def rescale_corner(c, in_scale, out_scale):
    c0 = int(np.round(c[0] * 1.0 / in_scale * out_scale))
    c1 = int(np.round(c[1] * 1.0 / in_scale * out_scale))
    new_c = (c0, c1)
    return new_c


def findBestTextLabelCenter(xs, ys, label_half_size_x, label_half_size_y):
    center = np.array([xs.mean(), ys.mean()])
    room_points = np.array([xs, ys]).transpose()
    min_point = room_points.min(axis=0, keepdims=True)
    max_point = room_points.max(axis=0, keepdims=True)
    size = np.array([label_half_size_x, label_half_size_y])
    avail_min_point = min_point + size
    avail_max_point = max_point - size

    avail_points = np.logical_and(room_points > avail_min_point, room_points < avail_max_point)
    avail_points = np.all(avail_points, axis=1)
    room_points_not_in_square = np.full(room_points.shape[0], 1, dtype=np.bool)

    good_points_mask = np.logical_and(avail_points, room_points_not_in_square)
    good_points = room_points[good_points_mask]
    good_points_center_dist = np.linalg.norm(good_points - center, axis=1)

    if len(good_points) == 0:
        return None
    best_point_idx = np.argmin(good_points_center_dist, axis=0)
    return good_points[best_point_idx]
