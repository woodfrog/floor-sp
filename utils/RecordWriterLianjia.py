import os
import cv2
import numpy as np
import math
import json
from skimage import measure
from plyfile import PlyData, PlyElement

from floorplan_utils import calcLineDirection, MAX_NUM_CORNERS, NUM_WALL_CORNERS, getRoomLabelMap, getLabelRoomMap
from utils import getDensity, drawDensityImage


import pdb
EPSILON = 3


def _convert_room_label(room_name):
    """ Convert room label from str to index, using the mapping defined by FloorNet"""
    room_name = room_name.split('-')[0]
    original_room_label_map = getRoomLabelMap()

    room_name = room_name.replace(' ', '_').lower()
    try:
        room_label = original_room_label_map[room_name]
    except KeyError:
        room_name = 'other'
        room_label = original_room_label_map[room_name]
    return room_label


def _get_corner_type(points, lines, point_dict, allow_non_manhattan=False):
    """
        Get the corner type according to the labels defined by FloorNet.
        When allow_non_manhattan is true, non-Manhattan corners are also included
        (FloorNet drops those non-Manhattan structures).
    """
    status = True
    err_msg = ''

    L_shape_dict = {
        'RU': 6,
        'DR': 7,
        'DL': 8,
        'LU': 5,
        'MAN': -1,
    }

    T_shape_dict = {
        'LRU': 11,
        'DRU': 12,
        'DLR': 9,
        'DLU': 10,
        'MAN': -1,
    }

    adjacency_dict = dict()
    for point in points:
        adjacency_dict[point['id']] = set()

    for line in lines:
        pt1, pt2 = line['points']
        adjacency_dict[pt1].add(pt2)
        adjacency_dict[pt2].add(pt1)

    corner_type_map = dict()

    for point, set_others in adjacency_dict.items():
        list_others = list(set_others)
        pt_xy = (point_dict[point]['img_x'], point_dict[point]['img_y'])
        if len(list_others) == 1:
            other_xy = (point_dict[list_others[0]]['img_x'], point_dict[list_others[0]]['img_y'])
            if abs(other_xy[0] - pt_xy[0]) <= EPSILON and other_xy[1] > pt_xy[1]:
                label = 3
            elif abs(other_xy[0] - pt_xy[0]) <= EPSILON and other_xy[1] < pt_xy[1]:
                label = 1
            elif other_xy[0] > pt_xy[0] and abs(other_xy[1] - pt_xy[1]) <= EPSILON:
                label = 2
            elif other_xy[0] < pt_xy[0] and abs(other_xy[1] - pt_xy[1]) <= EPSILON:
                label = 4
            else:
                status = False
                err_msg = 'invalid point connections, duplication'
                print(err_msg)
                return None, status, err_msg
        elif len(list_others) == 2 or len(list_others) == 3:
            position_str = ''
            for other in list_others:
                other_xy = (point_dict[other]['img_x'], point_dict[other]['img_y'])
                if abs(other_xy[0] - pt_xy[0]) <= EPSILON and other_xy[1] > pt_xy[1]:
                    position_str += 'D'
                elif abs(other_xy[0] - pt_xy[0]) <= EPSILON and other_xy[1] < pt_xy[1]:
                    position_str += 'U'
                elif other_xy[0] > pt_xy[0] and abs(other_xy[1] - pt_xy[1]) <= EPSILON:
                    position_str += 'R'
                elif other_xy[0] < pt_xy[0] and abs(other_xy[1] - pt_xy[1]) <= EPSILON:
                    position_str += 'L'
                else:
                    if not allow_non_manhattan:
                        status = False
                        err_msg = 'invalid point connections'
                        print(err_msg)
                        return None, status, err_msg
                    else:
                        # break directly, since currently all non-M corners are categorized as one type
                        position_str = 'MAN'
                        break
            if len(list_others) == 2:
                shape_dict = L_shape_dict
            else:
                shape_dict = T_shape_dict
            try:
                if position_str != 'MAN':
                    label = shape_dict[''.join(sorted(position_str))]
                else:
                    label = shape_dict[position_str]
            except Exception as e:
                status = False
                err_msg = 'invalid connections combination, missing edges'
                return None, status, err_msg
        else:
            # todo: might further check here
            assert len(list_others) >= 4
            label = 13

        corner_type_map[point] = label

    return corner_type_map, status, err_msg


def _get_internal_point(point_ids, point_dict):
    top_line = None
    top_line_idx = None
    for i, (pt1_id, pt2_id) in enumerate(zip(point_ids, point_ids[1:] + point_ids[:1])):
        try:
            pt1 = (point_dict[pt1_id]['img_x'], point_dict[pt1_id]['img_y'])
            pt2 = (point_dict[pt2_id]['img_x'], point_dict[pt2_id]['img_y'])
        except KeyError:
            print('KeyError when looking up point dict')
            pdb.set_trace()
        if not _is_horizontal_line((pt1, pt2)):
            continue
        else:
            assert _is_horizontal_line((pt1, pt2))
            if top_line is None:
                top_line = (pt1, pt2)
                top_line_idx = i
            else:
                if pt1[1] < top_line[0][1]:
                    top_line = (pt1, pt2)
                    top_line_idx = i

                # special case, choosing longer one as top line
                if abs(pt1[1] - top_line[0][1]) <= EPSILON and abs(pt1[0] - pt2[0]) > abs(
                        top_line[0][0] - top_line[1][0]):
                    top_line = (pt1, pt2)
                    top_line_idx = i

    top_point = None
    top_point_idx = None
    for i, pt_id in enumerate(point_ids):
        pt = (point_dict[pt_id]['img_x'], point_dict[pt_id]['img_y'])
        if top_point is None:
            top_point = pt
            top_point_idx = i
        else:
            if pt[1] < top_point[1]:
                top_point = pt
                top_point_idx = i

    # Manhattan case, there is a top horizontal line segment
    if top_line is not None and abs(top_point[1] - top_line[0][1]) <= EPSILON:
        prev_line = top_line
        prev_line_idx = top_line_idx

        while _is_horizontal_line(prev_line):
            prev_pt = (
                point_dict[point_ids[prev_line_idx - 1]]['img_x'], point_dict[point_ids[prev_line_idx - 1]]['img_y'])
            prev_line = (prev_pt, prev_line[0])
            prev_line_idx -= 1

        next_line = top_line
        next_line_idx = top_line_idx
        while _is_horizontal_line(next_line):
            next_line_idx += 1
            if next_line_idx >= len(point_ids):
                next_line_idx -= len(point_ids)
            next_pt = (point_dict[point_ids[next_line_idx]]['img_x'], point_dict[point_ids[next_line_idx]]['img_y'])
            next_line = (next_line[1], next_pt)

        # use the shorter one to compute the internal point, avoid crossing
        if abs(prev_line[0][1] - prev_line[1][1]) <= 2 * EPSILON and abs(
                next_line[0][1] - next_line[1][1]) <= 2 * EPSILON:
            # TODO: this one is a rough estimate, not safe enough here
            internal_point = (
                (top_line[0][0] + next_line[1][0]) // 2, (top_line[0][1] + next_line[1][1] + 2 * EPSILON) // 2)
        elif abs(prev_line[0][1] - prev_line[1][1]) <= 2 * EPSILON:
            internal_point = ((top_line[0][0] + next_line[1][0]) // 2, (top_line[0][1] + next_line[1][1]) // 2)
        elif abs(next_line[0][1] - next_line[1][1]) <= 2 * EPSILON:
            internal_point = ((top_line[1][0] + prev_line[0][0]) // 2, (top_line[1][1] + prev_line[0][1]) // 2)
        elif abs(prev_line[0][1] - prev_line[1][1]) < abs(next_line[0][1] - next_line[1][1]):
            internal_point = ((top_line[1][0] + prev_line[0][0]) // 2, (top_line[1][1] + prev_line[0][1]) // 2)
        else:
            internal_point = ((top_line[0][0] + next_line[1][0]) // 2, (top_line[0][1] + next_line[1][1]) // 2)
    # Non-Manhattan case, there is no horizontal line at all, or the top-line is not consistent with top point
    else:
        next_pt_idx = top_point_idx + 1 if top_point_idx + 1 < len(point_ids) else 0
        prev_pt_idx = top_point_idx - 1
        next_pt = (point_dict[point_ids[next_pt_idx]]['img_x'], point_dict[point_ids[next_pt_idx]]['img_y'])
        prev_pt = (point_dict[point_ids[prev_pt_idx]]['img_x'], point_dict[point_ids[prev_pt_idx]]['img_y'])
        internal_point = ((prev_pt[0] + next_pt[0]) // 2, (prev_pt[1] + next_pt[1]) // 2)

    return internal_point


def _is_horizontal_line(line):
    if abs(line[0][1] - line[1][1]) <= EPSILON:
        return True
    else:
        return False


def read_scene_pc(file_path):
    with open(file_path, 'rb') as f:
        plydata = PlyData.read(f)
        dtype = plydata['vertex'].data.dtype
    print('dtype of file{}: {}'.format(file_path, dtype))

    points_data = np.array(plydata['vertex'].data.tolist())

    return points_data


def write_scene_pc(points, output_path):
    vertex = np.array([tuple(x) for x in points],
                      dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('red', 'u1'), ('green', 'u1'), ('blue', 'u1'), ])
    vertex_el = PlyElement.describe(vertex, 'vertex')
    PlyData([vertex_el]).write(output_path)  # write the new ply file


class RecordWriter:
    def __init__(self, num_points, base_dir, phase, im_size, save_prefix, allow_non_man=False, extra_test=None):
        self.num_points = num_points
        self.base_dir = base_dir
        self.ply_base_dir = os.path.join \
            (self.base_dir, 'ply')
        self.annot_base_dir = os.path.join(self.base_dir, 'json')
        self.phase = phase
        self.im_size = im_size  # HEIGHT, WIDTH = SIZE
        self.allow_non_man = allow_non_man
        self.extra_test = extra_test

        self.train_test_split = 0.9
        self.gap = 3

        self.ply_paths, self.annot_paths = self.get_filepaths()

        self.log_path = os.path.join(self.base_dir, '{}_log_{}.txt'.format(save_prefix, self.phase))
        self.log_lines = list()

        # if a separate room dataset is required
        self.room_base_dir = os.path.join(self.base_dir, 'seperate_room_data')
        if not os.path.exists(self.room_base_dir):
            os.mkdir(self.room_base_dir)
        self.room_write_base = os.path.join(self.room_base_dir, phase)
        if not os.path.exists(self.room_write_base):
            os.mkdir(self.room_write_base)

    def get_filepaths(self):
        ply_filenames = sorted(os.listdir(self.ply_base_dir))
        json_filenames = sorted(os.listdir(self.annot_base_dir))

        assert len(ply_filenames) == len(json_filenames)

        if self.phase == 'train':
            ply_filenames = ply_filenames[:int(self.train_test_split * len(ply_filenames))]
            json_filenames = json_filenames[:int(self.train_test_split * len(json_filenames))]

        elif self.phase == 'test':
            ply_filenames = ply_filenames[int(self.train_test_split * len(ply_filenames)):]
            json_filenames = json_filenames[int(self.train_test_split * len(json_filenames)):]
        else:
            raise ValueError('Invalid phase {}'.format(self.phase))

        ply_file_paths = [os.path.join(self.ply_base_dir, filename) for filename in ply_filenames]
        annot_file_paths = [os.path.join(self.annot_base_dir, filename) for filename in json_filenames]

        if self.phase == 'test' and self.extra_test is not None:
            extra_ply_base = os.path.join(self.extra_test, 'ply')
            extra_annot_base = os.path.join(self.extra_test, 'json')
            extra_ply_filenames = sorted(os.listdir(extra_ply_base))
            extra_json_filenames = sorted(os.listdir(extra_annot_base))
            extra_ply_paths = [os.path.join(extra_ply_base, filename) for filename in extra_ply_filenames]
            extra_annot_paths = [os.path.join(extra_annot_base, filename) for filename in extra_json_filenames]
            ply_file_paths += extra_ply_paths
            annot_file_paths += extra_annot_paths

        return ply_file_paths, annot_file_paths

    def write(self):
        succ_count = 0

        room_metadata = {
            'base_dir': self.room_base_dir,
            'label_room_map': getLabelRoomMap()
        }
        metadata_path = os.path.join(self.base_dir, 'room_metadata.json')

        if not os.path.exists(metadata_path):
            with open(metadata_path, 'w') as f:
                json.dump(room_metadata, f)

        num_points = 0
        num_corners = 0
        num_edges = 0
        num_rooms = 0
        non_man_rooms = 0
        for ply_file_path, annot_file_path in zip(self.ply_paths, self.annot_paths):
            succ, stats = self.write_example(ply_file_path, annot_file_path)
            if succ:
                succ_count += 1
                num_points += stats['num_points']
                num_rooms += stats['num_rooms']
                num_edges += stats['num_edges']
                num_corners += stats['num_corners']
                non_man_rooms += stats['non_manhattan_rooms']

        avg_points = num_points * 1.0 / succ_count
        avg_corners = num_corners * 1.0 / succ_count
        avg_edges = num_edges * 1.0 / succ_count
        avg_rooms = num_rooms * 1.0 / succ_count

        self.log_lines.append(
            '{} / {} samples successfully processed'.format(succ_count, len(self.ply_paths)))
        self.log_lines.append('avg points {}\tavg corners {}\navg edges {}\tavg_rooms {}'.format(avg_points, avg_corners, avg_edges, avg_rooms))
        self.log_lines.append('room num: {} \t non man rooms: {}'.format(num_rooms, non_man_rooms))
        with open(self.log_path, 'w') as f:
            for line in self.log_lines:
                f.write(line + '\n')

    def write_example(self, ply_path, annot_path):
        points = read_scene_pc(ply_path)

        xyz = points[:, :3]

        mins = xyz.min(0, keepdims=True)
        maxs = xyz.max(0, keepdims=True)

        max_range = (maxs - mins)[:, :2].max()
        padding = max_range * 0.05

        mins = (maxs + mins) / 2 - max_range / 2
        mins -= padding
        max_range += padding * 2

        xyz = (xyz - mins) / max_range  # re-scale coords into [0.0, 1.0]

        new_points = np.concatenate([xyz, points[:, 3:9]], axis=1)
        points = new_points

        topview_mean_normal = self.get_topview_mean_normal(points)
        # down-sampling points to get a subset with size 50,000
        if points.shape[0] < self.num_points:
            indices = np.arange(points.shape[0])
            points = np.concatenate([points, points[np.random.choice(indices, self.num_points - points.shape[0])]],
                                    axis=0)
            topview_points = self.get_topview_data(points)
        else:
            normal_z = points[:, 8]

            horizontal_points = np.array([point for p_i, point in enumerate(points) if abs(normal_z[p_i]) >= 0.5])
            vertical_points = np.array([point for p_i, point in enumerate(points) if abs(normal_z[p_i]) < 0.5])

            point_subsets = [horizontal_points, vertical_points]
            subset_ratio = [0.3, 0.7]
            sampled_points = list()
            for point_subset, ratio in zip(point_subsets, subset_ratio):
                sampled_indices = np.arange(point_subset.shape[0])
                np.random.shuffle(sampled_indices)
                sampled_points.append(point_subset[sampled_indices[:int(self.num_points * ratio)]])
            points = np.concatenate(sampled_points, axis=0)
            # note: only use vertical points for the full density map. Otherwise all fine details are ruined
            topview_points = self.get_topview_data(vertical_points)

        annot = self.parse_annot(annot_path, mins, max_range)

        points[:, 3:6] = points[:, 3:6] / 255 - 0.5  # normalize color

        # Prepare other g.t. related inputs to be zeros for now
        corners = annot['points']
        lines = annot['lines']
        point_dict = annot['point_dict']

        corner_type_map, status, err_msg = _get_corner_type(corners, lines, point_dict,
                                                            allow_non_manhattan=self.allow_non_man)
        if status is False:
            print(err_msg)
            self.log_lines.append(err_msg + ' in {}'.format(ply_path))
            return False, None

        for corner_id, type in corner_type_map.items():
            if point_dict[corner_id]['img_x'] >= self.im_size or point_dict[corner_id]['img_x'] < 0 or \
                            point_dict[corner_id]['img_y'] >= self.im_size or point_dict[corner_id]['img_y'] < 0:
                err_msg = 'corner out of image boundary due to small point density'
                print(err_msg)
                self.log_lines.append(err_msg)
                return False, None

        # Prepare room segmentation map, get the label for every room
        room_segmentation = np.zeros((self.im_size, self.im_size), dtype=np.uint8)
        room_segmentation_large = np.zeros((self.im_size, self.im_size), dtype=np.uint8)

        line_coords = list()
        for line_item in lines:
            point_id_1, point_id_2 = line_item['points']
            line = ((point_dict[point_id_1]['img_x'], point_dict[point_id_1]['img_y']),
                    (point_dict[point_id_2]['img_x'], point_dict[point_id_2]['img_y']))
            cv2.line(room_segmentation, line[0], line[1], color=15 + calcLineDirection(line), thickness=self.gap)
            line_coords.append(line)
            cv2.line(room_segmentation_large, line[0], line[1], color=15 + calcLineDirection(line), thickness=2)

        rooms = measure.label(room_segmentation == 0, background=0)
        rooms_large = measure.label(room_segmentation_large == 0, background=0)

        wall_idx = rooms.min()
        for pixel in [(0, 0), (0, self.im_size - 1), (self.im_size - 1, 0), (self.im_size - 1, self.im_size - 1)]:
            bg_idx = rooms[pixel[1]][pixel[0]]
            if bg_idx != wall_idx:
                break

        room_annots = annot['areas']
        room_label_map = dict()
        room_instances = list()

        for room_annot in room_annots:
            room_label = _convert_room_label(room_annot['roomName'])

            internal_point = _get_internal_point(room_annot['points'], point_dict)

            room_corners = list()

            for point_id in room_annot['points']:
                room_corners.append((point_dict[point_id]['img_x'], point_dict[point_id]['img_y']))

            room_idx = rooms[internal_point[1]][internal_point[0]]
            if room_idx == wall_idx or room_idx == bg_idx:
                error_line = 'Data error: the room label is the same as bg or wall, in {}'.format(ply_path)
                self.log_lines.append(error_line)
                return False, None
            if room_idx in room_label_map:
                error_line = 'the room idx {} is encourted multiple times, in {}'.format(room_idx, ply_path)
                self.log_lines.append(error_line)
                return False, None

            room_label_map[room_idx] = room_label

            room_large_idx = rooms_large[internal_point[1]][internal_point[0]]
            room_instances.append({
                'mask': rooms == room_idx,
                'class': room_label,
                'mask_large': rooms_large == room_large_idx,
                'room_corners': room_corners
            })

        # For testing purpose: draw the density image to check the valadity and quality
        filename, _ = os.path.splitext(os.path.basename(ply_path))
        # write_scene_pc(points, './debug/{}.ply'.format(filename))
        density_img = drawDensityImage(getDensity(points=points))
        cv2.imwrite('./debug_nonm/{}_density.png'.format(filename), density_img)

        density_img = np.stack([density_img] * 3, axis=2)
        _, annot_image = self.parse_annot(annot_path, mins, max_range, draw_img=True, img=density_img)
        cv2.imwrite('./debug_nonm/{}_annot.png'.format(filename), annot_image)

        topview_image = drawDensityImage(topview_points[:, :, -1], nChannels=3)
        room_data = {
            'topview_image': topview_image,
            'topview_mean_normal': topview_mean_normal,
            'room_instances_annot': room_instances,
            'line_coords': line_coords,
            'room_map': rooms_large,
            'bg_idx': bg_idx,
            'point_dict': point_dict,
            'lines': lines,
        }
        file_id, _ = os.path.splitext(os.path.basename(ply_path))
        output_path = os.path.join(self.room_write_base, file_id + '.npy')
        with open(output_path, 'wb') as f:
            np.save(f, room_data)

        num_non_manhattan_room = 0
        for room_instance in room_instances:
            manhattan = self.check_manhattan(room_instance)
            if not manhattan:
                num_non_manhattan_room += 1

        stats = {
            'num_points': len(vertical_points),
            'num_corners': len(point_dict),
            'num_edges': len(lines),
            'num_rooms': len(room_annots),
            'non_manhattan_rooms': num_non_manhattan_room,
        }

        return True, stats

    def get_projection_indices(self, coordinates):
        indices_map = np.zeros([self.num_points], dtype=np.int64)
        for i, coord in enumerate(coordinates):
            x, y = coord
            indices_map[i] = y * self.im_size + x
        return indices_map

    @staticmethod
    def check_manhattan(room_instance):
        room_corners = room_instance['room_corners']
        room_edges = [(room_corners[i], room_corners[i+1]) for i in range(len(room_corners)-1)]
        room_edges.append((room_corners[-1], room_corners[0]))
        manhattan = True
        for edge in room_edges:
            if edge[0][0] != edge[1][0] and edge[0][1] != edge[1][1]:
                manhattan = False
                break
        return manhattan

    def get_topview_data(self, points):
        """
        Add one more channel for counting the density
        :param points: full point cloud data
        :return: top-view of full points
        """
        topview_image = np.zeros([self.im_size, self.im_size, points.shape[-1] + 1])
        full_2d_coordinates = np.clip(np.round(points[:, :2] * self.im_size).astype(np.int32), 0, self.im_size - 1)
        for point, coord in zip(points, full_2d_coordinates):
            topview_image[coord[1], coord[0], :-1] += point
            topview_image[coord[1], coord[0], -1] += 1
        return topview_image

    def get_topview_mean_normal(self, points):
        topview_normal = np.zeros([self.im_size, self.im_size, 3])
        count = np.ones([self.im_size, self.im_size])
        full_2d_coordinates = np.clip(np.round(points[:, :2] * self.im_size).astype(np.int32), 0, self.im_size - 1)
        for point, coord in zip(points, full_2d_coordinates):
            topview_normal[coord[1], coord[0], :] += point[6:9]
            count[coord[1], coord[0]] += 1
        count = np.stack([count, count, count], 2)
        topview_normal /= count
        return topview_normal

    def parse_annot(self, file_path, mins, max_range, draw_img=False, img=None):
        with open(file_path, 'r') as f:
            data = json.load(f)

        if draw_img:
            assert img is not None

        points = data['points']
        lines = data['lines']
        line_items = data['lineItems']
        areas = data['areas']

        point_dict = dict()
        for point in points:
            point_dict[point['id']] = point

        # img = np.zeros([self.im_size, self.im_size, 3], dtype=np.uint8)

        min_x = mins[0][0]
        min_y = mins[0][1]
        width = height = max_range

        adjacency_dict = dict()
        for point in points:
            adjacency_dict[point['id']] = set()

        for line in lines:
            pt1, pt2 = line['points']
            adjacency_dict[pt1].add(pt2)
            adjacency_dict[pt2].add(pt1)

        # draw all corners
        point_img_coord_set = set()
        for point in points:
            img_x, img_y = self._draw_corner_with_scaling(img, (point['x'], point['y']), min_x, width, min_y, height)
            point_dict[point['id']]['img_x'] = img_x
            point_dict[point['id']]['img_y'] = img_y
            point_img_coord_set.add((img_x, img_y))

        # draw all line segments
        for line in lines:
            assert len(line['points']) == 2
            point_id_1, point_id_2 = line['points']
            start_pt = (point_dict[point_id_1]['img_x'], point_dict[point_id_1]['img_y'])
            end_pt = (point_dict[point_id_2]['img_x'], point_dict[point_id_2]['img_y'])
            cv2.line(img, start_pt, end_pt, (255, 0, 0))

        # draw all line with labels, such as doors, windows
        for line_item in line_items:
            start_pt = (line_item['startPointAt']['x'], line_item['startPointAt']['y'])
            end_pt = (line_item['endPointAt']['x'], line_item['endPointAt']['y'])
            line_direction = calcLineDirection((start_pt, end_pt))  # 0 means horizontal and 1 means vertical
            img_start_pt = self._draw_corner_with_scaling(img, start_pt, min_x, width, min_y, height, color=(0, 255, 0))
            img_end_pt = self._draw_corner_with_scaling(img, end_pt, min_x, width, min_y, height, color=(0, 255, 0))

            # manually prevent opening corners to be exactly overlapping with other wall corners
            if img_start_pt in point_img_coord_set:
                if line_direction == 0:
                    img_start_pt = (img_start_pt[0] + int(np.sign(img_end_pt[0] - img_start_pt[0])), img_start_pt[1])
                else:
                    img_start_pt = (img_start_pt[0], img_start_pt[1] + int(np.sign(img_end_pt[1] - img_start_pt[1])))
            if img_end_pt in point_img_coord_set:
                if line_direction == 0:
                    img_end_pt = (img_end_pt[0] + int(np.sign(img_start_pt[0] - img_end_pt[0])), img_end_pt[1])
                else:
                    img_end_pt = (img_end_pt[0], img_end_pt[1] + int(np.sign(img_start_pt[1] - img_end_pt[1])))

            line_item['img_start_pt'] = img_start_pt
            line_item['img_end_pt'] = img_end_pt
            cv2.line(img, img_start_pt, img_end_pt, (0, 255, 255))
            cv2.putText(img, line_item['is'], (img_start_pt[0], img_start_pt[1] - 5), cv2.FONT_HERSHEY_PLAIN, 0.3,
                        (255, 255, 255))

        data['point_dict'] = point_dict

        if draw_img:
            return data, img
        else:
            return data

    def _draw_corner_with_scaling(self, img, corner, min_x, width, min_y, height, color=(0, 0, 255), text=None):
        img_x = int(math.floor((corner[0] - min_x) * 1.0 / width * self.im_size))
        img_y = int(math.floor((corner[1] - min_y) * 1.0 / height * self.im_size))
        cv2.circle(img, (img_x, img_y), 2, color, -1)
        if text is not None:
            cv2.putText(img, text, (img_x, img_y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, 1)
        return img_x, img_y


if __name__ == '__main__':
    base_dir = '/local-scratch/cjc/ICCV19/FloorPlotter/data/lianjia_500/processed'
    extra_test_dir = '/local-scratch/cjc/ICCV19/FloorPlotter/data/extra_test/processed'
    record_writer = RecordWriter(num_points=50000, base_dir=base_dir, phase='train', im_size=256, save_prefix='Lianjia_nonM', allow_non_man=True)
    record_writer.write()
    record_writer_test = RecordWriter(num_points=50000, base_dir=base_dir, phase='test', im_size=256,
                                      save_prefix='Lianjia_nonM', allow_non_man=True, extra_test=extra_test_dir)
    record_writer_test.write()
