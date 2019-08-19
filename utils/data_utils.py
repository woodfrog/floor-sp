import os
import os.path as osp
import sys
import numpy as np
import cv2
from plyfile import PlyData, PlyElement
import random
import json
import math
import copy
import pdb

ANCHOR_X = -1.64057
ANCHOR_Y = -3.2036
VOXEL_NUMBER_X = 550  # total number of voxels in X direction, every voxel is a square box
VOXEL_NUMBER_Y = 816
VOXEL_SIZE_X = 6.6  # total length in X direction(number of square voxels)
VOXEL_SIZE_Y = 9.792

TOP_DOWN_VIEW_PATH = './demo_nonm/map/point_evidence_visualize_0.jpg'

EXTRINSICS_PATH = './demo_nonm/final_extrinsics.txt'

IMAGE_SIZE = 256

ANNOT_OFFSET = 37500
ANNOT_SCALE = 1000


def get_extrinsics(extrinsic_path):
    with open(extrinsic_path) as f:
        lines = [line.strip() for line in f.readlines()]

    num_locations = int(lines[0])
    lines = lines[1:]

    lines = [line.split() for line in lines]
    extrinsics = list()

    for i in range(num_locations):
        mat_list = [[float(num) for num in line] for line in lines[i * 4:i * 4 + 4]]
        mat = np.array(mat_list)
        extrinsics.append(mat)

    return extrinsics


def visualize_camera_center(extrinsics, output_path):
    top_down_im = cv2.imread(TOP_DOWN_VIEW_PATH)
    global_origin_3d_all = list()
    num_locations = len(extrinsics)

    for camera_i in range(num_locations):
        extrinsic = extrinsics[camera_i]
        # extrinsic = np.linalg.inv(extrinsic)  # get the inverse matrix, for converting from local to global
        global_origin_3d = extrinsic[:3, 3]
        pix_x = int((global_origin_3d[0] - ANCHOR_X) / VOXEL_SIZE_X * VOXEL_NUMBER_X)
        pix_y = int((global_origin_3d[2] - ANCHOR_Y) / VOXEL_SIZE_Y * VOXEL_NUMBER_Y)
        # print(pix_x, pix_y)
        global_origin_3d_all.append(global_origin_3d)
        cv2.circle(top_down_im, (pix_y, pix_x), radius=5, color=(255, 0, 0), thickness=5)

    cv2.imwrite(osp.join(output_path, 'top_down_camera_locations.jpg'), top_down_im)

    # with open('camera_centres.txt', 'w') as f:
    # 	for coord in global_origin_3d_all:
    # 		coord_s = [str(num) for num in coord]
    # 		f.write(' '.join(coord_s) + '\n')


def pc_transfrom(read_file, pc_file, trans_matrix, downsample=True, write_output=False, out_path=None):
    if read_file:
        with open(pc_file, 'rb') as f:
            plydata = PlyData.read(f)
            dtype = plydata['vertex'].data.dtype
        print('dtype: {}'.format(dtype))

        data = np.array(plydata['vertex'].data.tolist())
    else:
        data = pc_file
        assert isinstance(data, np.ndarray)
        data = np.array(data.tolist())

    xyz = data[:, :3]
    xyz = np.concatenate([xyz, np.ones([xyz.shape[0], 1])], axis=1)

    transformed_xyz = np.matmul(trans_matrix, xyz.transpose([1, 0])).transpose([1, 0])
    transformed_xyz = transformed_xyz[:, :3]

    normal_trans_mat = trans_matrix.copy()
    normal_trans_mat[:3, 3] = 0
    surface_normal = data[:, 6:9]
    surface_normal = np.concatenate([surface_normal, np.ones([xyz.shape[0], 1])], axis=1)
    transformed_sn = np.matmul(normal_trans_mat, surface_normal.transpose([1, 0])).transpose([1, 0])
    transformed_sn = transformed_sn[:, :3]

    new_data = np.concatenate([transformed_xyz, data[:, 3:6], transformed_sn], axis=1)

    if downsample:
        new_data = [new_data[i] for i in range(len(new_data)) if random.random() > 0.5]

    vertex = np.array([tuple(x) for x in new_data],
                      dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('red', 'u1'), ('green', 'u1'), ('blue', 'u1'),
                             ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4')])
    vertex_el = PlyElement.describe(vertex, 'vertex')

    if write_output:
        assert out_path
        PlyData([vertex_el]).write(out_path)  # write the new ply file

    return vertex


def merge_point_clouds(base_dir, out_path=None, adjustment=False):
    print('merging point clouds for dir {}'.format(base_dir))
    extrinsics_path = osp.join(base_dir, 'final_extrinsics.txt')
    extrinsics = get_extrinsics(extrinsics_path)
    all_pc = list()
    for location_i, extrinsic in enumerate(extrinsics):
        ply_path = osp.join(base_dir, 'derived/{}/output.ply'.format(location_i))
        transformed_pc = pc_transfrom(read_file=True, pc_file=ply_path, trans_matrix=extrinsic)
        all_pc.append(transformed_pc)

    all_pc = np.concatenate(all_pc, axis=0)

    adjustment_trans_mat = np.array([[1, 0, 0, 0],
                                     [0, 0, 1, 0],
                                     [0, -1, 0, 0],
                                     [0, 0, 0, 1]])

    all_pc = pc_transfrom(read_file=False, pc_file=all_pc, trans_matrix=adjustment_trans_mat, downsample=False)

    vertex_el = PlyElement.describe(all_pc, 'vertex')

    if not out_path:
        out_path = osp.join(base_dir, 'output_all.ply')
    PlyData([vertex_el]).write(out_path)  # write the new ply file
    print('write merged ply file to {}'.format(out_path))


def process_annot(dir_path, out_path):
    dir_name = osp.basename(dir_path)
    file_path = osp.join(dir_path, dir_name + '.json')
    with open(file_path, 'r') as f:
        data = json.load(f)[0]

    points = data['points']
    lines = data['lines']
    line_items = data['lineItems']
    areas = data['areas']

    for point in points:
        point['x'] = (point['x'] - ANNOT_OFFSET) * 1.0 / ANNOT_SCALE
        point['y'] = (point['y'] - ANNOT_OFFSET) * 1.0 / ANNOT_SCALE

    for line_item in line_items:
        start_pt, end_pt = line_item['startPointAt'], line_item['endPointAt']
        start_pt['x'] = (start_pt['x'] - ANNOT_OFFSET) * 1.0 / ANNOT_SCALE
        start_pt['y'] = (start_pt['y'] - ANNOT_OFFSET) * 1.0 / ANNOT_SCALE
        end_pt['x'] = (end_pt['x'] - ANNOT_OFFSET) * 1.0 / ANNOT_SCALE
        end_pt['y'] = (end_pt['y'] - ANNOT_OFFSET) * 1.0 / ANNOT_SCALE

    img = visualize_annot(data)
    cv2.imwrite('./test.png', img)

    # ---------------------------------
    # 1. remove duplicate corners
    # ---------------------------------
    new_points = copy.deepcopy(points)
    point_dict = dict()
    removed_pt_ids = list()
    dup_pt_ids = list()
    for point in new_points:
        pt = (point['x'], point['y'])
        if len(point_dict) > 0:
            find_dup = False
            for other_pt in point_dict.keys():
                if pt_dist(pt, other_pt) <= 0.1:
                    print('handle duplicate pair')
                    removed_pt_ids.append(point['id'])
                    dup_pt_ids.append(point_dict[other_pt]['id'])
                    find_dup = True
                    break
            if not find_dup:
                point_dict[pt] = point
        else:
            point_dict[pt] = point

    if len(removed_pt_ids) > 0:
        new_points = [pt for key, pt in point_dict.items()]

        assert len(removed_pt_ids) == len(dup_pt_ids)
        for removed_id, dup_id in zip(removed_pt_ids, dup_pt_ids):
            for line in lines:
                if removed_id in line['points']:
                    idx = line['points'].index(removed_id)
                    line['points'][idx] = dup_id
        points = new_points  # update the points

    data['points'] = new_points
    img = visualize_annot(data)
    cv2.imwrite('./test_no_dup.png', img)

    # ------------------------------------------
    # 2. removing redundant edges
    # prepare adjacency list
    # -------------------------------------------
    point_dict = dict()
    adjacency_dict = dict()
    for point in new_points:
        adjacency_dict[point['id']] = set()
        point_dict[point['id']] = point

    for line in lines:
        pt1, pt2 = line['points']
        adjacency_dict[pt1].add(pt2)
        adjacency_dict[pt2].add(pt1)

    redundant_pt_ids = list()
    new_lines = copy.deepcopy(lines)

    for point in new_points:
        other_points = list(adjacency_dict[point['id']])
        if len(other_points) == 2:
            directions = list()

            for other_pt in other_points:
                directions.append(get_direction((point['x'], point['y']),
                                                (point_dict[other_pt]['x'], point_dict[other_pt]['y'])))
            if 'D' in directions and 'U' in directions or 'L' in directions and 'R' in directions:
                print('find strange pt')
                redundant_pt_ids.append(point['id'])
                removed_line_ids = list()
                for line in new_lines:
                    if line['id'] in removed_line_ids:
                        continue
                    if point['id'] in line['points']:
                        removed_line_ids.append(line['id'])
                new_lines = [line for line in new_lines if line['id'] not in removed_line_ids]

                # delete this point from graph
                for other_pt in other_points:
                    adjacency_dict[other_pt].remove(point['id'])
                del adjacency_dict[point['id']]
                del point_dict[point['id']]
                new_line = {'points': other_points, 'curve': 0, 'type': 0, 'id': removed_line_ids[0], 'items': list()}
                new_lines.append(new_line)
                adjacency_dict[other_points[0]].add(other_points[1])
                adjacency_dict[other_points[1]].add(other_points[0])

    if len(redundant_pt_ids) > 0:
        new_points = [point for point in points if point['id'] not in redundant_pt_ids]
        # img = visualize_annot(new_data)
        # cv2.imwrite('./test_no_redundant.png', img)

    # -------------------------------
    # 3. Further cleaning:  remove 1-degree points, finally clean areas,
    # -------------------------------

    # iteratively remove one-degree corner, not enabling this for now, just keep those dangling point
    # while True:
    #     one_degree_pt_ids = list()
    #     for point in new_points:
    #         if len(adjacency_dict[point['id']]) == 1:
    #             one_degree_pt_ids.append(point['id'])
    #             other_pt = list(adjacency_dict[point['id']])[0]
    #             adjacency_dict[other_pt].remove(point['id'])
    #             del adjacency_dict[point['id']]
    #             removed_line_ids = list()
    #             for line in new_lines:
    #                 if line['id'] in removed_line_ids:
    #                     continue
    #                 if point['id'] in line['points']:
    #                     removed_line_ids.append(line['id'])
    #             new_lines = [line for line in new_lines if line['id'] not in removed_line_ids]
    #     if len(one_degree_pt_ids) > 0:
    #         new_points = [point for point in new_points if point['id'] not in one_degree_pt_ids]
    #     else:
    #         break

    all_point_ids = [point['id'] for point in new_points]
    for idx, area in enumerate(areas):
        removed_ids = list()
        for point_id in area['points']:
            if point_id not in all_point_ids:
                removed_ids.append(point_id)
        for removed_id in removed_ids:
            areas[idx]['points'].remove(removed_id)
    # all room should have at least 4 corners (a simplest rectangle)
    areas = [area for area in areas if len(area['points']) >= 4]

    data = {
        'points': new_points,
        'lines': new_lines,
        'lineItems': line_items,
        'areas': areas,
    }

    img = visualize_annot(data)
    cv2.imwrite('./test_final.png', img)

    with open(out_path, 'w') as f:
        json.dump(data, f)

    print('write processed annotation into {}'.format(out_path))


def visualize_annot(annot):
    data = copy.deepcopy(annot)
    points = data['points']
    lines = data['lines']
    line_items = data['lineItems']

    point_dict = dict()
    all_x = list()
    all_y = list()
    for point in points:
        point_dict[point['id']] = point
        all_x.append(point['x'])
        all_y.append(point['y'])

    img = np.zeros([IMAGE_SIZE, IMAGE_SIZE, 3], dtype=np.uint8)

    min_x = min(all_x)
    min_y = min(all_y)
    width = height = max(max(all_x) - min(all_x), max(all_y) - min(all_y)) * 1.2

    # draw all corners
    for point in points:
        img_x, img_y = draw_corner_with_scaling(img, (point['x'], point['y']), min_x, width, min_y, height,
                                                text=None)
        point_dict[point['id']]['img_x'] = img_x
        point_dict[point['id']]['img_y'] = img_y

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
        img_start_pt = draw_corner_with_scaling(img, start_pt, min_x, width, min_y, height, color=(0, 255, 0))
        img_end_pt = draw_corner_with_scaling(img, end_pt, min_x, width, min_y, height, color=(0, 255, 0))
        line_item['img_start_pt'] = img_start_pt
        line_item['img_end_pt'] = img_end_pt
        cv2.line(img, img_start_pt, img_end_pt, (0, 255, 255))
        cv2.putText(img, line_item['is'], (img_start_pt[0], img_start_pt[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.3,
                    (255, 255, 255), 1)

    return img


def draw_corner_with_scaling(img, corner, min_x, width, min_y, height, color=(0, 0, 255), text=None):
    img_x = int(math.floor((corner[0] - min_x) * 1.0 / width * IMAGE_SIZE))
    img_y = int(math.floor((corner[1] - min_y) * 1.0 / height * IMAGE_SIZE))
    cv2.circle(img, (img_x, img_y), 2, color, -1)
    if text is not None:
        cv2.putText(img, text, (img_x, img_y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.2, (0, 0, 255), 1, 1)
    return img_x, img_y


def pt_dist(pt1, pt2):
    return math.sqrt((pt1[0] - pt2[0]) ** 2 + (pt1[1] - pt2[1]) ** 2)


def get_direction(pt_xy, other_xy, EPSILON=0.2):
    if abs(other_xy[0] - pt_xy[0]) <= EPSILON and other_xy[1] > pt_xy[1]:
        direction = 'D'
    elif abs(other_xy[0] - pt_xy[0]) <= EPSILON and other_xy[1] < pt_xy[1]:
        direction = 'U'
    elif other_xy[0] > pt_xy[0] and abs(other_xy[1] - pt_xy[1]) <= EPSILON:
        direction = 'R'
    elif other_xy[0] < pt_xy[0] and abs(other_xy[1] - pt_xy[1]) <= EPSILON:
        direction = 'L'
    else:
        direction = 'S'
    return direction


def dump_log(base_dir, error_list):
    log_path = osp.join(base_dir, 'log.txt')
    with open(log_path, 'w') as f:
        for error_name in error_list:
            f.write('error encountered in processing {} \n'.format(error_name))


if __name__ == '__main__':
    # Validate the given data by visualizing camera centers and transform point clouds 
    # extrinsics = get_extrinsics(EXTRINSICS_PATH)
    # visualize_camera_center(extrinsics, './')
    # for i, extrinsic_mat in enumerate(extrinsics):
    	# input_path = './demo/derived/{}/output.ply'.format(i)
    	# target_path = './demo/{}.ply'.format(i)
    	# pc_transfrom(input_path, extrinsic_mat, target_path)

    # Pre-processing: generate global point clouds + parse annotations (with some simple cleaning on the annotations)
    base_dir = '/local-scratch/cjc/Lianjia-inverse-cad/FloorNet/data/nonm_500/raw'
    out_base = '/local-scratch/cjc/Lianjia-inverse-cad/FloorNet/data/nonm_500/processed'
    out_base_ply = osp.join(out_base, 'ply')
    out_base_json = osp.join(out_base, 'json')

    error_list = list()
    for dir_name in os.listdir(base_dir):
        file_path = osp.join(base_dir, dir_name)
        filename_no_ext, _ = osp.splitext(dir_name)
        out_path_ply = osp.join(out_base_ply, filename_no_ext + '.ply')
        out_path_json = osp.join(out_base_json, filename_no_ext + '.json')
        annot_path = osp.join(file_path, dir_name + '.json')
        if not os.path.exists(annot_path):
            print('{} does not contain annotation!'.format(dir_name))
            continue
        if os.path.exists(out_path_ply):
            print('skip {}, it exists already'.format(dir_name))
            continue
        merge_point_clouds(file_path, out_path_ply)
        process_annot(file_path, out_path_json)

    dump_log(base_dir=out_base, error_list=error_list)
