import os.path as osp
from data_utils import get_extrinsics
import numpy as np
import cv2
import math
from plyfile import PlyData, PlyElement
import pdb

CAMERA_EXTRINSICS = './demo/camera_extrinsics.txt'
IMAEG_BASE_DIR = './demo/derived/0'
PANO_WIDTH = 1600
PANO_HEIGHT = 800

CAMERA_TOP_PARAMETERS = [519.08599854, 519.08599854, 300.32199097, 239.82499695]
CAMERA_MID_PARAETERS = [518.80999756, 518.80999756, 327.62799072, 252.43400574]
CAMERA_DOWN_PARAMETERS = [518.48101807, 518.48101807, 325.09298706, 245.67100525]
CAMERA_PARAMETERS = [CAMERA_TOP_PARAMETERS, CAMERA_MID_PARAETERS, CAMERA_DOWN_PARAMETERS]


def get_camera_intrinsics_mat(fx, fy, cx, cy):
	return np.array([[fx, 0, cx , 0], [0, fy, cy, 0], [0, 0, 1, 0], [0, 0, 0, 1]])


def pano_stitching(image_dir, camera_extrinsics):
	pano = np.zeros([PANO_HEIGHT, PANO_WIDTH, 3], dtype='uint8')
	count = np.zeros([PANO_HEIGHT, PANO_WIDTH])

	for i, extrinsic in enumerate(camera_extrinsics):
		extrinsic = np.linalg.inv(extrinsic)
		file_name = '{:02d}.jpg'.format(i)
		file_path = osp.join(image_dir, file_name)
		sub_im = cv2.imread(file_path)
		im_size = sub_im.shape
		print('processing image.{}'.format(i))

		pano_i = np.array([[j for _ in range(PANO_WIDTH)] for j in range(PANO_HEIGHT)])  # height direction
		pano_j = np.array([[i for i in range(PANO_WIDTH)] for _ in range(PANO_HEIGHT)])  # width direction

		longtitude = -pano_j * 1.0 / (PANO_WIDTH-1) * 2 * np.pi + np.pi / 2
		latitude = pano_i * 1.0 / (PANO_HEIGHT-1) * np.pi - np.pi / 2

		x_w = np.cos(latitude) * np.cos(longtitude) * 5
		z_w = np.cos(latitude) * np.sin(longtitude) * 5
		y_w = np.sin(latitude) * 5

		extra_ones = np.ones([PANO_HEIGHT, PANO_WIDTH])
		global_coords = np.stack([x_w, y_w, z_w, extra_ones], axis=2).transpose([2, 0, 1]).reshape([4, -1])

		# -------------------
		# For generating 3d points to visualize in Meshlab
		# ----
		# top_points = global_coords[:3, 50, 200:400].transpose([1, 0])
		# mid_points = global_coords[:3, 400, :300].transpose([1, 0])
		# bottom_points = global_coords[:3, 750, :].transpose([1, 0])

		# all_points = np.concatenate([top_points, mid_points, bottom_points], axis=0)

		# vertex = np.array([tuple(x) for x in all_points], dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])
		# vertex_el = PlyElement.describe(vertex, 'vertex')
		# PlyData([vertex_el]).write('./demo/derived/0/debug.ply')  # write the new ply file
		# -------------------

		camera_intrinsic = get_camera_intrinsics_mat(*CAMERA_PARAMETERS[i % 3])
		print('intrinsic index : {}'.format(i % 3))

		camera_matrix = np.matmul(camera_intrinsic, extrinsic)

		local_coords = np.matmul(camera_matrix, global_coords).reshape([4, PANO_HEIGHT, PANO_WIDTH]).transpose([1, 2, 0])

		positive_depth = np.nonzero(local_coords[:, :, 2] > 0)

		for valid_i, valid_j in zip(*positive_depth):	
			local_x = int(local_coords[valid_i][valid_j][0] / local_coords[valid_i][valid_j][2])
			local_y = int(local_coords[valid_i][valid_j][1] / local_coords[valid_i][valid_j][2])

			if 0 <= local_x < im_size[1] and 0 <= local_y < im_size[0]:
				pano[valid_i][valid_j] = sub_im[local_y][local_x]
			else:
				continue

		cv2.imwrite('./demo/derived/0/pano_step_{}.png'.format(i), pano)



if __name__ == '__main__':
	camera_extrinsics = get_extrinsics(CAMERA_EXTRINSICS)
	pano = pano_stitching(IMAEG_BASE_DIR, camera_extrinsics)