import _init_paths
import os
import numpy as np
import skimage
from torch.utils.data import Dataset, DataLoader
from utils.data_utils import generate_corner_annot, augment_rotation, generate_edge_annot

import pdb


class LianjiaCornerDataset(Dataset):
    def __init__(self, data_dir, phase, augmentation=''):
        super(LianjiaCornerDataset, self).__init__()
        if phase not in ['train', 'test']:
            raise ValueError('invalid phase {}'.format(phase))
        self.phase = phase
        self.base_dir = os.path.join(data_dir, phase)
        self.augmentation = augmentation
        if phase == 'test':
            self.augmentation = ''

        self._data_path_list = list()
        for filename in sorted(os.listdir(self.base_dir)):
            file_path = os.path.join(self.base_dir, filename)
            self._data_path_list.append(file_path)

        # set image size, will be used for normalization
        sample = np.load(self._data_path_list[0], encoding='latin1').tolist()
        self.im_size = sample['topview_image'].shape[0]

    def __len__(self):
        return len(self._data_path_list)

    def __getitem__(self, index):
        sample_path = self._data_path_list[index]
        with open(sample_path, 'rb') as f:
            sample_data = np.load(f, encoding='latin1').tolist()
        image = np.transpose(sample_data['topview_image'], [2, 0, 1])
        mean_normal = np.transpose(sample_data['mean_normal'], [2, 0, 1])
        room_map = sample_data['room_map']
        room_masks_map = sample_data['room_mask_pred']
        room_annots = sample_data['room_annot']

        # todo: note that the normalization here is for natural image, might be different for our input?

        all_room_corners = [room_annot['room_corners'] for room_annot in room_annots]

        if 'r' in self.augmentation:
            corner_to_id = dict([((x['img_x'], x['img_y']), y) for y, x in sample_data['point_dict'].items()])

            for room_i in range(len(all_room_corners)):
                room_corners_id = [corner_to_id[x] for x in all_room_corners[room_i]]
                all_room_corners[room_i] = room_corners_id

            image, mean_normal, room_masks_map, point_dict = augment_rotation(image, mean_normal, room_masks_map,
                                                                              sample_data['point_dict'])
            sample_data['point_dict'] = point_dict

            for room_i in range(len(all_room_corners)):
                room_corners_id = all_room_corners[room_i]
                rot_room_corners = [(point_dict[x]['img_x'], point_dict[x]['img_y']) for x in room_corners_id]
                all_room_corners[room_i] = rot_room_corners

        normalized_image = skimage.img_as_float(image[0])  # only keep one channel since they are highly overlapped
        corner_annot, corner_gt_map = generate_corner_annot(sample_data['point_dict'], sample_data['lines'])
        edge_gt_map = generate_edge_annot(all_room_corners, self.im_size)

        # for testing the augmentation
        # from scipy.misc import imsave
        # import cv2
        # test_img = image.transpose([1, 2, 0]).astype(np.float32) + (np.stack([corner_gt_map[0]] * 3, -1) * 255).astype(np.float32)
        # test_img = np.clip(test_img, 0, 255).astype(np.uint8)
        # for corner_id, corner_data in corner_annot.items():
        #     if 0 < corner_data['x'] < 256 and 0 < corner_data['y'] < 256:
        #         for connection in corner_data['connections']:
        #             vec_x = np.cos(connection * 10 / 180 * np.pi)
        #             vec_y = np.sin(connection * 10 / 180 * np.pi)
        #             end_x = int(np.round(corner_data['x'] + 10 * vec_x))
        #             end_y = int(np.round(corner_data['y'] - 10 * vec_y))
        #             if 0 < end_x < 256 and 0 < end_y < 256:
        #                 cv2.line(test_img, (corner_data['x'], corner_data['y']), (end_x, end_y), (255, 0, 0), 1)
        # imsave('./test/test_{}_rot.png'.format(index), test_img)
        # imsave('./test/test_{}_rot_mask.png'.format(index), room_masks_map)

        data = {
            'image_non_normalized': image,
            'image': normalized_image,
            'mean_normal': mean_normal,
            'room_masks_map': room_masks_map,
            'room_map': room_map,
            'corner_gt_map': corner_gt_map,
            'edge_gt_map': edge_gt_map
        }

        return data

    def normalize(self, image):
        image = image / 255.0
        return image


if __name__ == '__main__':
    dataset = LianjiaCornerDataset(
        data_dir='/local-scratch/cjc/Lianjia-inverse-cad/FloorPlotter/data/Lianjia_corner', phase='train', augmentation='r')

    # currently only support batch_size = 1
    data_loader = DataLoader(dataset, batch_size=1)

    for idx, batch_data in enumerate(data_loader):
        pdb.set_trace()
