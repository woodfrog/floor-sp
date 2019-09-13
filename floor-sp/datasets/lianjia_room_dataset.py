import _init_paths
import os
import numpy as np
import skimage
from torch.utils.data import Dataset, DataLoader
from utils.data_utils import get_room_gt_map, get_room_edge_map, augment_rotation

import pdb


class LianjiaRoomDataset(Dataset):
    def __init__(self, data_dir, phase, augmentation=''):
        super(LianjiaRoomDataset, self).__init__()
        if phase not in ['train', 'test']:
            raise ValueError('invalid phase {}'.format(phase))
        self.phase = phase
        self.base_dir = os.path.join(data_dir, phase)
        self.augmentation = augmentation

        # pre-processing to build the mapping from room index --> file path + index in the indoor scene sample
        self.index_mapping = dict()
        if self.phase == 'test':
            self.room_contours = list()
        room_idx = 0
        for filename in sorted(os.listdir(self.base_dir)):
            file_path = os.path.join(self.base_dir, filename)

            with open(file_path, 'rb') as f:
                sample_data = np.load(f, encoding='latin1').tolist()
            room_data = sample_data['room_data']
            num_room = len(room_data)
            for i in range(room_idx, room_idx + num_room):
                self.index_mapping[i] = {'path': file_path,
                                         'room_index': i - room_idx,
                                         'room_num': num_room}
                if self.phase == 'test':
                    self.room_contours.append(room_data[i - room_idx]['contour'])
            room_idx += num_room

        # set image size, will be used for normalization
        sample = np.load(self.index_mapping[0]['path'], encoding='latin1').tolist()
        self.im_size = sample['room_data'][0]['topview_image'].shape[0]  # pick the first room instance in the sample

    def __len__(self):
        return len(self.index_mapping)

    def __getitem__(self, index):
        sample_path = self.index_mapping[index]['path']
        with open(sample_path, 'rb') as f:
            sample_data = np.load(f, encoding='latin1').tolist()
        room_data = sample_data['room_data'][self.index_mapping[index]['room_index']]

        image = np.transpose(room_data['topview_image'], [2, 0, 1])
        mean_normal = np.transpose(room_data['mean_normal'], [2, 0, 1])
        pred_room_mask = room_data['pred_room_mask'].astype(np.float64)
        room_corners = room_data['room_corners']

        if self.phase == 'train':
            assert room_corners is not None

        normalized_image = skimage.img_as_float(image[0])  # only keep one channel since they are highly overlapped

        if 'r' in self.augmentation:
            corner_to_id = dict([((x['img_x'], x['img_y']), y) for y, x in sample_data['point_dict'].items()])
            room_corners_id = [corner_to_id[x] for x in room_corners]
            image, mean_normal, pred_room_mask, point_dict = augment_rotation(image, mean_normal, pred_room_mask,
                                                                              sample_data['point_dict'])
            sample_data['point_dict'] = point_dict
            rot_room_corners = [(point_dict[x]['img_x'], point_dict[x]['img_y']) for x in room_corners_id]
            room_corners = rot_room_corners

        if self.phase == 'train':
            room_edge_map = get_room_edge_map(room_corners, image.shape[1])
            data = {
                'image_non_normalized': image,
                'image': normalized_image,
                'mean_normal': mean_normal,
                'room_edge_map': room_edge_map,
                'pred_room_mask': pred_room_mask,
            }
        else:
            data = {
                'image_non_normalized': image,
                'image': normalized_image,
                'mean_normal': mean_normal,
                'pred_room_mask': pred_room_mask,
            }

        return data

    def normalize(self, image):
        image = image / 255.0
        return image


if __name__ == '__main__':
    dataset = LianjiaRoomDataset(data_dir='/local-scratch/cjc/floor-sp/floor-sp/data/Lianjia_room',
                                 phase='train')

    data_loader = DataLoader(dataset, batch_size=4)

    for idx, batch_data in enumerate(data_loader):
        pdb.set_trace()
