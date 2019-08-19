import _init_paths
import os
import numpy as np
import skimage
from torch.utils.data import Dataset, DataLoader
from utils.data_utils import compute_corner_bins, get_single_corner_map, compute_bin_idx, get_edge_map
from utils.data_utils import corner_random_jittering, connections_random_dropping

import pdb


class LianjiaAffiliationDataset(Dataset):
    def __init__(self, data_dir, phase, mode):
        super(LianjiaAffiliationDataset, self).__init__()
        if phase not in ['train', 'test']:
            raise ValueError('invalid phase {}'.format(phase))
        self.phase = phase
        self.mode = mode
        self.base_dir = os.path.join(data_dir, phase)
        self.num_bins = 36

        if self.mode not in ['room_corner', 'corner_corner']:
            raise ValueError('Invalid mode {}'.format(self.mode))

        # pre-processing to build the mapping from room index --> file path + index in the indoor scene sample
        self.index_mapping = dict()
        self.corner_binnings = list()
        self.binning_maps = list()
        affiliation_idx = 0
        for sample_idx, filename in enumerate(sorted(os.listdir(self.base_dir))):
            file_path = os.path.join(self.base_dir, filename)

            with open(file_path, 'rb') as f:
                sample_data = np.load(f, encoding='latin1').tolist()
            room_data = sample_data['room_data']
            all_corners = sample_data['all_corners']
            point_dict = sample_data['point_dict']
            lines = sample_data['lines']

            selected_pairs = list()
            all_non_pairs = list()

            if self.mode == 'room_corner':
                for room_idx, room_info in enumerate(room_data):
                    if room_info['gt_room_mask'] is None:
                        continue
                    for corner_idx, corner in enumerate(room_info['room_corners']):
                        next_idx = 0 if corner_idx == len(room_info['room_corners']) - 1 else corner_idx + 1
                        adjacent_corners = (
                        room_info['room_corners'][corner_idx - 1], room_info['room_corners'][next_idx])
                        selected_pairs.append((room_idx, corner, True, adjacent_corners))
                    for corner in all_corners:
                        if corner not in room_info['room_corners']:
                            all_non_pairs.append((room_idx, corner, False, None))

                sampled_indices = np.random.choice(np.array([x for x in range(len(all_non_pairs))]),
                                                   size=len(selected_pairs))

                for idx in sampled_indices:
                    selected_pairs.append(all_non_pairs[idx])

                for i in range(affiliation_idx, affiliation_idx + len(selected_pairs)):
                    r_c_pair = selected_pairs[i - affiliation_idx]
                    self.index_mapping[i] = {'sample_index': sample_idx,
                                             'path': file_path,
                                             'room_index': r_c_pair[0],
                                             'corner': r_c_pair[1],
                                             'label': r_c_pair[2],
                                             'adjacent_corners': r_c_pair[3]}
                affiliation_idx += len(selected_pairs)

            elif self.mode == 'corner_corner':
                for room_idx, room_info in enumerate(room_data):
                    if room_info['gt_room_mask'] is None:
                        continue
                    
                    for corner_idx, corner in enumerate(room_info['room_corners']):
                        next_idx = 0 if corner_idx == len(room_info['room_corners']) - 1 else corner_idx + 1
                        adjacent_corners = (room_info['room_corners'][corner_idx - 1], room_info['room_corners'][next_idx])
                        next_corner = adjacent_corners[1]
                        # every room provides us with N_ROOM positive pairs along the room chain
                        selected_pairs.append((room_idx, (corner, next_corner), True))
                        for other_corner in room_info['room_corners']:
                            if corner == other_corner or other_corner in adjacent_corners:
                                continue
                            else:
                                all_non_pairs.append((room_idx, (corner, other_corner), False))

                sampled_indices = np.random.choice(np.array([x for x in range(len(all_non_pairs))]),
                                                   size=len(selected_pairs))
                for idx in sampled_indices:
                    selected_pairs.append(all_non_pairs[idx])

                for i in range(affiliation_idx, affiliation_idx + len(selected_pairs)):
                    r_c_pair = selected_pairs[i - affiliation_idx]
                    self.index_mapping[i] = {'sample_index': sample_idx,
                                             'path': file_path,
                                             'room_index': r_c_pair[0],
                                             'corner_pair': r_c_pair[1],
                                             'label': r_c_pair[2]}
                affiliation_idx += len(selected_pairs)

            corner_binnings, corner_to_bin = self._compute_corner_binnings(point_dict, lines)
            self.corner_binnings.append(corner_binnings)
            self.binning_maps.append(corner_to_bin)

        # set image size, will be used for normalization
        sample = np.load(self.index_mapping[0]['path'], encoding='latin1').tolist()
        self.im_size = sample['room_data'][0]['topview_image'].shape[0]  # pick the first room instance in the sample

    def __len__(self):
        return len(self.index_mapping)

    def __getitem__(self, index):
        if self.mode == 'room_corner':
            return self._get_item_corner_room(index)
        elif self.mode == 'corner_corner':
            return self._get_item_corner_corner(index)
        else:
            raise ValueError('wrong mode {}'.format(self.mode))

    def _get_item_corner_room(self, index):
        """
        Every item is one combination of corner and room. One datasample can provide hundred of
        items actually. So we randomly pick 50 samples from evey scene, 400 * 50 = 20,000.... too many
        :param index:
        :return:
        """
        sample_path = self.index_mapping[index]['path']
        sample_index = self.index_mapping[index]['sample_index']
        with open(sample_path, 'rb') as f:
            sample_data = np.load(f, encoding='latin1').tolist()
        room_data = sample_data['room_data']
        all_corners = sample_data['all_corners']
        room_instance = room_data[self.index_mapping[index]['room_index']]
        corner_instance = self.index_mapping[index]['corner']
        label = self.index_mapping[index]['label']
        adjacent_corners = self.index_mapping[index]['adjacent_corners']
        corner_connections = self.corner_binnings[sample_index]
        bin_map = self.binning_maps[sample_index]

        edge_dirs_label = self._compute_room_corner_dir(corner_instance, adjacent_corners, bin_map)

        if self.phase == 'training':
            jittered_corner_instance = corner_random_jittering(corner_instance)
            dropped_corner_connections = connections_random_dropping(corner_connections)
        else:
            jittered_corner_instance = corner_instance
            dropped_corner_connections = corner_connections

        image = np.transpose(room_instance['topview_image'], [2, 0, 1])[
            0]  # only pick one channel, since it's a density image
        image = self.normalize(image)

        mean_normal = np.transpose(room_instance['mean_normal'], [2, 0, 1])
        pred_room_mask = room_instance['pred_room_mask'].astype(np.float64)
        corner_map = get_single_corner_map(jittered_corner_instance, dropped_corner_connections[corner_instance], self.im_size)

        label = np.concatenate([np.array([label], dtype=np.float64), edge_dirs_label], axis=0)

        data = {
            'image': image,
            'mean_normal': mean_normal,
            'room_mask': pred_room_mask,
            'corner_map': corner_map,
            'label': label,
        }

        return data

    def _get_item_corner_corner(self, index):
        sample_path = self.index_mapping[index]['path']
        sample_index = self.index_mapping[index]['sample_index']
        with open(sample_path, 'rb') as f:
            sample_data = np.load(f, encoding='latin1').tolist()
        room_data = sample_data['room_data']
        room_instance = room_data[self.index_mapping[index]['room_index']]
        corner_pair = self.index_mapping[index]['corner_pair']
        label = self.index_mapping[index]['label']
        corner_connections = self.corner_binnings[sample_index]

        # add some random noise for robustness
        if self.phase == 'training':
            jittered_corner_pair = (corner_random_jittering(corner_pair[0]), corner_random_jittering(corner_pair[1]))
            dropped_corner_connections = connections_random_dropping(corner_connections)
        else:
            jittered_corner_pair = corner_pair
            dropped_corner_connections = corner_connections

        image = np.transpose(room_instance['topview_image'], [2, 0, 1])[
            0]  # only pick one channel, since it's a density image
        image = self.normalize(image)

        mean_normal = np.transpose(room_instance['mean_normal'], [2, 0, 1])
        pred_room_mask = room_instance['pred_room_mask'].astype(np.float64)
        corner_map_1 = get_single_corner_map(jittered_corner_pair[0], dropped_corner_connections[corner_pair[0]], self.im_size)
        corner_map_2 = get_single_corner_map(jittered_corner_pair[1], dropped_corner_connections[corner_pair[1]], self.im_size)
        corners_map = np.clip(corner_map_1 + corner_map_2, 0, 1)
        edge_map = get_edge_map(jittered_corner_pair[0], jittered_corner_pair[1], self.im_size)

        label = np.concatenate([np.array([label], dtype=np.float64)], axis=0)

        # # For debugging use
        # viz_img = np.clip(pred_room_mask + corners_map, 0, 1)
        # from scipy.misc import imsave
        # imsave('./debug/corner-corner/{}_corner_corner_viz_{}.png'.format(index, int(label[0])), viz_img)
        # imsave('./debug/corner-corner/{}_edge_map.png'.format(index), edge_map)
        
        data = {
            'image': image,
            'mean_normal': mean_normal,
            'room_mask': pred_room_mask,
            'corners_map': corners_map,
            'edge_map': edge_map,
            'label': label,
        }

        return data

    @staticmethod
    def normalize(image):
        image = image / 255.0
        return image

    def _compute_corner_binnings(self, point_dict, lines):
        corner_bins_dict = compute_corner_bins(point_dict, lines, self.num_bins)
        corner_binnings = dict()
        bin_map = dict()
        for _, corner_info in corner_bins_dict.items():
            corner_binnings[corner_info['x'], corner_info['y']] = corner_info['connections']
            bin_map[corner_info['x'], corner_info['y']] = corner_info['bin_map']
        return corner_binnings, bin_map

    def _compute_room_corner_dir(self, corner, adj_corners, bin_map):
        edge_dir_label = np.zeros([self.num_bins])
        if adj_corners is None:
            pass
        else:
            assert len(adj_corners) == 2
            corner_to_bin = bin_map[corner]
            for adj_corner in adj_corners:
                if adj_corner not in corner_to_bin:  # this is due to data glitch (inconsistency on room annots)
                    vec_to_adj = (adj_corner[0] - corner[0], adj_corner[1] - corner[1])
                    bin_idx = compute_bin_idx(vec_to_adj, self.num_bins)
                    edge_dir_label[bin_idx] = 1
                else:
                    edge_dir_label[corner_to_bin[adj_corner]] = 1
        return edge_dir_label


if __name__ == '__main__':
    dataset = LianjiaAffiliationDataset(data_dir='/local-scratch/cjc/Lianjia-inverse-cad/FloorPlotter/data/Lianjia_room',
                                        phase='train', mode='room_corner')

    # currently only support batch_size = 1
    data_loader = DataLoader(dataset, batch_size=1)

    for idx, batch_data in enumerate(data_loader):
        pdb.set_trace()
