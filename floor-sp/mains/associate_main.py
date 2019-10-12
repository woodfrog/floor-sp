import _init_paths
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
from scipy.misc import imsave
from scipy.ndimage.morphology import binary_dilation
import numpy as np

from datasets.lianjia_affiliation_dataset import LianjiaAffiliationDataset
from models.corner_room_module import CornerRoomAssociate, CornerCornerAssociate
from trainers.associate_trainer import AssociateTrainer
from utils.misc import save_checkpoint, count_parameters, transfer_optimizer_to_gpu
from utils.config import Struct, load_config, compose_config_str
from utils.data_utils import get_single_corner_map, get_edge_map, get_direction_hist, get_room_heatmap
from utils.floorplan_utils.floorplan_misc import visualize_rooms_info


def train(configs):
    train_dataset = LianjiaAffiliationDataset(
        data_dir='/local-scratch/cjc/floor-sp/floor-sp/data/Lianjia_room',
        phase='train', mode=configs.mode)

    train_loader = DataLoader(train_dataset, batch_size=configs.batch_size, shuffle=True)

    if configs.mode == 'room_corner':
        model = CornerRoomAssociate(im_size=256, configs=configs)
    elif configs.mode == 'corner_corner':
        model = CornerCornerAssociate(im_size=256, configs=configs)
    else:
        raise ValueError('Invalid mode {}'.format(configs.mode))
    model.double()

    criterion = nn.BCEWithLogitsLoss(reduce=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=configs.lr, weight_decay=configs.decay_rate)
    scheduler = StepLR(optimizer, step_size=configs.lr_step, gamma=0.1)

    num_parameters = count_parameters(model)
    print('total number of trainable parameters is: {}'.format(num_parameters))

    trainer = AssociateTrainer(model=model, train_loader=train_loader, val_loader=None, criterion=criterion,
                               optimizer=optimizer, configs=configs)
    start_epoch = 0

    if configs.resume:
        if os.path.isfile(configs.model_path):
            print("=> loading checkpoint '{}'".format(configs.model_path))
            checkpoint = torch.load(configs.model_path)
            model.load_state_dict(checkpoint['state_dict'])
            start_epoch = checkpoint['epoch']
            optimizer.load_state_dict(checkpoint['optimizer'])
            if configs.use_cuda:
                transfer_optimizer_to_gpu(optimizer)
            print('=> loaded checkpoint {} (epoch {})'.format(configs.model_path, start_epoch))
        else:
            print('no checkpoint found at {}'.format(configs.model_path))

    if configs.use_cuda:
        model.cuda()
        criterion.cuda()

    ckpt_save_path = os.path.join(configs.exp_dir, 'checkpoints')
    if not os.path.exists(ckpt_save_path):
        os.makedirs(ckpt_save_path)

    for epoch_num in range(start_epoch, start_epoch + configs.num_epochs):
        scheduler.step()
        trainer.train_epoch(epoch_num)

        if epoch_num % configs.val_interval == 0:
            # valid_loss, valid_acc, _, _, _ = trainer.validate()

            # is_best = valid_acc > best_acc
            # best_acc = max(valid_acc, best_acc)
            save_checkpoint({
                'epoch': epoch_num + 1,
                # 'best_acc': best_acc,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict()
            }, is_best=False, checkpoint=ckpt_save_path,
                filename='checkpoint_associate_module_{}.pth.tar'.format(epoch_num))


def test(configs):
    test_dataset = LianjiaAffiliationDataset(
        data_dir='/local-scratch/cjc/floor-sp/floor-sp/data/Lianjia_room', phase='test', mode=configs.mode)
    # only support bs = 1 for testing
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    model = CornerRoomAssociate(im_size=256, configs=configs)
    model.double()

    if configs.model_path:
        checkpoint = torch.load(configs.model_path)
        model.load_state_dict(checkpoint['state_dict'])
        epoch_num = checkpoint['epoch']
        print('=> loaded checkpoint {} (epoch {})'.format(configs.model_path, epoch_num))

    viz_dir = os.path.join(configs.exp_dir, 'visualization')
    if not os.path.exists(viz_dir):
        os.makedirs(viz_dir)

    model.eval()

    N = 0
    n_correct = 0
    # disable gradient computation, we only feed forward-pass during inference
    with torch.no_grad():
        for idx, batch_data in enumerate(test_loader):
            image_inputs = batch_data['image']
            mean_normal = batch_data['mean_normal']
            room_mask = batch_data['room_mask']
            corner_map = batch_data['corner_map']
            label = batch_data['label']

            inputs = torch.cat(
                [image_inputs.unsqueeze(1), mean_normal, room_mask.unsqueeze(1), corner_map.unsqueeze(1)], dim=1)
            logits, preds = model(inputs)
            preds = preds.squeeze().cpu().numpy()

            image_inputs_np = image_inputs.cpu().numpy()[0]
            room_mask_np = room_mask.cpu().numpy()[0]
            corner_map = corner_map.cpu().numpy()[0]
            viz_image = np.clip(image_inputs_np * 255 + room_mask_np * 255 + corner_map * 255, 0, 255)

            result = 'true' if preds[0] >= 0.5 else 'false'

            N += 1
            if result == 'true' and label[0][0] == 1 or result == 'false' and label[0][0] == 0:
                n_correct += 1
                correct = 'correct'
            else:
                correct = 'wrong'

            save_path = os.path.join(viz_dir, '{}_associate_{}_{}.png'.format(idx, result, correct))
            imsave(save_path, viz_image)

            print('In {} pairs, {} preditions are correct, rate {}'.format(N, n_correct, 1.0 * n_correct / N))


def predict(configs):
    model_room_corner = CornerRoomAssociate(im_size=256, configs=configs)
    model_room_corner.double()

    if configs.model_path:
        checkpoint = torch.load(configs.model_path)
        model_room_corner.load_state_dict(checkpoint['state_dict'])
        epoch_num = checkpoint['epoch']
        print('=> loaded checkpoint {} (epoch {})'.format(configs.model_path, epoch_num))

    if configs.use_cuda:
        model_room_corner.cuda()

    viz_dir = os.path.join(configs.exp_dir, 'preds_visualization')
    save_dir = os.path.join(configs.exp_dir, 'processed_preds')
    if not os.path.exists(viz_dir):
        os.makedirs(viz_dir)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    model_room_corner.eval()

    # Test set
    data_dir = '/local-scratch/cjc/floor-sp/floor-sp/data/Lianjia_room/test'
    # Corner preds on the test set
    corner_preds_dir = '/local-scratch/cjc/floor-sp/floor-sp/results_corner/lr_0.0001_batch_size_4_augmentation_r_corner_edge/test_preds'

    assert len(os.listdir(data_dir)) == len(os.listdir(corner_preds_dir))
    predict_batch_size = 32

    # disable gradient computation, we only do forward-pass during inference
    with torch.no_grad():
        for idx, filename in enumerate(sorted(os.listdir(data_dir))):
            file_path = os.path.join(data_dir, filename)
            with open(file_path, 'rb') as f:
                data = np.load(f).tolist()

            corner_preds_path = os.path.join(corner_preds_dir, '{}_corner_preds.npy'.format(idx))
            with open(corner_preds_path, 'rb') as f:
                corner_preds = np.load(corner_preds_path).tolist()

            room_data = data['room_data']
            room_labels = data['class_names']
            vectorized_corner_preds = corner_preds['vectorized_preds']
            corner_preds_heatmap = corner_preds['corner_heatmap']
            edge_preds = corner_preds['edge_preds']
            edge_heatmap = edge_preds[0]
            global_direction_hist = corner_preds['direction_hist']

            vectorized_rooms_info = list()
            # iterate through rooms of the scene
            for room_idx, room_info in enumerate(room_data):
                image_inputs = room_info['topview_image'][:, :, 0].astype(
                    np.float64) / 255  # only get the first channel of the density image
                mean_normal = room_info['mean_normal']
                room_mask = room_info['pred_room_mask'].astype(np.float64)
                room_class_id = room_info['class_id']
                room_viz_color = room_info['viz_color']

                image_inputs = torch.from_numpy(image_inputs)
                mean_normal = torch.from_numpy(mean_normal.transpose([2, 0, 1]))
                room_mask = torch.from_numpy(room_mask)

                if configs.use_cuda:
                    image_inputs = image_inputs.cuda()
                    mean_normal = mean_normal.cuda()
                    room_mask = room_mask.cuda()

                room_corners = list()
                # read corner preds from dumped results of stage-1
                batched_inputs = list()
                batched_corner_idx = list()
                # iterate through global corner preds, do room_corner associoation
                for corner_idx, corner_pred in enumerate(vectorized_corner_preds):
                    corner_map = get_single_corner_map(corner_pred['corner'], corner_pred['edge_dirs'], configs.im_size)
                    corner_map = torch.from_numpy(corner_map)

                    if configs.use_cuda:
                        corner_map = corner_map.cuda()

                    inputs = torch.cat(
                        [image_inputs.unsqueeze(0).unsqueeze(1), mean_normal.unsqueeze(0),
                         room_mask.unsqueeze(0).unsqueeze(1), corner_map.unsqueeze(0).unsqueeze(1)], dim=1)

                    batched_inputs.append(inputs)
                    batched_corner_idx.append(corner_idx)

                    # we batch the corner-room pairs for acceleration
                    if len(batched_inputs) == predict_batch_size or corner_idx == len(vectorized_corner_preds) - 1:
                        inputs = torch.cat(batched_inputs, 0)
                        logits, preds = model_room_corner(inputs)
                        preds = preds.cpu().numpy()
                        for batch_i in range(preds.shape[0]):
                            if preds[batch_i][0] > 0.5:
                                print('Sample.{} find positive affiliation between room.{} and corner.{}'.format(idx,
                                                                                                                 room_idx,
                                                                                                                 batched_corner_idx[
                                                                                                                     batch_i]))
                                batch_corner_pred = vectorized_corner_preds[batched_corner_idx[batch_i]]
                                room_corner = {
                                    'corner': batch_corner_pred['corner'],
                                    'edge_dirs': batch_corner_pred['edge_dirs'],
                                    'binning': batch_corner_pred['binning'],
                                    'corner_conf': corner_preds_heatmap[
                                        batch_corner_pred['corner'][1], batch_corner_pred['corner'][0]],
                                }
                                room_corners.append(room_corner)

                        # clear the batch cache
                        batched_inputs = list()
                        batched_corner_idx = list()

                room_corner_heatmap = get_room_heatmap(room_corners, corner_preds_heatmap, mode='corner')

                mask_size = room_info['pred_room_mask'].sum()
                if mask_size < 2000:
                    expand_iter = 10
                else:
                    expand_iter = 20
                expanded_mask = binary_dilation(room_info['pred_room_mask'].astype(np.float32),
                                                iterations=expand_iter).astype(np.float32)
                # filter the edge map using expanded mask regions, rather than using bounding box
                room_edge_map = edge_heatmap * expanded_mask
                # compute the edge direction histogram for the current room, this will be used for computing room dominant direction as described in the paper
                room_edge_preds = edge_preds * np.expand_dims(expanded_mask, axis=0)
                room_direction_hist = get_direction_hist(room_edge_preds)

                room_details = {
                    'corners_info': room_corners,
                    'contour': room_info['contour'],
                    'mask': room_info['pred_room_mask'],
                    'edge_map': room_edge_map,
                    'room_corner_heatmap': room_corner_heatmap,
                    'room_direction_histogram': room_direction_hist,
                    'class_id': room_class_id,
                    'viz_color': room_viz_color,
                }
                vectorized_rooms_info.append(room_details)

            # refine rooms info, computing the source/end nodes according to corner conf
            room_imgs = visualize_rooms_info(vectorized_rooms_info)

            sample_data = {
                'rooms_info': vectorized_rooms_info,
                'all_corners_info': vectorized_corner_preds,
                'density_img': room_data[0]['topview_image'][:, :, 0].astype(
                    np.float64) / 255,
                'mean_normal': room_data[0]['mean_normal'],
                'direction_hist': global_direction_hist,
                'room_labels': room_labels,
            }

            save_path = os.path.join(save_dir, '{}_rooms_info.npy'.format(idx))
            with open(save_path, 'wb') as f:
                np.save(f, sample_data)

            # for debudding use
            viz_dir_rooms = os.path.join(viz_dir, 'rooms_{}'.format(idx))
            if not os.path.exists(viz_dir_rooms):
                os.makedirs(viz_dir_rooms)
            for room_idx, room_img in enumerate(room_imgs):
                room_save_path = os.path.join(viz_dir_rooms, '{}_room_corners.png'.format(room_idx))
                imsave(room_save_path, room_img)


if __name__ == '__main__':
    config_dict = load_config(file_path='/local-scratch/cjc/floor-sp/floor-sp/configs/config_associate_module.yaml')
    configs = Struct(**config_dict)
    extra_option = configs.extra if hasattr(configs, 'extra') else None
    config_str = compose_config_str(configs, keywords=['mode', 'lr', 'batch_size'], extra=extra_option)
    exp_dir = os.path.join(configs.exp_base_dir, config_str)
    if not os.path.exists(exp_dir):
        os.makedirs(exp_dir)
    configs.exp_dir = exp_dir
    if configs.seed:
        torch.manual_seed(configs.seed)
        if configs.use_cuda:
            torch.cuda.manual_seed(configs.seed)
        np.random.seed(configs.seed)
        print('set random seed to {}'.format(configs.seed))
    if configs.phase == 'train':
        print('training phase')
        train(configs)
    elif configs.phase == 'test':
        print('testing phase')
        test(configs)
    elif configs.phase == 'predict':
        print('infernece phase for room-corner relationships')
        predict(configs)
    else:
        raise NotImplementedError
