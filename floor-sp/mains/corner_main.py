import _init_paths
import os
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
from scipy.misc import imsave, imread
import numpy as np

from datasets.lianjia_corner_dataset import LianjiaCornerDataset
from datasets.lianjia_room_dataset import LianjiaRoomDataset
from models.corner_net import CornerEdgeNet
from trainers.corner_trainer import CornerTrainer
from utils.misc import save_checkpoint, count_parameters, transfer_optimizer_to_gpu
from utils.config import Struct, load_config, compose_config_str
from utils.floorplan_utils.floorplan_misc import get_corner_dir_map, get_room_shape
from utils.data_utils import get_direction_hist
import pdb


def train(configs):
    train_dataset = LianjiaCornerDataset(
        data_dir='/local-scratch/cjc/Lianjia-inverse-cad/FloorPlotter/data/Lianjia_corner',
        phase='train', augmentation=configs.augmentation)  # use rotation for augmentation

    train_loader = DataLoader(train_dataset, batch_size=configs.batch_size, shuffle=True)

    model = CornerEdgeNet(num_input_channel=5, base_pretrained=False, bin_size=36,
                      im_size=256, configs=configs)
    model.double()

    criterion = nn.BCEWithLogitsLoss(reduce=False)
    optimizer = torch.optim.Adam(model.parameters(), lr=configs.lr, weight_decay=configs.decay_rate)
    scheduler = StepLR(optimizer, step_size=configs.lr_step, gamma=0.1)

    num_parameters = count_parameters(model)
    print('total number of trainable parameters is: {}'.format(num_parameters))

    trainer = CornerTrainer(model=model, train_loader=train_loader, val_loader=None, criterion=criterion,
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
                filename='checkpoint_corner_edge_{}.pth.tar'.format(epoch_num))


def predict_corners(configs):
    predict_phase = 'test'
    predict_dataset = LianjiaCornerDataset(
        data_dir='/local-scratch/cjc/Lianjia-inverse-cad/FloorPlotter/data/Lianjia_corner', phase=predict_phase,
        augmentation='')
    # only support bs = 1 for testing
    predict_loader = DataLoader(predict_dataset, batch_size=1, shuffle=False)

    model = CornerEdgeNet(num_input_channel=5, base_pretrained=False, bin_size=36,
                      im_size=256, configs=configs)
    model.double()

    if configs.model_path:
        checkpoint = torch.load(configs.model_path)
        model.load_state_dict(checkpoint['state_dict'])
        epoch_num = checkpoint['epoch']
        print('=> loaded checkpoint {} (epoch {})'.format(configs.model_path, epoch_num))

    viz_dir = os.path.join(configs.exp_dir, 'visualization_{}'.format(predict_phase))
    if not os.path.exists(viz_dir):
        os.makedirs(viz_dir)

    model.eval()

    # disable gradient computation, we only feed forward-pass during inference
    with torch.no_grad():
        for idx, batch_data in enumerate(predict_loader):
            non_norm_image = batch_data['image_non_normalized']
            image_inputs = batch_data['image']
            mean_normal = batch_data['mean_normal']
            corner_gt_map = batch_data['corner_gt_map']
            edge_gt_map = batch_data['edge_gt_map']
            room_masks_map = batch_data['room_masks_map']

            criterion = nn.BCEWithLogitsLoss()

            inputs = torch.cat([image_inputs.unsqueeze(1), mean_normal, room_masks_map.unsqueeze(1)], dim=1)
            corner_preds_logits, edge_preds_logits, edge_preds, corner_preds = model(inputs)
            corner_preds = corner_preds.squeeze(0).cpu().numpy()
            edge_preds = edge_preds.squeeze(0).cpu().numpy()
            gt_edge = edge_gt_map.squeeze(0).cpu().numpy()
            gt_corner = corner_gt_map.squeeze(0).cpu().numpy()

            loss_c = criterion(corner_preds_logits[0], corner_gt_map[0])
            loss_e = criterion(edge_preds_logits[0], edge_gt_map[0])
            loss = loss_c + loss_e

            vectorized_preds, corner_edge_map = get_corner_dir_map(corner_preds, 256)
            vectorized_gt, gt_corner_edge_map = get_corner_dir_map(gt_corner, 256)

            input_im = np.transpose(non_norm_image.cpu().numpy().squeeze(0), [1, 2, 0])
            corner_edge_map = np.clip(corner_edge_map + input_im, 0, 255)
            gt_corner_edge_map = np.clip(gt_corner_edge_map + input_im, 0, 255)
            heatmap = corner_preds[0]
            heatmap_edge = edge_preds[0]

            global_direction_hist = get_direction_hist(edge_preds)

            print('instance loss {}'.format(loss))
            print('finish corner predictions scene No.{}'.format(idx))

            save_path_cornermap = os.path.join(viz_dir, '{}_corners_pred.png'.format(idx))
            save_path_cornermap_gt = os.path.join(viz_dir, '{}_corners_pred_gt.png'.format(idx))
            imsave(save_path_cornermap, corner_edge_map)
            imsave(save_path_cornermap_gt, gt_corner_edge_map)
            save_path_heatmap = os.path.join(viz_dir, '{}_heatmap_pred.png'.format(idx))
            save_path_heatmap_edge = os.path.join(viz_dir, '{}_heatmap_edge_pred.png'.format(idx))
            imsave(save_path_heatmap, heatmap)
            imsave(save_path_heatmap_edge, heatmap_edge)

            # dump the intermediate results to disk
            if configs.dump_prediction is True:
                dump_dir = os.path.join(exp_dir, '{}_preds'.format(predict_phase))
                if not os.path.exists(dump_dir):
                    os.makedirs(dump_dir)
                for pred_idx, pred_item in enumerate(vectorized_preds):  # keep the full binning information
                    vectorized_preds[pred_idx]['binning'] = corner_preds[1:, pred_item['corner'][1], pred_item['corner'][0]]
                dump_path = os.path.join(dump_dir, '{}_corner_preds.npy'.format(idx))
                dump_data = {
                    'vectorized_preds': vectorized_preds,
                    'corner_heatmap': heatmap,
                    'edge_preds': edge_preds,
                    'direction_hist': global_direction_hist
                }
                with open(dump_path, 'wb') as f:
                    np.save(f, dump_data)


if __name__ == '__main__':
    config_dict = load_config(file_path='./configs/config_cornernet.yaml')
    configs = Struct(**config_dict)
    extra_option = configs.extra if hasattr(configs, 'extra') else None
    config_str = compose_config_str(configs, keywords=['lr', 'batch_size', 'augmentation'], extra=extra_option)
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
    elif configs.phase == 'predict':
        print('inference phase')
        predict_corners(configs)
    else:
        raise NotImplementedError
