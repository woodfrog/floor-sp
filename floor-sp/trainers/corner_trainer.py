import time
import os
import torch
from utils.misc import AverageMeter
from scipy.misc import imsave
import numpy as np
from utils.floorplan_utils.floorplan_misc import get_corner_dir_map
import pdb


class CornerTrainer:
    def __init__(self, model, train_loader, val_loader, criterion, optimizer, configs, mode='corner'):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.configs = configs
        self.mode = mode

    def train_epoch(self, epoch_num):
        batch_time = AverageMeter()
        losses_edge = AverageMeter()
        losses_corner = AverageMeter()

        self.model.train()

        end = time.time()
        for iter_i, batch_data in enumerate(self.train_loader):
            image_inputs = batch_data['image']
            if self.mode == 'corner':
                corner_target_maps = batch_data['corner_gt_map']
                edge_target_maps = batch_data['edge_gt_map']
                room_masks_map = batch_data['room_masks_map']
            else:
                raise ValueError('Invalid mode {}'.format(self.mode))

            mean_normal = batch_data['mean_normal']

            # contour_image = batch_data['contour_image']

            if self.configs.use_cuda:
                image_inputs = image_inputs.cuda()
                mean_normal = mean_normal.cuda()
                corner_target_maps = corner_target_maps.cuda()
                edge_target_maps = edge_target_maps.cuda()

            room_masks_map = room_masks_map.cuda()
            inputs = torch.cat([image_inputs.unsqueeze(1), mean_normal, room_masks_map.unsqueeze(1)], dim=1)

            corner_preds_logits, edge_preds_logits, edge_preds, corner_preds = self.model(inputs)

            # # mask the binning part, only predicting directions for places with corners
            loss_mask_c = corner_target_maps[:, 0, :, :].clone().unsqueeze(1) * 4 + 1
            loss_c = self.criterion(corner_preds_logits, corner_target_maps)
            loss_c = loss_c * loss_mask_c
            loss_c = loss_c.mean(2).mean(2).mean(0).sum()  # take mean over batch, H, W, sum over C

            loss_mask_e = edge_target_maps[:, 0, :, :].clone().unsqueeze(1) * 4 + 1
            loss_e = self.criterion(edge_preds_logits, edge_target_maps)
            loss_e = loss_e * loss_mask_e
            loss_e = loss_e.mean(2).mean(2).mean(0).sum()  # take mean over batch, H, W, sum over C

            loss = loss_e + loss_c

            losses_edge.update(loss_e.data, image_inputs.size(0))
            losses_corner.update(loss_c.data, image_inputs.size(0))

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Edge pred Loss {loss1.val:.4f} ({loss1.avg:.4f})\t'
                  'Corner pred Loss {loss2.val:.4f} ({loss2.avg:.4f})'.format(epoch_num, iter_i, len(self.train_loader),
                                                                              batch_time=batch_time,
                                                                              loss1=losses_edge, loss2=losses_corner))

            if iter_i % self.configs.visualize_iter == 0:
                viz_dir = os.path.join(self.configs.exp_dir, 'training_viz')
                gt_file_path = os.path.join(viz_dir, 'epoch_{}_iter_{}_gt.png'.format(epoch_num, iter_i))
                gt_edge_file_path = os.path.join(viz_dir, 'epoch_{}_iter_{}_gt_edge.png'.format(epoch_num, iter_i))
                heatmap_path = os.path.join(viz_dir, 'epoch_{}_iter_{}_preds.png'.format(epoch_num, iter_i))
                heatmap_edge_path = os.path.join(viz_dir, 'epoch_{}_iter_{}_preds_edge.png'.format(epoch_num, iter_i))
                # corner_edge_path = os.path.join(viz_dir, 'epoch_{}_iter_{}_corner_edge.png'.format(epoch_num, iter_i))
                gt_map_path = os.path.join(viz_dir, 'epoch_{}_iter_{}_gt_corner_edge.png'.format(epoch_num, iter_i))
                # simply use the first element in the batch
                corner_preds_np = corner_preds[0].detach().cpu().numpy()
                edge_preds_np = edge_preds[0].detach().cpu().numpy()
                edge_gt_np = edge_target_maps[0].cpu().numpy()
                corner_gt_np = corner_target_maps[0].cpu().numpy()
                if self.mode == 'corner':
                    _, gt_corner_edge_map = get_corner_dir_map(corner_gt_np, 256)
                    imsave(gt_map_path, gt_corner_edge_map)
                imsave(gt_file_path, corner_gt_np[0])
                imsave(heatmap_path, corner_preds_np[0])
                imsave(gt_edge_file_path, edge_gt_np[0])
                imsave(heatmap_edge_path, edge_preds_np[0])
                # imsave(corner_edge_path, corner_edge_map)
