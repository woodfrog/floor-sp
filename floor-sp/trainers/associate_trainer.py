import time
import os
import torch
from utils.misc import AverageMeter, binary_pred_accuracy
from scipy.misc import imsave
import pdb


class AssociateTrainer:
    def __init__(self, model, train_loader, val_loader, criterion, optimizer, configs):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.configs = configs

    def train_epoch(self, epoch_num):
        batch_time = AverageMeter()
        losses = AverageMeter()
        acces = AverageMeter()

        self.model.train()

        end = time.time()
        for iter_i, batch_data in enumerate(self.train_loader):
            image_inputs = batch_data['image']
            mean_normal = batch_data['mean_normal']
            room_mask = batch_data['room_mask']
            if self.configs.mode == 'room_corner':
                corner_map = batch_data['corner_map']
            else:
                corner_map = torch.stack([batch_data['corners_map'], batch_data['edge_map']], 1)
            label = batch_data['label']

            if self.configs.use_cuda:
                image_inputs = image_inputs.cuda()
                mean_normal = mean_normal.cuda()
                room_mask = room_mask.cuda()
                corner_map = corner_map.cuda()
                label = label.cuda()

            if self.configs.mode == 'room_corner':
                corner_map = corner_map.unsqueeze(1)

            inputs = torch.cat([image_inputs.unsqueeze(1), mean_normal, room_mask.unsqueeze(1), corner_map], dim=1)
            logits, preds = self.model(inputs)

            loss = self.criterion(logits, label)
            losses.update(loss.data, image_inputs.size(0))

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            if self.configs.mode == 'corner_corner':
                acc = binary_pred_accuracy(preds.detach().cpu().numpy(), label.cpu().numpy())
            else:
                acc = binary_pred_accuracy(preds.detach().cpu().numpy()[:, 0], label.cpu().numpy()[:, 0])
            acces.update(acc, image_inputs.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Corner pred Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Acc {acc.val:.3f} ({acc.avg:.3f})'.format(epoch_num, iter_i, len(self.train_loader), batch_time=batch_time, loss=losses, acc=acces))

            if iter_i > self.configs.max_iter_per_epoch:
                break
