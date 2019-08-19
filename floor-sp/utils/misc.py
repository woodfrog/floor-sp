import numpy as np
import os
import torch
import shutil


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def save_checkpoint(state, is_best, checkpoint='checkpoints/', filename='checkpoint.pth.tar', snapshot=None):
    """Saves checkpoint to disk"""
    # todo: also save the actual preds
    # preds = to_numpy(preds)
    if not os.path.exists(checkpoint):
        os.makedirs(checkpoint)
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)

    if snapshot and state.epoch % snapshot == 0:
        shutil.copyfile(filepath, os.path.join(checkpoint, 'checkpoint_{}.pth.tar'.format(state.epoch)))

    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, 'model_best.pth.tar'))


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def transfer_optimizer_to_gpu(optimizer):
    for state in optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.cuda()


def to_numpy(tensor):
    if torch.is_tensor(tensor):
        if tensor.requires_grad:
            return tensor.detach().numpy()
        else:
            return tensor.cpu().numpy()
    elif type(tensor).__module__ != 'numpy':
        raise ValueError("Cannot convert {} to numpy array"
                         .format(type(tensor)))
    return tensor


def binary_pred_accuracy(preds, labels):
    N = preds.shape[0]
    preds[preds >= 0.5] = 1
    preds[preds < 0.5] = 0
    correct = preds == labels
    num_correct = np.sum(correct)
    return num_correct * 1.0 / N
