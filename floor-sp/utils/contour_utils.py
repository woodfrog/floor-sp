from utils.misc import to_numpy
from utils.snake import active_contour

import numpy as np
import matplotlib.pyplot as plt
from scipy.misc import imresize, imread
import pdb


def visualize_contours(image, inputs, preds, configs, save_path=None):
    show_img = np.copy(image)[0].transpose([1, 2, 0])
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.imshow(show_img, cmap=plt.cm.gray)

    assert len(inputs) == len(preds)
    for idx in range(len(inputs)):
        init = to_numpy(inputs[idx].squeeze(0) * configs.im_size)
        pred = to_numpy(preds[idx].squeeze(0) * configs.im_size)

        ax.plot(init[:, 0], init[:, 1], 'ro', markersize=0.5)
        ax.plot(pred[:, 0], pred[:, 1], 'bo', markersize=0.5)

    if save_path:
        plt.savefig(save_path)
    plt.close()


def visualize_contours_using_snake(image, inputs, vf, configs, save_path=None, corner_path=None):
    img = np.copy(image)[0].transpose([1, 2, 0])
    vf = vf.numpy()[0].transpose([1, 2, 0])
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.imshow(img, cmap=plt.cm.gray)

    preds = list()

    for init in inputs:
        pred = active_contour(img, to_numpy(init.squeeze(0) * configs.im_size), vf, w_edge=0.5, gamma=0.1)
        preds.append(pred)


    for idx in range(len(inputs)):
        init = to_numpy(inputs[idx].squeeze(0) * configs.im_size)
        pred = preds[idx]
        # ax.plot(init[:, 0], init[:, 1], 'ro', markersize=0.5)
        ax.plot(pred[:, 0], pred[:, 1], 'bo', markersize=1.0)

    corner_img = imread(corner_path)

    for y in range(configs.im_size):
        for x in range(configs.im_size):
            if corner_img[y, x] >= 128:
                ax.plot(x, y, 'ro', markersize=0.5)

    if save_path:
        plt.savefig(save_path)
    plt.close()


def visualize_vector_fields(room_map, vf, inputs, preds, configs, save_path=None):
    room_map = room_map.numpy()[0]
    vf = vf.numpy()[0].transpose([1, 2, 0])
    fig, ax = plt.subplots()
    ax.set_axis_off()
    image = imresize(room_map * 20, [64, 64])
    plt.imshow(image)
    plt.quiver(vf[::4, ::4, 0], -vf[::4, ::4, 1], units='width')
    assert len(inputs) == len(preds)

    for idx in range(len(inputs)):
        init = to_numpy(inputs[idx].squeeze(0) * configs.im_size)
        pred = to_numpy(preds[idx].squeeze(0) * configs.im_size)
        ax.plot(init[:, 0] / 4, init[:, 1] / 4, 'ro', markersize=0.5)
        ax.plot(pred[:, 0] / 4, pred[:, 1] / 4, 'bo', markersize=0.5)

    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    plt.close()
