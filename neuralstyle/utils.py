import os
import sys

import torch
from torch.utils.serialization import load_lua

from vgg16 import Vgg16

import numpy as np
from scipy.misc import imread, imsave, imresize
from scipy.ndimage.filters import median_filter


def subtract_imagenet_mean(img):
    """Subtract ImageNet mean pixel-wise from a BGR image."""
    img[0, :, :] -= 103.939
    img[1, :, :] -= 116.779
    img[2, :, :] -= 123.68


def add_imagenet_mean(img):
    """Add ImageNet mean pixel-wise to a BGR image."""
    img[0, :, :] += 103.939
    img[1, :, :] += 116.779
    img[2, :, :] += 123.68


def load_and_preprocess_img(filename, size=None, center_crop=False):
    """Load an image, and pre-process it as needed by models."""
    try:
        img = imread(filename, mode="RGB")
    except OSError as e:
        print(e)
        sys.exit(1)

    if center_crop:
        # Extract a square crop from the center of the image.
        cur_shape = img.shape[:2]
        shorter_side = min(cur_shape)
        longer_side_xs = max(cur_shape) - shorter_side
        longer_side_start = int(longer_side_xs / 2.)
        longer_side_slice = slice(longer_side_start, longer_side_start + shorter_side)
        if shorter_side == cur_shape[0]:
            img = img[:, longer_side_slice, :]
        else:
            img = img[longer_side_slice, :, :]

    if size is not None:
        # Resize the image.
        cur_shape = img.shape[:2]
        shorter_side = min(cur_shape)
        aspect = max(cur_shape) / float(shorter_side)
        new_shorter_side = int(size / aspect)
        if shorter_side == cur_shape[0]:
            new_shape = (new_shorter_side, size)
        else:
            new_shape = (size, new_shorter_side)
        img = imresize(img, new_shape)

    # Bring the color dimension to the front, convert to BGR.
    img = img.transpose((2, 0, 1))[::-1].astype(float)

    subtract_imagenet_mean(img)
    return img[np.newaxis, :]


def deprocess_img_and_save(img, filename):
    """Undo pre-processing on an image, and save it."""
    img = img[0, :, :, :]
    add_imagenet_mean(img)
    img = img[::-1].transpose((1, 2, 0))
    img = np.clip(img, 0, 255).astype(np.uint8)
    img = median_filter(img, size=(3, 3, 1))
    try:
        imsave(filename, img)
    except OSError as e:
        print(e)
        sys.exit(1)


def init_vgg16(model_folder):
    """load the vgg16 model feature"""
    if not os.path.exists(model_folder + '/vgg16.weight'):
        if not os.path.exists(model_folder + '/vgg16.t7'):
            os.system(
                'wget http://cs.stanford.edu/people/jcjohns/fast-neural-style/models/vgg16.t7 -O ' + model_folder + '/vgg16.t7')
        vgglua = load_lua(model_folder + '/vgg16.t7')
        vgg = Vgg16()
        for (src, dst) in zip(vgglua.parameters()[0], vgg.parameters()):
            dst[:] = src[:]
        torch.save(vgg.state_dict(), model_folder + '/vgg16.weight')
        # load using :
        # vgg_model.load_state_dict( torch.load('model/vgg16.weight'))
