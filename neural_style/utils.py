import os
import sys

import torch
from torch.utils.serialization import load_lua

from vgg16 import Vgg16

import numpy as np
from PIL import Image
from scipy.misc import imread, imsave, imresize
from scipy.ndimage.filters import median_filter


# result: RGB CxHxW [0,255] torch.FloatTensor
def tensor_load_rgbimage(filename, size=None):
    img = Image.open(filename)
    if size is not None:
        img = img.resize((size, size), Image.ANTIALIAS)
    img = np.array(img).transpose(2, 0, 1)
    img = torch.from_numpy(img).float()
    return img


def gram_matrix(y):
    (b, ch, h, w) = y.size()
    features = y.view(b, ch, w * h)
    features_t = features.transpose(1, 2)
    gram = features.bmm(features_t) / (ch * h * w)
    return gram


def add_imagenet_mean(img):
    """Add ImageNet mean pixel-wise to a BGR image."""
    img[0, :, :] += 103.939
    img[1, :, :] += 116.779
    img[2, :, :] += 123.68


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


def subtract_imagenet_mean_batch(batch):
    """Subtract ImageNet mean pixel-wise from a BGR image."""
    tensortype = type(batch)
    mean = tensortype(batch.size())
    mean[:, 0, :, :] = 103.939
    mean[:, 1, :, :] = 116.779
    mean[:, 2, :, :] = 123.680
    batch -= mean


def preprocess_batch(batch):
    batch = batch.transpose(0, 1)
    (r, g, b) = torch.chunk(batch, 3)
    batch = torch.cat((b, g, r))
    batch = batch.transpose(0, 1)
    subtract_imagenet_mean_batch(batch)
    return batch


def init_vgg16(model_folder):
    """load the vgg16 model feature"""
    if not os.path.exists(model_folder + '/vgg16.weight'):
        if not os.path.exists(model_folder + '/vgg16.t7'):
            os.system(
                'wget http://cs.stanford.edu/people/jcjohns/fast-neural-style/models/vgg16.t7 -O ' + model_folder + '/vgg16.t7')
        vgglua = load_lua(model_folder + '/vgg16.t7')
        vgg = Vgg16()
        for (src, dst) in zip(vgglua.parameters()[0], vgg.parameters()):
            dst[:].data = src[:]
        torch.save(vgg.state_dict(), model_folder + '/vgg16.weight')
