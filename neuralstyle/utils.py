import os
import torch
from torch.utils.serialization import load_lua
from vgg16 import Vgg16


def init_vgg16(model_folder):
    """load the vgg16 model feature"""
    if not os.path.exists(model_folder + '/vgg16.weight'):
        if not os.path.exists(model_folder + '/vgg16.t7'):
            os.system(
                'wget http://cs.stanford.edu/people/jcjohns/fast-neural-style/models/vgg16.t7 ' + model_folder + '/vgg16.t7')
        vgglua = load_lua(model_folder + '/vgg16.t7')
        vgg = Vgg16()
        for (src, dst) in zip(vgglua.parameters()[0], vgg.parameters()):
            dst[:] = src[:]
        torch.save(vgg.state_dict(), model_folder + '/vgg16.weight')
        # load using :
        # vgg_model.load_state_dict( torch.load('model/vgg16.weight'))
