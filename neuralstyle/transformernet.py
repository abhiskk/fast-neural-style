import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class TransformerNet(torch.nn.Module):
    def __init__(self):
        super(TransformerNet, self).__init__()

        # Padding layer
        self.reflect_padding = nn.ReflectionPad2d(20)

        # Initial convolution layers
        self.conv1 = ConvLayer(3, 32, 9, 1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = ConvLayer(32, 64, 3, 2)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = ConvLayer(64, 128, 3, 2)
        self.bn3 = nn.BatchNorm2d(128)

        # Residual layers
        self.res1 = ResidualBlock(128)
        self.res2 = ResidualBlock(128)
        self.res3 = ResidualBlock(128)
        self.res4 = ResidualBlock(128)
        self.res5 = ResidualBlock(128)

        # Upsampling Layers
        self.deconv1 = ResizeConvLayer(128, 64, 3, 2)
        self.bn4 = nn.BatchNorm2d(64)
        self.deconv2 = ResizeConvLayer(64, 32, 3, 2)
        self.bn5 = nn.BatchNorm2d(32)
        self.deconv3 = ConvLayer(32, 3, 9, 1)
        self.bn6 = nn.BatchNorm2d(3)

        # Non-linearities
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

    def forward(self, X):
        in_X = self.reflect_padding(X)
        y = self.relu(self.bn1(self.conv1(in_X)))
        y = self.relu(self.bn2(self.conv2(y)))
        y = self.relu(self.bn3(self.conv3(y)))
        y = self.res1(y)
        y = self.res2(y)
        y = self.res3(y)
        y = self.res4(y)
        y = self.res5(y)
        y = self.relu(self.bn4(self.deconv1(y)))
        y = self.relu(self.bn5(self.deconv2(y)))
        # TODO: Check if batch-normalization is needed here
        y = self.tanh(self.bn6(self.deconv3(y)))
        # TODO: Implement scaling tanh
        raise NotImplementedError


class ResidualBlock(torch.nn.Module):
    """ResidualBlock
    introduced in: https://arxiv.org/abs/1512.03385
    recommended architecture: http://torch.ch/blog/2016/02/04/resnets.html
    """

    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = ConvLayer(channels, channels, 3, 1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = ConvLayer(channels, channels, 3, 1)
        self.bn2 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        # TODO: Remove the following batch-normalization when doing instance-norm
        out = self.bn2(out)
        out += residual
        return out


class ConvLayer(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(ConvLayer, self).__init__()
        reflection_padding = int(np.floor(kernel_size / 2))
        self.reflection_pad = nn.ReflectionPad2d(reflection_padding)
        # TODO: Check the method for initialization of conv layer weights
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride)

    def forward(self, x):
        out = self.reflection_pad(x)
        out = self.conv2d(out)
        return out


class ResizeConvLayer(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(ResizeConvLayer, self).__init__()
        self.resize_conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride)

    def forward(self, x):
        # TODO: maybe add reflection padding here also
        out = self.resize_conv(x)
        return out


# TODO: Implement InstanceNormalization
class InstanceNormalization(torch.nn.Module):
    def __init__(self, dim):
        super(InstanceNormalization, self).__init__()
        raise NotImplementedError

    def forward(self, x):
        raise NotImplementedError


"""
Use as:
feature_generated = vgg_model(y)
feature_content = vgg_model(X_content)
"""
class Vgg16Part(torch.nn.Module):
    def __init__(self):
        super(Vgg16Part, self).__init__()
        self.conv1_1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)

        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)

        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.conv3_3 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)

        self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv4_3 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)

        self.conv5_1 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv5_3 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)

    def forward(self, X):
        h = F.relu(self.conv1_1(X))
        h = F.relu(self.conv1_2(h))
        relu1_2 = h
        h = F.max_pool2d(h, kernel_size=2, stride=2)

        h = F.relu(self.conv2_1(h))
        h = F.relu(self.conv2_2(h))
        relu2_2 = h
        h = F.max_pool2d(h, kernel_size=2, stride=2)

        h = F.relu(self.conv3_1(h))
        h = F.relu(self.conv3_2(h))
        h = F.relu(self.conv3_3(h))
        relu3_3 = h
        h = F.max_pool2d(h, kernel_size=2, stride=2)

        h = F.relu(self.conv4_1(h))
        h = F.relu(self.conv4_2(h))
        h = F.relu(self.conv4_3(h))
        relu4_3 = h

        return [relu1_2, relu2_2, relu3_3, relu4_3]
