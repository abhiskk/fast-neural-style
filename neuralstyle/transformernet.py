import numpy as np
import torch
import torch.nn as nn


class TransformerNet(torch.nn.Module):
    def __init__(self):
        super(TransformerNet, self).__init__()

        # Initial convolution layers
        self.conv1 = ConvLayer(3, 32, 9, 1)
        self.in1 = InstanceNormalization(32)
        self.conv2 = ConvLayer(32, 64, 3, 2)
        self.in2 = InstanceNormalization(64)
        self.conv3 = ConvLayer(64, 128, 3, 2)
        self.in3 = InstanceNormalization(128)

        # Residual layers
        self.res1 = ResidualBlock(128)
        self.res2 = ResidualBlock(128)
        self.res3 = ResidualBlock(128)
        self.res4 = ResidualBlock(128)
        self.res5 = ResidualBlock(128)

        # Upsampling Layers
        self.deconv1 = ResizeConvLayer(128, 64, 3, 2)
        self.in4 = InstanceNormalization(64)
        self.deconv2 = ResizeConvLayer(64, 32, 3, 2)
        self.in5 = InstanceNormalization(32)
        self.deconv3 = ConvLayer(32, 3, 9, 1)
        self.in6 = InstanceNormalization(3)

        # Non-linearities
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

    def forward(self, X):
        in_X = X
        y = self.relu(self.in1(self.conv1(in_X)))
        y = self.relu(self.in2(self.conv2(y)))
        y = self.relu(self.in3(self.conv3(y)))
        y = self.res1(y)
        y = self.res2(y)
        y = self.res3(y)
        y = self.res4(y)
        y = self.res5(y)
        y = self.relu(self.in4(self.deconv1(y)))
        y = self.relu(self.in5(self.deconv2(y)))
        # TODO: Try adding instance-normalization below
        y = self.tanh(self.deconv3(y))
        y *= 150.0
        return y


class ResidualBlock(torch.nn.Module):
    """ResidualBlock
    introduced in: https://arxiv.org/abs/1512.03385
    recommended architecture: http://torch.ch/blog/2016/02/04/resnets.html
    """

    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = ConvLayer(channels, channels, 3, 1)
        self.in1 = InstanceNormalization(channels)
        self.conv2 = ConvLayer(channels, channels, 3, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.in1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out += residual
        return out


class ConvLayer(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(ConvLayer, self).__init__()
        reflection_padding = int(np.floor(kernel_size / 2))
        self.reflection_pad = nn.ReflectionPad2d(reflection_padding)
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride)

    def forward(self, x):
        out = self.reflection_pad(x)
        out = self.conv2d(out)
        return out


class ResizeConvLayer(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(ResizeConvLayer, self).__init__()
        reflection_padding = int(np.floor(kernel_size / 2))
        self.reflection_pad = nn.ReflectionPad2d(reflection_padding)
        self.resize_conv2d = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride)

    def forward(self, x):
        out = self.resize_conv2d(x)
        return out


class InstanceNormalization(torch.nn.Module):
    """InstanceNormalization
    Improves convergence of neural-style.
    ref: https://arxiv.org/pdf/1607.08022.pdf
    """
    def __init__(self, dim):
        super(InstanceNormalization, self).__init__()
        self.scale = nn.Parameter(torch.FloatTensor(dim))
        self.shift = nn.Parameter(torch.FloatTensor(dim))
        self.reset_parameters()

    def reset_parameters(self):
        self.scale.data.uniform_()
        self.shift.data.zero_()

    def _check_dim(self, x):
        if x.dim() != 4:
            raise ValueError('expected 4D input (got {}D input)'
                             .format(x.dim()))

    def forward(self, x):
        self._check_dim(x)
        n = x.size()[2] * x.size()[3]
        t = x.resize(x.size()[0], x.size()[1], 1, n)
        mean = torch.mean(t, 3).expand_as(x)
        # Calculate the biased var. torch.var returns unbiased var
        var = torch.var(t, 3).expand_as(x)
        scale_broadcast = self.scale.unsqueeze(1).unsqueeze(1).unsqueeze(0)
        scale_broadcast = scale_broadcast.expand_as(x)
        shift_broadcast = self.shift.unsqueeze(1).unsqueeze(1).unsqueeze(0)
        shift_broadcast = shift_broadcast.expand_as(x)
        out = (x - mean) / torch.sqrt(var + 1e-5)
        out = (out * scale_broadcast) + shift_broadcast
        return out
