import os
import torch
import torch.nn as nn
import imageio
import cv2
import torch.nn.functional as F
import numpy as np
from torch.optim import lr_scheduler

from torch.autograd import Variable
from math import exp

from torch.utils.tensorboard import SummaryWriter

""" --------------- """
from config import BaseOptions

""" 
Basic Block 
    # ConvBlock(input_size, output_size, kernel_size=3, stride=1, padding=1, bias=True, activation='prelu'
        , norm=None, pad_model=None)

    # ResnetBlock(input_size, kernel_size=3, stride=1, padding=1, bias=True, scale=1, activation='prelu'
        , norm='batch', pad_model=None)

Sample Block
    # Conv_up(c_in, mid_c, up_factor)

    # Conv_down(c_in,mid_c, up_factor)
"""
class ConvBlock(torch.nn.Module):
    def __init__(self, input_size, output_size, kernel_size=3, stride=1, padding=1, bias=True, activation='prelu', norm=None, pad_model=None):
        super(ConvBlock, self).__init__()

        self.pad_model = pad_model
        self.norm = norm
        self.input_size = input_size
        self.output_size = output_size
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.bias = bias

        if self.norm =='batch':
            self.bn = torch.nn.BatchNorm2d(self.output_size)
        elif self.norm == 'instance':
            self.bn = torch.nn.InstanceNorm2d(self.output_size)

        self.activation = activation
        if self.activation == 'relu':
            self.act = torch.nn.ReLU(True)
        elif self.activation == 'prelu':
            self.act = torch.nn.PReLU(init=0.5)
        elif self.activation == 'lrelu':
            self.act = torch.nn.LeakyReLU(0.2, True)
        elif self.activation == 'tanh':
            self.act = torch.nn.Tanh()
        elif self.activation == 'sigmoid':
            self.act = torch.nn.Sigmoid()
        
        if self.pad_model == None:   
            self.conv = torch.nn.Conv2d(self.input_size, self.output_size, self.kernel_size, self.stride, self.padding, bias=self.bias)
        elif self.pad_model == 'reflection':
            self.padding = nn.Sequential(nn.ReflectionPad2d(self.padding))
            self.conv = torch.nn.Conv2d(self.input_size, self.output_size, self.kernel_size, self.stride, 0, bias=self.bias)

    def forward(self, x):
        out = x
        if self.pad_model is not None:
            out = self.padding(out)

        if self.norm is not None:
            out = self.bn(self.conv(out))
        else:
            out = self.conv(out)

        if self.activation is not None:
            return self.act(out)
        else:
            return out


class ResnetBlock(torch.nn.Module):
    def __init__(self, input_size, kernel_size=3, stride=1, padding=1, bias=True, scale=1, activation='prelu', norm='batch', pad_model=None):
        super().__init__()

        self.norm = norm
        self.pad_model = pad_model
        self.input_size = input_size
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.bias = bias
        self.scale = scale
        
        if self.norm =='batch':
            self.normlayer = torch.nn.BatchNorm2d(input_size)
        elif self.norm == 'instance':
            self.normlayer = torch.nn.InstanceNorm2d(input_size)
        else:
            self.normlayer = None

        self.activation = activation
        if self.activation == 'relu':
            self.act = torch.nn.ReLU(True)
        elif self.activation == 'prelu':
            self.act = torch.nn.PReLU(init=0.5)
        elif self.activation == 'lrelu':
            self.act = torch.nn.LeakyReLU(0.2, True)
        elif self.activation == 'tanh':
            self.act = torch.nn.Tanh()
        elif self.activation == 'sigmoid':
            self.act = torch.nn.Sigmoid()
        else:
            self.act = None

        if self.pad_model == None:   
            self.conv1 = torch.nn.Conv2d(input_size, input_size, kernel_size, stride, padding, bias=bias)
            self.conv2 = torch.nn.Conv2d(input_size, input_size, kernel_size, stride, padding, bias=bias)
            self.pad = None
        elif self.pad_model == 'reflection':
            self.pad = nn.Sequential(nn.ReflectionPad2d(padding))
            self.conv1 = torch.nn.Conv2d(input_size, input_size, kernel_size, stride, 0, bias=bias)
            self.conv2 = torch.nn.Conv2d(input_size, input_size, kernel_size, stride, 0, bias=bias)

        layers = filter(lambda x: x is not None, [self.pad, self.conv1, self.normlayer, self.act, self.pad, self.conv2, self.normlayer, self.act])
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        residual = x
        out = x
        out = self.layers(x)
        out = out * self.scale
        out = torch.add(out, residual)
        return out

""" Sampling Block """

def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size // 2), bias=bias)


class Conv_up(nn.Module):
    def __init__(self, c_in, mid_c, up_factor):
        super(Conv_up, self).__init__()

        body = [nn.Conv2d(in_channels=c_in, out_channels=mid_c, kernel_size=3, padding=3 // 2), nn.ReLU(),
                ]
        self.body = nn.Sequential(*body)
        conv = default_conv
        if up_factor == 2:
            modules_tail = [
                nn.Upsample(scale_factor=2),
                conv(mid_c, mid_c,3),
                conv(mid_c, c_in,3),
                conv(c_in, c_in, 3)]

        elif up_factor == 3:
            modules_tail = [
                nn.Upsample(scale_factor=3),
                conv(mid_c, mid_c,3),
                conv(mid_c, c_in,3),
                conv(c_in, c_in, 3)]

        elif up_factor == 4:
            modules_tail = [
                nn.Upsample(scale_factor=4),
                conv(mid_c, mid_c,3),
                conv(mid_c, c_in,3),
                conv(c_in, c_in, 3)]
        self.tail = nn.Sequential(*modules_tail)

    def forward(self, input):

        out = self.body(input)
        out = self.tail(out)
        return out


class Conv_down(nn.Module):
    def __init__(self, c_in,mid_c, up_factor):
        super(Conv_down, self).__init__()

        body = [nn.Conv2d(in_channels=c_in, out_channels=mid_c, kernel_size=3, padding=3 // 2), nn.ReLU(),
                ]
        self.body = nn.Sequential(*body)
        conv = default_conv
        if up_factor == 4:
            modules_tail = [
                nn.MaxPool2d(4),
                conv(mid_c, c_in,3),
                conv(c_in, c_in, 3)]

        elif up_factor == 3:
            modules_tail = [
                nn.MaxPool2d(3),
                conv(mid_c, c_in,3),
                conv(c_in, c_in, 3)]

        elif up_factor == 2:
            modules_tail = [
                nn.MaxPool2d(2),
                conv(mid_c, c_in,3),
                conv(c_in, c_in, 3)]
        self.tail = nn.Sequential(*modules_tail)

    def forward(self, input):

        out = self.body(input)
        out = self.tail(out)
        return out



class att_spatial(nn.Module):
    def __init__(self, res_num=2):
        super(att_spatial, self).__init__()

        block = [
            ConvBlock(2, 32, 3, 1, 1, activation='prelu', norm=None, bias=False),
        ]
        for i in range(res_num):
            block.append(ResnetBlock(32, 3, 1, 1, 0.1, activation='prelu', norm=None))
        self.block = nn.Sequential(*block)
        self.spatial = ConvBlock(2, 1, 3, 1, 1, activation='prelu', norm=None, bias=False)

    def forward(self, x):
        x = self.block(x)
        x_compress = torch.cat([torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)], dim=1)
        x_out = self.spatial(x_compress)

        scale = torch.sigmoid(x_out)  # broadcasting
        return scale
