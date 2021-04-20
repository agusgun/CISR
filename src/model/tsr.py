import cv2
import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable

from model import common

def make_model(args, parent=False):
    return TSR(args)

def conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2), bias=bias)


### RRG Based Model from CycleISP ###

### Spatial Attention from CBAM
class BasicConv(nn.Module):
    def __init__(self, in_feature, out_feature, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=False, bias=False):
        super(BasicConv, self).__init__()
        self.conv = nn.Conv2d(in_feature, out_feature, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = BatchNorm2d(out_feature, eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat( (torch.max(x,1)[0].unsqueeze(1), torch.mean(x,1).unsqueeze(1)), dim=1 )

class SpatialAttentionLayer(nn.Module):
    def __init__(self, kernel_size=3):
        super(SpatialAttentionLayer, self).__init__()
        self.compress = ChannelPool()
        self.spatial = BasicConv(2, 1, kernel_size, padding=kernel_size // 2, relu=False)

    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = torch.sigmoid(x_out)
        return x * scale
    
### Channel Attention from SE Network

class ChannelAttentionLayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(ChannelAttentionLayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_fc = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_fc(y)
        return x * y

class DAB(nn.Module):
    def __init__(self, n_feat, reduction, bias=True, bn=False):
        super(DAB, self).__init__()

        modules_head = []
        modules_head.append(conv(n_feat, n_feat, 3, bias=bias))
        modules_head.append(nn.ReLU())
        modules_head.append(conv(n_feat, n_feat, 3, bias=bias))
        self.head = nn.Sequential(*modules_head)

        self.spatial_attention = SpatialAttentionLayer()
        self.channel_attention = ChannelAttentionLayer(n_feat, reduction)
        self.conv_last = conv(n_feat * 2, n_feat, 1)

    def forward(self, x):
        out = self.head(x)
        sa_out = self.spatial_attention(out)
        ca_out = self.channel_attention(out)
        out = torch.cat([sa_out, ca_out], dim=1)
        out = self.conv_last(out)
        out += x
        return out

class RRG(nn.Module):
    def __init__(self, n_feat, reduction, n_dab):
        super(RRG, self).__init__()

        modules_body = []
        for i in range(n_dab):
            modules_body.append(DAB(n_feat, reduction))
        self.body = nn.Sequential(*modules_body)
        self.conv_last = conv(n_feat, n_feat, 3)
    
    def forward(self, x):
        out = self.body(x)
        out = self.conv_last(out)
        out += x
        return out
        

class TSR(nn.Module):
    def __init__(self, args):
        super(TSR, self).__init__()
        n_rrg = args.n_rrg
        n_dab = args.n_dab
        in_channel = 3
        out_channel = 3
        n_feat = 96
        reduction = 8
        scale = args.scale[0]

        self.conv_first = conv(in_channel, n_feat, 3)

        modules_body = []
        for i in range(n_rrg):
            modules_body.append(RRG(n_feat, reduction, n_dab))
        self.body = nn.Sequential(*modules_body)
        self.conv_last = conv(n_feat, n_feat, 3)

        modules_tail = [
            common.Upsampler(conv, scale, n_feat, act=False),
            conv(n_feat, out_channel, 3)]
        self.tail = nn.Sequential(*modules_tail)

    def forward(self, x):
        x = self.conv_first(x)
        out = self.body(x)
        out = self.conv_last(out)
        out += x

        out = self.tail(out)
        return out
