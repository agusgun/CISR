import cv2
import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable

from model import common

def make_model(args, parent=False):
    return CISR(args)

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
        scale = self.compress(x)
        scale = self.spatial(scale)
        scale = torch.sigmoid(scale)
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
        scale = self.avg_pool(x)
        scale = self.conv_fc(scale)
        return x * scale

class PDAB(nn.Module):
    # Parallel dual attention block similar to RRG from CycleISP
    def __init__(self, n_feat, reduction, bias=True, bn=False):
        super(PDAB, self).__init__()

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

class SDAB(nn.Module):
    # Sequential dual attention block similar to CBAM
    def __init__(self, n_feat, reduction, bias=True, bn=False):
        super(SDAB, self).__init__()

        modules_head = []
        modules_head.append(conv(n_feat, n_feat, 3, bias=bias))
        modules_head.append(nn.ReLU())
        modules_head.append(conv(n_feat, n_feat, 3, bias=bias))
        self.head = nn.Sequential(*modules_head)

        self.spatial_attention = SpatialAttentionLayer()
        self.channel_attention = ChannelAttentionLayer(n_feat, reduction)

    def forward(self, x):
        out = self.head(x)
        out = self.channel_attention(out)
        out = self.spatial_attention(out)
        out += x
        return out

class PDABRG(nn.Module):
    # Recursive group with parallel dual attention block similar to RRG
    def __init__(self, n_feat, reduction, n_dab):
        super(PDABRG, self).__init__()

        modules_body = []
        for i in range(n_dab):
            modules_body.append(PDAB(n_feat, reduction))
        self.body = nn.Sequential(*modules_body)
        self.conv_last = conv(n_feat, n_feat, 3)
    
    def forward(self, x):
        out = self.body(x)
        out = self.conv_last(out)
        out += x
        return out

class SDABRG(nn.Module):
    # Recursive group with sequential dual attention block
    def __init__(self, n_feat, reduction, n_dab):
        super(SDABRG, self).__init__()

        modules_body = []
        for i in range(n_dab):
            modules_body.append(SDAB(n_feat, reduction))
        self.body = nn.Sequential(*modules_body)
        self.conv_last = conv(n_feat, n_feat, 3)
    
    def forward(self, x):
        out = self.body(x)
        out = self.conv_last(out)
        out += x
        return out

class CISR(nn.Module):
    def __init__(self, args):
        super(CISR, self).__init__()
        n_coarse_darg = args.n_coarse_darg
        n_coarse_dab = args.n_coarse_dab
        n_fine_darg = args.n_fine_darg
        n_fine_dab = args.n_fine_dab
        in_channel = 3
        out_channel = 3
        n_feat = 96
        reduction = 8
        scale = args.scale[0]
        self.auxiliary_out = args.auxiliary_out

        self.shallow_feature_extraction = conv(in_channel, n_feat, 3)

        modules_coarse_rirg = []
        for i in range(n_coarse_darg):
            modules_coarse_rirg.append(PDABRG(n_feat, reduction, n_coarse_dab))
        modules_coarse_rirg.append(conv(n_feat, n_feat, 3))
        self.coarse_rirg = nn.Sequential(*modules_coarse_rirg)

        modules_coarse_upsampler = [
            common.Upsampler(conv, scale, n_feat, act=False),
            conv(n_feat, out_channel, 3)] # out channel = n_feat (the first network)
        self.coarse_upsampler = nn.Sequential(*modules_coarse_upsampler)
        
        self.refinement_first = conv(out_channel, n_feat, 3) # don't use this (the first network)
        modules_fine_rirg = []
        for i in range(n_fine_darg):
            modules_fine_rirg.append(SDABRG(n_feat, reduction, n_fine_dab))
        self.fine_rirg = nn.Sequential(*modules_fine_rirg)
        self.refinement_last = conv(n_feat, out_channel, 3)

    def forward(self, x):
        x = self.shallow_feature_extraction(x)
        coarse = self.coarse_rirg(x)
        coarse += x
        
        coarse = self.coarse_upsampler(coarse)
        
        fine = self.refinement_first(coarse)
        fine = self.fine_rirg(fine)
        fine = self.refinement_last(fine)
        fine += coarse

        #TODO: fix the network so it is simmilar to the network created in draw io

        if self.auxiliary_out:
            return coarse, fine
        else:
            return fine
