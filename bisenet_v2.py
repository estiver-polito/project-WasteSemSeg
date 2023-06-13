"""Bilateral Segmentation Network"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from base import xception39,ConvBNReLU



__all__ = ['BiSeNet', 'get_bisenet', 'get_bisenet_resnet18_citys']


class BiSeNet(nn.Module):
    def __init__(self, nclass, aux=False, jpu=False, pretrained_base=True, **kwargs):
        super(BiSeNet, self).__init__()
        self.aux = aux
        self.spatial_path = SpatialPath(3, 128, **kwargs)
        self.context_path = xception39()
        self.ffm = FeatureFusion(256, 256, 4, **kwargs)
        self.head = _BiSeHead(256, 64, nclass, **kwargs)
        if aux:
            self.auxlayer1 = _BiSeHead(128, 256, nclass, **kwargs)
            self.auxlayer2 = _BiSeHead(128, 256, nclass, **kwargs)

        self.__setattr__('exclusive',
                         ['spatial_path', 'context_path', 'ffm', 'head', 'auxlayer1', 'auxlayer2'] if aux else [
                             'spatial_path', 'context_path', 'ffm', 'head'])

    def forward(self, x):
        size = x.size()[2:]
        spatial_out = self.spatial_path(x)
        context_out = self.context_path(x)
        fusion_out = self.ffm(spatial_out, context_out[-1])
        outputs = []
        x = self.head(fusion_out)
        x = F.interpolate(x, size, mode='bilinear', align_corners=True)
        outputs.append(x)

        if self.aux:
            auxout1 = self.auxlayer1(context_out[0])
            auxout1 = F.interpolate(auxout1, size, mode='bilinear', align_corners=True)
            outputs.append(auxout1)
            auxout2 = self.auxlayer2(context_out[1])
            auxout2 = F.interpolate(auxout2, size, mode='bilinear', align_corners=True)
            outputs.append(auxout2)
        return tuple(outputs)


class _BiSeHead(nn.Module):
    def __init__(self, in_channels, inter_channels, nclass, norm_layer=nn.BatchNorm2d, **kwargs):
        super(_BiSeHead, self).__init__()
        self.block = nn.Sequential(
            ConvBNReLU(in_channels, inter_channels, 3, 1, 1, norm_layer=norm_layer),
            nn.Dropout(0.1),
            nn.Conv2d(inter_channels, nclass, 1)
        )

    def forward(self, x):
        x = self.block(x)
        return x


class SpatialPath(nn.Module):
    """Spatial path"""

    def __init__(self, in_channels, out_channels, norm_layer=nn.BatchNorm2d, **kwargs):
        super(SpatialPath, self).__init__()
        inter_channels = 64
        self.conv7x7 = ConvBNReLU(in_channels, inter_channels, 7, 2, 3, norm_layer=norm_layer)
        self.conv3x3_1 = ConvBNReLU(inter_channels, inter_channels, 3, 2, 1, norm_layer=norm_layer)
        self.conv3x3_2 = ConvBNReLU(inter_channels, inter_channels, 3, 2, 1, norm_layer=norm_layer)
        self.conv1x1 = ConvBNReLU(inter_channels, out_channels, 1, 1, 0, norm_layer=norm_layer)

    def forward(self, x):
        x = self.conv7x7(x)
        x = self.conv3x3_1(x)
        x = self.conv3x3_2(x)
        x = self.conv1x1(x)

        return x


class _GlobalAvgPooling(nn.Module):
    def __init__(self, in_channels, out_channels, norm_layer, **kwargs):
        super(_GlobalAvgPooling, self).__init__()
        self.gap = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            norm_layer(out_channels),
            nn.ReLU(True)
        )

    def forward(self, x):
        size = x.size()[2:]
        pool = self.gap(x)
        out = F.interpolate(pool, size, mode='bilinear', align_corners=True)
        return out


class AttentionRefinmentModule(nn.Module):
    def __init__(self, in_channels, out_channels, norm_layer=nn.BatchNorm2d, **kwargs):
        super(AttentionRefinmentModule, self).__init__()
        self.conv3x3 = ConvBNReLU(in_channels, out_channels, 3, 1, 1, norm_layer=norm_layer)
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            ConvBNReLU(out_channels, out_channels, 1, 1, 0, norm_layer=norm_layer),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.conv3x3(x)
        attention = self.channel_attention(x)
        x = x * attention
        return x




class FeatureFusion(nn.Module):
    def __init__(self, in_channels, out_channels, reduction=1, norm_layer=nn.BatchNorm2d, **kwargs):
        super(FeatureFusion, self).__init__()
        self.conv1x1 = ConvBNReLU(in_channels, out_channels, 1, 1, 0, norm_layer=norm_layer, **kwargs)
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            ConvBNReLU(out_channels, out_channels // reduction, 1, 1, 0, norm_layer=norm_layer),
            ConvBNReLU(out_channels // reduction, out_channels, 1, 1, 0, norm_layer=norm_layer),
            nn.Sigmoid()
        )

    def forward(self, x1, x2):
        fusion = torch.cat([x1, x2], dim=1)
        out = self.conv1x1(fusion)
        attention = self.channel_attention(out)
        out = out + out * attention
        return out


