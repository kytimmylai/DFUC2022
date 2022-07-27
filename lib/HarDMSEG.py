#!/usr/bin/env python
# coding: utf-8
# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
from .kingnet import kingnet53
from .kingnet2 import CSPkingnet53
from .lawinloss import LawinHead, LawinHead2, LawinHead3, LawinHead4, LawinHead5

class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, 
                              stride=stride, padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.bn(self.conv(x))
        
class Mish(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        x = x * (torch.tanh(F.softplus(x)))
        return x

class RFB_modified(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(RFB_modified, self).__init__()
        self.relu = nn.ReLU(True)
        self.mish = Mish()
        self.branch0 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
        )
        self.branch1 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 3), padding=(0, 1)),
            BasicConv2d(out_channel, out_channel, kernel_size=(3, 1), padding=(1, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=3, dilation=3)
        )
        self.branch2 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 5), padding=(0, 2)),
            BasicConv2d(out_channel, out_channel, kernel_size=(5, 1), padding=(2, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=5, dilation=5)
        )
        self.branch3 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 7), padding=(0, 3)),
            BasicConv2d(out_channel, out_channel, kernel_size=(7, 1), padding=(3, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=7, dilation=7)
        )
        self.conv_cat = BasicConv2d(4*out_channel, out_channel, 3, padding=1)
        self.conv_res = BasicConv2d(in_channel, out_channel, 1)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        x_cat = self.conv_cat(torch.cat((x0, x1, x2, x3), 1))

        #x = nn.ReLU(True)(x_cat + self.conv_res(x))
        x = self.mish(x_cat + self.conv_res(x))
        return x

class aggregation_base(nn.Module):
    def __init__(self, channel, class_num = 1):
        super(aggregation_base, self).__init__()
        
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            
        self.conv_upsample1 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample2 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample3 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample4 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample5 = BasicConv2d(2*channel, 2*channel, 3, padding=1)

        self.conv_concat2 = BasicConv2d(2*channel, 2*channel, 3, padding=1)
        self.conv_concat3 = BasicConv2d(3*channel, 3*channel, 3, padding=1)
        #self.sap = SAPblock(3*channel)

        self.conv4 = BasicConv2d(3*channel, 3*channel, 3, padding=1)
        # (input channel, class channel, kernel)
        self.conv5 = nn.Conv2d(3*channel, class_num, 1)
        
    def forward(self, x1, x2, x3):
        x1_1 = self.upsample(x1)
            
        x2_1 = self.conv_upsample1(x1_1) * x2
        x2_2 = torch.cat((x2_1, self.conv_upsample4(x1_1)), 1)
        x2_2 = self.conv_concat2(x2_2)
            
        x3_1 = self.conv_upsample2(self.upsample(x1_1))* self.conv_upsample3(self.upsample(x2)) * x3
        x3_2 = torch.cat((x3_1, self.conv_upsample5(self.upsample(x2_2))), 1)
        x3_2 = self.conv_concat3(x3_2)
        #x3_2 = self.sap(x3_2)
        x = self.conv4(x3_2)
        x = self.conv5(x)
        return x

class KingMSEG(nn.Module):
    def __init__(self, class_num=1):
        super(KingMSEG, self).__init__()
        # ---- Partial Decoder ----
        channel = 32
        self.agg1 = aggregation_base(channel, class_num)
        self.upsample = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.kingnet = kingnet53(arch=53, depth_wise=False, pretrained = True)
        
        self.rfb2_1 = RFB_modified(540, channel)
        self.rfb3_1 = RFB_modified(800, channel)
        self.rfb4_1 = RFB_modified(1200, channel)
    
    def forward(self, x):
        kingnetout = self.kingnet(x)
        x1 = kingnetout[0]
        x2 = kingnetout[1]
        x3 = kingnetout[2]
        x4 = kingnetout[3]

        x2_moduled = self.rfb2_1(x2)
        x3_moduled = self.rfb3_1(x3)
        x4_moduled = self.rfb4_1(x4)
        ra5_feat = self.agg1(x4_moduled, x3_moduled, x2_moduled)
        
        lateral_map_5 = F.interpolate(ra5_feat, scale_factor=16, mode='bilinear')
        
        return lateral_map_5

# w/o deep supervision and boundary
class KingMSEG_lawin(nn.Module):
    def __init__(self, class_num=1):
        super(KingMSEG_lawin, self).__init__()
        # ---- Partial Decoder ----
        channel = 32
        self.agg1 = LawinHead(in_channels=[140, 540, 800, 1200])
        self.upsample = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.kingnet = kingnet53(arch=53, depth_wise=False, pretrained = True)
    
    def forward(self, x):
        kingnetout = self.kingnet(x)
        x1 = kingnetout[0]
        x2 = kingnetout[1]
        x3 = kingnetout[2]
        x4 = kingnetout[3]

        ra5_feat = self.agg1(x1, x2, x3, x4)
        lateral_map_5 = F.interpolate(ra5_feat, size=x.size()[2:], mode='bilinear')
        
        return lateral_map_5

# w/ deep1 and boundary
class KingMSEG_lawin_loss(nn.Module):
    def __init__(self, class_num=1):
        super(KingMSEG_lawin_loss, self).__init__()
        self.backbone = kingnet53(arch=53, depth_wise=False, pretrained = True)
        self.head = LawinHead2(in_channels=[140, 540, 800, 1200], num_classes = class_num)
        self.last3_seg = nn.Conv2d(512, class_num, kernel_size=1)
    
    def forward(self, x):
        kingnetout = self.backbone(x)
        x_4 = kingnetout[0]
        x_8 = kingnetout[1]
        x_16 = kingnetout[2]
        x_32 = kingnetout[3]

        output, last3_feat, low_level_feat = self.head(x_4, x_8, x_16, x_32)
        output = F.interpolate(output, size=x.size()[2:], mode='bilinear')    
        
        if self.training:
            last3_feat = F.interpolate(self.last3_seg(last3_feat), size=x.size()[2:], mode='bilinear')
            low_level_feat = F.interpolate(low_level_feat, size=x.size()[2:], mode='bilinear')
            return output, last3_feat, low_level_feat

        return output

# w/ deep1 and deep2
class KingMSEG_lawin_loss2(nn.Module):
    def __init__(self, class_num=1):
        super(KingMSEG_lawin_loss2, self).__init__()
        self.backbone = kingnet53(arch=53, depth_wise=False, pretrained = True)

        outch = self.backbone(torch.zeros(1, 3, 512, 512))[-4:]
        outch = [x.size(1) for x in outch]
        #print('Encoder ch: ', outch)
        self.head = LawinHead3(in_channels=outch, num_classes=class_num)
        self.last3_seg = nn.Conv2d(512, class_num, kernel_size=1)
        self.last3_seg2 = nn.Conv2d(768, class_num, kernel_size=1)
    
    def forward(self, x):
        x_4, x_8, x_16, x_32 = self.backbone(x)[-4:]
        
        output, last3_feat, last3_feat2 = self.head(x_4, x_8, x_16, x_32)
        output = F.interpolate(output, size=x.size()[2:], mode='bilinear')    
        
        if self.training:
            last3_feat = F.interpolate(self.last3_seg(last3_feat), size=x.size()[2:], mode='bilinear')
            last3_feat2 = F.interpolate(self.last3_seg2(last3_feat2), size=x.size()[2:], mode='bilinear')

            return output, last3_feat, last3_feat2

        return output

# w/ deep1
class KingMSEG_lawin_loss3(nn.Module):
    def __init__(self, class_num=1):
        super(KingMSEG_lawin_loss3, self).__init__()
        self.backbone = kingnet53(arch=53, depth_wise=False, pretrained = True)

        outch = self.backbone(torch.zeros(1, 3, 512, 512))[-4:]
        outch = [x.size(1) for x in outch]
        self.head = LawinHead4(in_channels=outch, num_classes=class_num)
        self.last3_seg = nn.Conv2d(512, class_num, kernel_size=1)
    
    def forward(self, x):
        x_4, x_8, x_16, x_32 = self.backbone(x)[-4:]
        
        output, last3_feat = self.head(x_4, x_8, x_16, x_32)
        output = F.interpolate(output, size=x.size()[2:], mode='bilinear')    
        
        if self.training:
            last3_feat = F.interpolate(self.last3_seg(last3_feat), size=x.size()[2:], mode='bilinear')

            return output, last3_feat

        return output

# w/ deep1, deep2 and boundary
class KingMSEG_lawin_loss4(nn.Module):
    def __init__(self, class_num=1):
        super(KingMSEG_lawin_loss4, self).__init__()
        self.backbone = kingnet53(arch=53, depth_wise=False, pretrained = True)
            
        outch = self.backbone(torch.zeros(1, 3, 512, 512))[-4:]
        outch = [x.size(1) for x in outch]
        #print('Encoder ch: ', outch)
        self.head = LawinHead5(in_channels=outch, num_classes=class_num)
        self.last3_seg = nn.Conv2d(512, class_num, kernel_size=1)
        self.last3_seg2 = nn.Conv2d(768, class_num, kernel_size=1)
    
    def forward(self, x):
        x_4, x_8, x_16, x_32 = self.backbone(x)[-4:]
        
        output, last3_feat, last3_feat2, low_level_feat = self.head(x_4, x_8, x_16, x_32)
        output = F.interpolate(output, size=x.size()[2:], mode='bilinear')    
        
        if self.training:
            last3_feat = F.interpolate(self.last3_seg(last3_feat), size=x.size()[2:], mode='bilinear')
            last3_feat2 = F.interpolate(self.last3_seg2(last3_feat2), size=x.size()[2:], mode='bilinear')
            low_level_feat = F.interpolate(low_level_feat, size=x.size()[2:], mode='bilinear')

            return output, last3_feat, last3_feat2, low_level_feat

        return output

class CSPKingMSEG(nn.Module):
    def __init__(self, class_num=1):
        super(CSPKingMSEG, self).__init__()
        # ---- Partial Decoder ----
        channel = 32
        self.agg1 = aggregation_base(channel, class_num)
        self.upsample = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.kingnet = CSPkingnet53(arch=53, depth_wise=False, pretrained = True)
        
        self.rfb2_1 = RFB_modified(540, channel)
        self.rfb3_1 = RFB_modified(800, channel)
        self.rfb4_1 = RFB_modified(1200, channel)
    
    def forward(self, x):
        kingnetout = self.kingnet(x)
        x1 = kingnetout[0]
        x2 = kingnetout[1]
        x3 = kingnetout[2]
        x4 = kingnetout[3]
        
        x2_moduled = self.rfb2_1(x2)
        x3_moduled = self.rfb3_1(x3)
        x4_moduled = self.rfb4_1(x4)
        
        ra5_feat = self.agg1(x4_moduled, x3_moduled, x2_moduled)
        lateral_map_5 = F.interpolate(ra5_feat, scale_factor=8, mode='bilinear')
        
        return lateral_map_5
