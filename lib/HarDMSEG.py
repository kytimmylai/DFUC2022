#!/usr/bin/env python
# coding: utf-8
# %%
from operator import imod
import torch
import torch.nn as nn
import torch.nn.functional as F
from .kingnet import kingnet53, KingBlock
from .kingnet2 import DCSPKingBlock, ConvBnAct, SELayer, CSPkingnet53
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
        #return self.relu(self.bn(self.conv(x)))
        
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

class aggregation_mod(nn.Module):
    def __init__(self, channel, class_num = 1):
        super(aggregation_mod, self).__init__()
        
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.upsample4 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
            
        self.conv_upsample1 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample2 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample3 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample4 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample5 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample6 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample7 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample8 = BasicConv2d(2*channel, 2*channel, 3, padding=1)
        self.conv_upsample9 = BasicConv2d(3*channel, 3*channel, 3, padding=1)

        self.conv_concat2 = BasicConv2d(2*channel, 2*channel, 3, padding=1)
        self.conv_concat3 = BasicConv2d(3*channel, 3*channel, 3, padding=1)
        self.conv_concat4 = BasicConv2d(4*channel, 4*channel, 3, padding=1)
        self.conv4 = BasicConv2d(4*channel, 4*channel, 3, padding=1)
        
        # (input channel, class channel, kernel)
        self.conv5 = nn.Conv2d(4*channel, class_num, 1)
        
    def forward(self, x1, x2, x3, x4):
        x1_1 = self.upsample(x1)
            
        x2_1 = self.conv_upsample1(x1_1) * x2
        x2_2 = torch.cat((x2_1, self.conv_upsample7(x1_1)), 1)
        x2_2 = self.conv_concat2(x2_2)
            
        x3_1 = self.conv_upsample2(self.upsample(x1_1))* self.conv_upsample3(self.upsample(x2)) * x3
        x3_2 = torch.cat((x3_1, self.conv_upsample8(self.upsample(x2_2))), 1)
        x3_2 = self.conv_concat3(x3_2)

        x4_1 = self.conv_upsample4(self.upsample4(x1_1))* self.conv_upsample5(self.upsample4(x2)) * self.conv_upsample6(self.upsample(x3)) * x4
        x4_2 = torch.cat((x4_1, self.conv_upsample9(self.upsample(x3_2))), 1)
        x4_2 = self.conv_concat4(x4_2)

        x = self.conv4(x4_2)
        x = self.conv5(x)
        return x

class aggregation_mod2(nn.Module): # layer: 0,4,12,16,20
    def __init__(self, channel, class_num = 1):
        super(aggregation_mod2, self).__init__()
        
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.upsample4 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.upsample8 = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)
            
        self.conv_upsample1 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample2 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample3 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample4 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample5 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample6 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample7 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample8 = BasicConv2d(2*channel, 2*channel, 3, padding=1)
        self.conv_upsample9 = BasicConv2d(3*channel, 3*channel, 3, padding=1)
        self.conv_upsample10 = BasicConv2d(4*channel, 4*channel, 3, padding=1)

        self.conv_concat2 = BasicConv2d(2*channel, 2*channel, 3, padding=1)
        self.conv_concat3 = BasicConv2d(3*channel, 3*channel, 3, padding=1)
        self.conv_concat4 = BasicConv2d(4*channel, 4*channel, 3, padding=1)
        self.conv_concat5 = BasicConv2d(5*channel, 5*channel, 3, padding=1)
        self.conv4 = BasicConv2d(5*channel, 5*channel, 3, padding=1)
        
        # (input channel, class channel, kernel)
        self.conv5 = nn.Conv2d(5*channel, class_num, 1)
        
    def forward(self, x1, x2, x3, x4, x5):
        x1_1 = self.upsample(x1)
            
        x2_1 = self.conv_upsample1(x1_1) * x2
        x2_2 = torch.cat((x2_1, self.conv_upsample7(x1_1)), 1)
        x2_2 = self.conv_concat2(x2_2)
            
        x3_1 = self.conv_upsample2(self.upsample(x1_1))* self.conv_upsample3(self.upsample(x2)) * x3
        x3_2 = torch.cat((x3_1, self.conv_upsample8(self.upsample(x2_2))), 1)
        x3_2 = self.conv_concat3(x3_2)

        x4_1 = self.conv_upsample4(self.upsample4(x1_1))* self.conv_upsample5(self.upsample4(x2)) * self.conv_upsample6(self.upsample(x3)) * x4
        x4_2 = torch.cat((x4_1, self.conv_upsample9(self.upsample(x3_2))), 1)
        x4_2 = self.conv_concat4(x4_2)

        x5_1 = self.conv_upsample4(self.upsample8(x1_1))* self.conv_upsample5(self.upsample8(x2)) * self.conv_upsample6(self.upsample4(x3)) * self.conv_upsample6(self.upsample(x4)) * x5
        x5_2 = torch.cat((x5_1, self.conv_upsample10(self.upsample(x4_2))), 1)
        x5_2 = self.conv_concat5(x5_2)

        x = self.conv4(x5_2)
        x = self.conv5(x)
        return x

class aggregation_mod3(nn.Module): # layer: 4,8,12,16,20
    def __init__(self, channel, class_num = 1):
        super(aggregation_mod3, self).__init__()
        
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.upsample4 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.upsample8 = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)
            
        self.conv_upsample1 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample2 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample3 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample4 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample5 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample6 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample7 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample8 = BasicConv2d(2*channel, 2*channel, 3, padding=1)
        self.conv_upsample9 = BasicConv2d(3*channel, 3*channel, 3, padding=1)
        self.conv_upsample10 = BasicConv2d(4*channel, 4*channel, 3, padding=1)

        self.conv_concat2 = BasicConv2d(2*channel, 2*channel, 3, padding=1)
        self.conv_concat3 = BasicConv2d(3*channel, 3*channel, 3, padding=1)
        self.conv_concat4 = BasicConv2d(4*channel, 4*channel, 3, padding=1)
        self.conv_concat5 = BasicConv2d(5*channel, 5*channel, 3, padding=1)
        self.conv4 = BasicConv2d(5*channel, 5*channel, 3, padding=1)
        
        # (input channel, class channel, kernel)
        self.conv5 = nn.Conv2d(5*channel, class_num, 1)
        
    def forward(self, x1, x2, x3, x4, x5):
        x1_1 = self.upsample(x1)
            
        x2_1 = self.conv_upsample1(x1_1) * x2
        x2_2 = torch.cat((x2_1, self.conv_upsample7(x1_1)), 1)
        x2_2 = self.conv_concat2(x2_2)
            
        x3_1 = self.conv_upsample2(self.upsample(x1_1))* self.conv_upsample3(self.upsample(x2)) * x3
        x3_2 = torch.cat((x3_1, self.conv_upsample8(self.upsample(x2_2))), 1)
        x3_2 = self.conv_concat3(x3_2)
        
        x4_1 = self.conv_upsample4(self.upsample(x1_1))* self.conv_upsample5(self.upsample(x2)) * self.conv_upsample6(x3) * x4
        x4_2 = torch.cat((x4_1, self.conv_upsample9(x3_2)), 1)
        x4_2 = self.conv_concat4(x4_2)
        
        x5_1 = self.conv_upsample4(self.upsample4(x1_1))* self.conv_upsample5(self.upsample4(x2)) * self.conv_upsample6(self.upsample(x3)) * self.conv_upsample6(self.upsample(x4)) * x5
        x5_2 = torch.cat((x5_1, self.conv_upsample10(self.upsample(x4_2))), 1)
        x5_2 = self.conv_concat5(x5_2)

        x = self.conv4(x5_2)
        x = self.conv5(x)
        return x

# %%
class IFMa(nn.Module): # =s1
    def __init__(self,
                 inplanes):
        super(IFMa, self).__init__()
        self.inplanes = inplanes
        self.conv_mask = nn.Conv2d(self.inplanes, 1, kernel_size=1)
#   
        # reset_parameters
#         kaiming_init(self.conv_mask, mode='fan_in')   
        if hasattr(self.conv_mask, 'weight') and self.conv_mask.weight is not None:
            nn.init.kaiming_normal_(self.conv_mask.weight, a=0, mode='fan_in', nonlinearity='relu')
        if hasattr(self.conv_mask, 'bias') and self.conv_mask.bias is not None:
            nn.init.constant_(self.conv_mask.bias, 0)
        
        self.conv_mask.inited = True
        #--       
        
    def forward(self, x):
        context_mask = self.conv_mask(x)
        context_mask_ = torch.sigmoid(context_mask)
        out_x = torch.mul(x ,context_mask_)
#         print('out_x.shape:',out_x.shape)
        out = x + out_x 
        
        if self.training:
#             print('hi self.training')
            return [out, context_mask]
#         print('hi eval')
        return out #,context_mask,height , width

class BNReLU(nn.Module):
    def __init__(self, nIn):
        super().__init__()
        self.bn = nn.BatchNorm2d(nIn, eps=1e-3)
        self.acti = Mish()

    def forward(self, input):
        output = self.bn(input)
        output = self.acti(output)
        
        return output

class CFPModule(nn.Module):
    def __init__(self, nIn, d=1, KSize=3,dkSize=3):
        super().__init__()
        
        self.bn_relu_1 = BNReLU(nIn)
        self.bn_relu_2 = BNReLU(nIn)
        self.conv1x1_1 = BasicConv2d(nIn, nIn//4, KSize, 1, padding=1)
        
        self.dconv_4_1 = BasicConv2d(nIn //4, nIn//16, (dkSize,dkSize),1,padding = (1*d+1,1*d+1),
                            dilation=d+1)
        self.dconv_4_2 = BasicConv2d(nIn //16, nIn//16, (dkSize,dkSize),1,padding = (1*d+1,1*d+1),
                            dilation=d+1,)
        self.dconv_4_3 = BasicConv2d(nIn //16, nIn //8, (dkSize,dkSize),1,padding = (1*d+1,1*d+1),
                            dilation=d+1)
        
        self.dconv_1_1 = BasicConv2d(nIn //4, nIn //16, (dkSize,dkSize),1,padding = (1,1),
                            dilation=1)
        self.dconv_1_2 = BasicConv2d(nIn //16, nIn //16, (dkSize,dkSize),1,padding = (1,1),
                            dilation=1)
        self.dconv_1_3 = BasicConv2d(nIn //16, nIn //8, (dkSize,dkSize),1,padding = (1,1),
                            dilation=1)
        
        self.dconv_2_1 = BasicConv2d(nIn //4, nIn //16, (dkSize,dkSize),1,padding = (int(d/4+1),int(d/4+1)),
                            dilation=int(d/4+1))
        self.dconv_2_2 = BasicConv2d(nIn //16, nIn //16, (dkSize,dkSize),1,padding = (int(d/4+1),int(d/4+1)),
                            dilation=int(d/4+1))
        self.dconv_2_3 = BasicConv2d(nIn //16, nIn //8, (dkSize,dkSize),1,padding = (int(d/4+1),int(d/4+1)),
                            dilation=int(d/4+1))
        
        self.dconv_3_1 = BasicConv2d(nIn //4, nIn //16, (dkSize,dkSize),1,padding = (int(d/2+1),int(d/2+1)),
                            dilation=int(d/2+1))
        self.dconv_3_2 = BasicConv2d(nIn //16, nIn //16, (dkSize,dkSize),1,padding = (int(d/2+1),int(d/2+1)),
                            dilation=int(d/2+1))
        self.dconv_3_3 = BasicConv2d(nIn //16, nIn //8, (dkSize,dkSize),1,padding = (int(d/2+1),int(d/2+1)),
                            dilation=int(d/2+1))
        
        self.conv1x1 = BasicConv2d(nIn, nIn, 1, 1, padding=0)

    def forward(self, input):
        inp = self.bn_relu_1(input)
        inp = self.conv1x1_1(inp)
        
        o1_1 = self.dconv_1_1(inp)
        o1_2 = self.dconv_1_2(o1_1)
        o1_3 = self.dconv_1_3(o1_2)
        
        o2_1 = self.dconv_2_1(inp)
        o2_2 = self.dconv_2_2(o2_1)
        o2_3 = self.dconv_2_3(o2_2)
        
        o3_1 = self.dconv_3_1(inp)
        o3_2 = self.dconv_3_2(o3_1)
        o3_3 = self.dconv_3_3(o3_2)
        
        o4_1 = self.dconv_4_1(inp)
        o4_2 = self.dconv_4_2(o4_1)
        o4_3 = self.dconv_4_3(o4_2)
        
        output_1 = torch.cat([o1_1,o1_2,o1_3], 1)
        output_2 = torch.cat([o2_1,o2_2,o2_3], 1)      
        output_3 = torch.cat([o3_1,o3_2,o3_3], 1)       
        output_4 = torch.cat([o4_1,o4_2,o4_3], 1)   

        output = torch.cat([output_1, output_2, output_3, output_4],1)
        output = self.bn_relu_2(output)
        output = self.conv1x1(output)
        
        return output+input

class ConvBnRelu(nn.Module):
    def __init__(self, in_planes, out_planes, ksize, stride, pad, dilation=1,
                 groups=1, has_bn=True, norm_layer=nn.BatchNorm2d,
                 has_relu=True, inplace=True, has_bias=False):
        super(ConvBnRelu, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=ksize,
                              stride=stride, padding=pad,
                              dilation=dilation, groups=groups, bias=has_bias)
        self.has_bn = has_bn
        if self.has_bn:
            self.bn = nn.BatchNorm2d(out_planes)
        self.has_relu = has_relu
        if self.has_relu:
            self.relu = nn.ReLU(inplace=inplace)

    def forward(self, x):
        x = self.conv(x)
        if self.has_bn:
            x = self.bn(x)
        if self.has_relu:
            x = self.relu(x)

        return x

class agg(nn.Module):
    def __init__(self, channel, class_num = 1):
        super(agg, self).__init__()
        
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            
        self.conv_upsample1 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample2 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample3 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample4 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample5 = BasicConv2d(2*channel, 2*channel, 3, padding=1)

        self.conv_concat2 = BasicConv2d(2*channel, 2*channel, 3, padding=1)

        self.conv4 = BasicConv2d(2*channel, 2*channel, 3, padding=1)
        self.conv5 = nn.Conv2d(2*channel, class_num, 1)
        
    def forward(self, x1, x2):
        x1_1 = self.upsample(x1)
        
        x2_1 = self.conv_upsample1(x1_1) * x2
        x2_2 = torch.cat((x2_1, self.conv_upsample4(x1_1)), 1)
        x2_2 = self.conv_concat2(x2_2)
            
        x = self.conv4(x2_2)
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
    def __init__(self, class_num=1, arch=53, name='base'):
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

class KingMSEG_INFOMAXAFTER(nn.Module):
    def __init__(self, class_num=1):
        super(KingMSEG_INFOMAXAFTER, self).__init__()
        # ---- Partial Decoder ----
        channel = 32
        self.agg1 = aggregation_base(channel, class_num)
        self.upsample = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.kingnet = kingnet53(arch=53, depth_wise=False, pretrained = True)
        
        self.rfb2_1 = RFB_modified(540, channel)
        self.rfb3_1 = RFB_modified(800, channel)
        self.rfb4_1 = RFB_modified(1200, channel)

        self.ifma_x2 = IFMa(channel)
        self.ifma_x3 = IFMa(channel)
        self.ifma_x4 = IFMa(channel)
    
    def forward(self, x):
        kingnetout = self.kingnet(x)
        x1 = kingnetout[0]
        x2 = kingnetout[1]
        x3 = kingnetout[2]
        x4 = kingnetout[3]

        ifm_list = []

        x2_moduled = self.rfb2_1(x2)
        x3_moduled = self.rfb3_1(x3)
        x4_moduled = self.rfb4_1(x4)

        if self.training:
            x2_moduled,ifm2 = self.ifma_x2(x2_moduled)        
            x3_moduled,ifm3 = self.ifma_x3(x3_moduled)        
            x4_moduled,ifm4 = self.ifma_x4(x4_moduled) 

            ifm_list.append(ifm2)
            ifm_list.append(ifm3)
            ifm_list.append(ifm4)
        else:
            x2_moduled = self.ifma_x2(x2_moduled)        
            x3_moduled = self.ifma_x3(x3_moduled)        
            x4_moduled = self.ifma_x4(x4_moduled) 
        
        ra5_feat = self.agg1(x4_moduled, x3_moduled, x2_moduled)
        lateral_map_5 = F.interpolate(ra5_feat, scale_factor=8, mode='bilinear')

        if self.training:
            return lateral_map_5,ifm_list
        
        return lateral_map_5

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
