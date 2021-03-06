# -*- coding: utf-8 -*-
import torch.nn as nn
import torch
import torch.nn.functional as F
from .CIT import CIT


class Attention_block(nn.Module):
    def __init__(self,F_g,F_l,F_int):
        super(Attention_block,self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(F_int)
        )
        
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(1),
            # nn.Sigmoid()
            nn.ReLU(inplace=False)
        )
        
        self.bat = nn.BatchNorm2d(F_int)
        self.relu = nn.ReLU(inplace=False)
        
    def forward(self,g,x):
        # 下采样的gating signal 卷积
        g1 = self.W_g(g)
        # 上采样的 l 卷积
        x1 = self.W_x(x)
        # concat + relu
        psi = self.relu(g1+x1)
        # channel 减为1，并Sigmoid,得到权重矩阵
        psi = self.psi(psi)
		# 返回加权的 x
        return x*psi
        # return psi

def get_activation(activation_type):
    activation_type = activation_type.lower()
    if hasattr(nn, activation_type):
        return getattr(nn, activation_type)()
    else:
        return nn.ReLU()

def _make_nConv(in_channels, out_channels, nb_Conv, activation='ReLU'):
    layers = []
    layers.append(ConvBatchNorm(in_channels, out_channels, activation))

    for _ in range(nb_Conv - 1):
        layers.append(ConvBatchNorm(out_channels, out_channels, activation))
    return nn.Sequential(*layers)

class ConvBatchNorm(nn.Module):
    """(convolution => [BN] => ReLU)"""

    def __init__(self, in_channels, out_channels, activation='ReLU'):
        super(ConvBatchNorm, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels,
                              kernel_size=3, padding=1)
        self.norm = nn.BatchNorm2d(out_channels)
        self.activation = get_activation(activation)

    def forward(self, x):
        out = self.conv(x)
        out = self.norm(out)
        return self.activation(out)

class DownBlock(nn.Module):
    """Downscaling with maxpool convolution"""
    def __init__(self, in_channels, out_channels, nb_Conv, activation='ReLU'):
        super(DownBlock, self).__init__()
        self.maxpool = nn.MaxPool2d(2)
        self.avgpool = nn.AvgPool2d(2)#############
        self.nConvs = _make_nConv(in_channels, out_channels, nb_Conv, activation)

    def forward(self, x):
        # if x.shape[1]>256:
        #     out = self.avgpool(x)
        # else:
        out = self.maxpool(x)
        return self.nConvs(out)


class MDC(nn.Module):
    def __init__(self, out_channels, activation='ReLU'):
        super().__init__()
        self.cell4 = MDCCell(512, out_channels, activation='ReLU')
        self.cell3 = MDCCell(256, out_channels, activation='ReLU')
        self.cell2 = MDCCell(128, out_channels, activation='ReLU')
        self.cell1 = MDCCell(64, out_channels, activation='ReLU')
        self.up2 = nn.Upsample(scale_factor=2)
        self.up3 = nn.Upsample(scale_factor=4)
        self.up4 = nn.Upsample(scale_factor=8)
        self.activation = get_activation(activation)

    def forward(self, x1,x2,x3,x4):
        x4=self.cell4(x4)
        x4=self.up4(x4)
        x3=self.cell3(x3)
        x3=self.up3(x3)
        x2=self.cell2(x2)
        x2=self.up2(x2)
        x1=self.cell1(x1)
        x=torch.cat([x1, x2, x3, x4], dim=1) 

        return x


class MDCCell(nn.Module):
    def __init__(self, in_channels, out_channels, activation='ReLU'):
        super().__init__()
        self.mixlist=nn.ModuleList()
        # kernel_size=1
        self.mixlist.append(nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0, dilation=1))
        self.mixlist.append(nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0, dilation=6))
        self.mixlist.append(nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0, dilation=12))
        # kernel_size=3
        self.mixlist.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, dilation=1))
        self.mixlist.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=6, dilation=6))
        self.mixlist.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=12, dilation=12))
        # kernel_size=5
        self.mixlist.append(nn.Conv2d(in_channels, out_channels, kernel_size=5, padding=2, dilation=1))
        self.mixlist.append(nn.Conv2d(in_channels, out_channels, kernel_size=5, padding=12, dilation=6))
        self.mixlist.append(nn.Conv2d(in_channels, out_channels, kernel_size=5, padding=24, dilation=12))
        self.conv= nn.Sequential(
            nn.Conv2d(out_channels*9, out_channels, kernel_size=1, padding=0, dilation=1),
            nn.BatchNorm2d(out_channels)
        )
        # self.activation = get_activation(activation)
        self.bn = nn.BatchNorm2d(out_channels*9)

    def forward(self, x):
        xlist = []
        for convmix in self.mixlist:
            xlist.append(convmix(x))

        x=torch.cat(xlist, dim=1)
        x = self.bn(x)
        return self.conv(x)

class UpBlock_attention(nn.Module):
    def __init__(self, in_channels, activation='ReLU'):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2)
        # self.coatt = CCA(F_g=in_channels//2, F_x=in_channels//2)
        self.Att = Attention_block(F_g=in_channels*2,F_l=in_channels,F_int=in_channels)
        # self.nConvs = _make_nConv(in_channels, out_channels, nb_Conv, activation)
        self.conv = ConvBatchNorm(in_channels*3,in_channels, activation)

    def forward(self, x, skip_x):
        up = self.up(x)
        skip_x = self.Att(g=up, x=skip_x)
        x = torch.cat([skip_x, up], dim=1)  # dim 1 is the channel dimension
        return self.conv(x)

class LSUnetMix(nn.Module):
    def __init__(self, config,n_channels=3, n_classes=1,img_size=224):
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        in_channels = config.base_channel
        self.inc = ConvBatchNorm(n_channels, in_channels)
        self.down1 = DownBlock(in_channels, in_channels*2, nb_Conv=2, activation='Relu')
        self.down2 = DownBlock(in_channels*2, in_channels*4, nb_Conv=2, activation='Relu')
        self.down3 = DownBlock(in_channels*4, in_channels*8, nb_Conv=2, activation='Relu')
        self.down4 = DownBlock(in_channels*8, in_channels*16, nb_Conv=2, activation='Relu')
        self.cit = CIT(config, img_size,channel_num=[in_channels, in_channels*2, in_channels*4, in_channels*8],
                                     patchSize=config.patch_sizes)
        self.up4 = UpBlock_attention(in_channels*8, activation='Relu')
        self.up3 = UpBlock_attention(in_channels*4, activation='Relu')
        self.up2 = UpBlock_attention(in_channels*2, activation='Relu')
        self.up1 = UpBlock_attention(in_channels, activation='Relu')


        self.outc = nn.Conv2d(96, n_classes, kernel_size=(1,1), stride=(1,1))
        self.last_activation = nn.Sigmoid() # if using BCELoss
        #######################
        # self.conv1 = ConvBatchNorm(in_channels, 1)
        self.convMix = MDC(out_channels = 8)

    def forward(self, x):
        x = x.float()
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x1,x2,x3,x4= self.cit(x1,x2,x3,x4)
        y = self.convMix(x1,x2,x3,x4)
        x = self.up4(x5, x4)
        x = self.up3(x, x3)
        x = self.up2(x, x2)
        x = self.up1(x, x1)
        
        # x = self.conv1(x)
        x = torch.cat([y,x], dim=1)
        if self.n_classes ==1:
            logits = self.last_activation(self.outc(x))
        else:
            logits = self.outc(x) # if nusing BCEWithLogitsLoss or class>1

        return logits




