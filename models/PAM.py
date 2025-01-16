import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from models.coordatt import *

class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True,
                 bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class ChannelGate(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max']):
        super(ChannelGate, self).__init__()
        self.gate_channels = gate_channels
        self.mlp = nn.Sequential(
            Flatten(),
            nn.Linear(gate_channels, gate_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(gate_channels // reduction_ratio, gate_channels)
        )
        self.pool_types = pool_types

    def forward(self, x):
        channel_att_sum = None
        for pool_type in self.pool_types:
            if pool_type == 'avg':
                avg_pool = F.avg_pool2d(x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp(avg_pool)
            elif pool_type == 'max':
                max_pool = F.max_pool2d(x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp(max_pool)
            elif pool_type == 'lp':
                lp_pool = F.lp_pool2d(x, 2, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp(lp_pool)
            elif pool_type == 'lse':
                # LSE pool only
                lse_pool = logsumexp_2d(x)
                channel_att_raw = self.mlp(lse_pool)

            if channel_att_sum is None:
                channel_att_sum = channel_att_raw
            else:
                channel_att_sum = channel_att_sum + channel_att_raw

        scale = F.sigmoid(channel_att_sum).unsqueeze(2).unsqueeze(3).expand_as(x)
        print(scale.shape,"scale.shape")
        return x * scale


def logsumexp_2d(tensor):
    tensor_flatten = tensor.view(tensor.size(0), tensor.size(1), -1)
    s, _ = torch.max(tensor_flatten, dim=2, keepdim=True)
    outputs = s + (tensor_flatten - s).exp().sum(dim=2, keepdim=True).log()
    return outputs


class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat((torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1)


class SpatialGate(nn.Module):
    def __init__(self):
        super(SpatialGate, self).__init__()
        kernel_size = 7
        self.compress = ChannelPool()
        self.spatial = BasicConv(2, 1, kernel_size, stride=1, padding=(kernel_size - 1) // 2, relu=False)

    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = F.sigmoid(x_out)  # broadcasting
        return x * scale


# class CBAM(nn.Module):
#     def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max'], no_spatial=False):
#         super(CBAM, self).__init__()
#         self.ChannelGate = ChannelGate(gate_channels, reduction_ratio, pool_types)
#         self.no_spatial = no_spatial
#         if not no_spatial:
#             self.SpatialGate = SpatialGate()
#
#         self.sigmoid = nn.Sigmoid()
#
#
#     def forward(self, x):
#         x_out = self.ChannelGate(x)
#         if not self.no_spatial:
#             x_out = self.SpatialGate(x_out)
#
#         x_out = self.sigmoid(x_out)
#         x_out = x_out.expand_as(x)
#         return x*x_out

class SimAM_Module(nn.Module):
    def __init__(self,in_channels,e_lambda=1e-4):
        super(SimAM_Module,self).__init__()
        self.activaon=nn.Sigmoid()
        self.e_lambda=e_lambda
        self.sigmoid = nn.Sigmoid()
        self.avgpool = nn.AvgPool2d(kernel_size=2,stride=None, padding=0)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=None, padding=0)
        self.conv = nn.Conv2d(in_channels, in_channels//2, kernel_size=1, stride=1, padding=0)
        self.upsasmple = nn.ConvTranspose2d(in_channels//2, in_channels, kernel_size=4, stride=2,padding=1)
    def __repr__(self):
        s=self.__call__.__name__+'('
        s+=('lambda=%f'% self.e_lambda)
        return s

    @staticmethod
    def get_moudle_name():
        return "simam"

    def forward(self,x):
        b,c,h,w=x.size()
        n=w*h-1
        x_max=x.max(dim=2,keepdim=True)[0].max(dim=3,keepdim=True)[0]
        x_minus_max=x-x_max
        x_minus_max_square=x_minus_max.pow(2)
        x_minus_mu_square =(x-x.mean(dim=[2,3],keepdim=True)).pow(2)
        z=x_minus_mu_square/(4*(x_minus_mu_square.sum(dim=[2,3],keepdim=True)/n+self.e_lambda))
        y=x_minus_max_square/(4*(x_minus_max_square.sum(dim=[2,3],keepdim=True)/n+self.e_lambda))
        out=x*self.activaon(y)*self.activaon(z)
        out=self.sigmoid(out)
        return out




class MCALayer(nn.Module):
    def __init__(self, inp, no_spatial=False):
        super(MCALayer,self).__init__()
        lambd = 1.5
        gamma = 1
        temp = round(abs((math.log2(inp) - gamma) / lambd))
        kernel = temp if temp % 2 else temp - 1
        self.coordatt_w=CoordAtt(4,4)
        # self.h_cw=SimAM_Module(2)
        # self.w_hc=SimAM_Module(2)
        self.no_spatial=no_spatial
        if not no_spatial:
            self.c_hw=SimAM_Module(512)
            self.coordatt_hw = CoordAtt(512, 512)



    def forward(self,x):
        x_h=x.permute(0, 2, 1, 3).contiguous()
        x_h = self.coordatt_w(x_h)
        x_h = x_h.permute(0, 2, 1, 3).contiguous()

        # x_w = x.permute(0, 3, 2, 1).contiguous()
        # x_w = self.w_hc(x_w)
        # x_w = x_w.permute(0, 3, 2, 1).contiguous()

        if not self.no_spatial:
            x_c = self.coordatt_hw(x)
            # x_out = 1 / 3 * (x_c + x_h+x_w)
        x_11, x_12 = torch.split(x_h, x_h.size(1) // 2, dim=1)
        x_21, x_22 = torch.split(x_c, x_c.size(1) // 2, dim=1)
        x_out=torch.cat([x_11 + x_22, x_12 + x_21], dim=1)

        return x_out



if __name__=='__main__':
    conv=MCALayer(512)
    input=torch.rand(64,512,2,2)
    output=conv(input)
    # output=tran1(input)
    print(input.shape)
    print(output.shape)