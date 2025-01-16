import math
import sys
import os
import importlib






from models.basic_modules import *
import models.memory_v2 as Memory
# import models.memory as Memory
from models.inpaint import InpaintBlock
# from models.Transformer import *
import torch
from torch import nn
import torch.nn.functional as F
from models.UFI import *
from models.PAM import *
from models.FFS import *


def make_window(x, window_size, stride=1, padding=0):
    """
    Args:
        x: (B, W, H, C)
        window_size (int): window size

    Returns:print(x.shape,"x")

        windows: (B*N,  ws**2, C)
    """
    x = x.permute(0, 3, 1, 2).contiguous()
    B, C, W, H = x.shape
    windows = F.unfold(x, window_size, padding=padding, stride=stride)  # B, C*N, #of windows
    windows = windows.view(B, C, window_size ** 2, -1)  # B, C, ws**2, N
    windows = windows.permute(0, 3, 2, 1).contiguous().view(-1, window_size, window_size, C)  # B*N, ws**2, C

    return windows



def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


def binarize(integer, num_bits=8):
    """Turn integer tensor to binary representation.
    Args:
    integer : torch.Tensor, tensor with integers
    num_bits : Number of bits to specify the precision. Default: 8.
    Returns:
    Tensor: Binary tensor. Adds last dimension to original tensor for
    bits.
    """
    dtype = integer.type()
    exponent_bits = -torch.arange(-(num_bits - 1), 1).type(dtype)
    exponent_bits = exponent_bits.repeat(integer.shape + (1,))
    out = integer.unsqueeze(-1) / 2 ** exponent_bits
    return (out - (out % 1)) % 2


class double_conv(nn.Module):
    '''(conv => BN => ReLU) * 2'''

    def __init__(self, in_ch, out_ch, stride=1, use_se=False, bottleneck=False):
        super(double_conv, self).__init__()
        self.use_se = use_se
        self.bottleneck = bottleneck


        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=out_ch, out_channels=out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=out_ch, out_channels=out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch)
        )

        if self.use_se:
            self.se = SE(out_ch)

    def forward(self, x):
        x = self.conv1(x)
        sc = x
        x = self.conv2(x)

        if self.use_se:
            x = self.se(x)

        x += sc
        x = F.relu(x, inplace=True)
        return x


class SE(nn.Module):
    def __init__(self, in_ch):
        super(SE, self).__init__()
        self.se = nn.Sequential(
            nn.Linear(in_ch, in_ch // 8, bias=False),
            nn.ReLU(),
            nn.Linear(in_ch // 8, in_ch, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        if len(x.size()) > 3:
            x = F.adaptive_avg_pool2d(x, 1).squeeze()
        sc = x
        # print(sc.shape)
        return self.se(x) * sc + sc


class inconv(nn.Module):
    '''
    inconv only changes the number of channels
    '''

    def __init__(self, in_ch, out_ch):
        super(inconv, self).__init__()
        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x):
        x = self.conv(x)
        return x


class down(nn.Module):
    def __init__(self, in_ch, out_ch, use_se=False):
        super(down, self).__init__()
        self.mpconv = double_conv(in_ch, out_ch, stride=2, use_se=use_se)

    def forward(self, x):
        x = self.mpconv(x)
        return x


class up(nn.Module):
    def __init__(self, in_ch, out_ch, bilinear=False, op="none", use_se=False):
        super(up, self).__init__()
        self.bilinear = bilinear
        self.op = op
        self.mixup_ratio = 0.95
        assert op in ["concat", "none", "add", 'mix']

        self.conv = double_conv(in_ch, out_ch, use_se=use_se)

    def forward(self, x1, x2=None):
        if x2 is not None:
            if torch.is_tensor(x2):
                x1 = F.interpolate(x1, x2.size()[-2:], mode='bilinear', align_corners=False)
            else:
                x1 = F.interpolate(x1, x2, mode='bilinear', align_corners=False)
        else:
            x1 = F.interpolate(x1, scale_factor=2, mode='bilinear', align_corners=False)

        if self.op == "concat":
            x = torch.cat([x2, x1], dim=1)
        elif self.op == 'add':
            x = x1 + x2
        else:
            x = x1

        x = self.conv(x)

        return x


class outconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(outconv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1)
        # self.ScConv=ScConv(32)

    def forward(self, x):
        # x = self.ScConv(x)
        x = self.conv(x)
        return x


class AE(nn.Module):
    def __init__(self, num_in_ch, features_root, shrink_thres, 
                 num_slots=200, num_patch=2, level=4, ratio=0.95,
                 drop=0.0, memory_channel=2048, dist=False, initial_combine=None, mem_num_slots=200,
                 ops=['concat', 'concat', 'none', 'none'], decoder_memory=None):
        super(AE, self).__init__()
        self.num_in_ch = num_in_ch
        self.num_slots = num_slots
        self.shrink_thres = shrink_thres
        self.initial_combine = initial_combine
        self.level = level
        self.num_patch = num_patch
        self.drop = drop

        self.SAM=SAM()
        self.CBAM_SimAM=MCALayer(512)
        self.CS= CS(512, 4, 2)

        print('SQUID ops:', ops)


        assert len(ops) == level # make sure there is an op for every decoder level

        self.filter_list = [features_root, features_root*2, features_root*4, features_root*8, features_root*16, features_root*16]
        self.size_list=[32,16,8,4]
        self.size_list_up = [128, 64, 16, 8]
        self.in_conv = inconv(num_in_ch, features_root)
        self.down_blocks = nn.ModuleList()
        self.up_blocks = nn.ModuleList()
        self.GatedCNNBlock=nn.ModuleList()




        for i in range(level):
            self.down_blocks.append(down(self.filter_list[i], self.filter_list[i+1], use_se=False))

            if ops[i] == 'concat':
                filter = self.filter_list[level-i] + self.filter_list[level-i-1]#//2
            else:
                filter = self.filter_list[level-i]
            self.up_blocks.append(up(filter, self.filter_list[level-1-i], op=ops[i], use_se=False))


        # self.inpaint_block = InpaintBlock(self.filter_list[level], num_slots, num_memory=self.num_patch**2, memory_channel=memory_channel, shrink_thres=shrink_thres, ratio=ratio, drop=drop)
        
        assert decoder_memory is not None # decoder memory should NOT be none in all cases

        self.memory_blocks = nn.ModuleList()
        for i, config in enumerate(decoder_memory):
            if config is None:
                self.memory_blocks.append(nn.Identity())
            else:
                self.memory_blocks.append(getattr(Memory, config['type'])(config['dim'],mem_num_slots,
                                                                          self.filter_list[i] * config['multiplier'],
                                                                          num_memory=config['num_memory'],
                                                                          shrink_thres=shrink_thres))

        self.out_conv = outconv(features_root, num_in_ch)








        self.mse_loss = nn.MSELoss()

        self.dist = dist
        if dist:
            self.teacher_ups = nn.ModuleList()
            for i in range(level):
                if ops[i] == 'concat':
                    filter = self.filter_list[level-i] + self.filter_list[level-i-1]
                else:
                    filter = self.filter_list[level-i]
                self.teacher_ups.append(up(filter, self.filter_list[level-1-i], op=ops[i], use_se=False))
            self.teacher_out = outconv(features_root, num_in_ch)

    def forward(self, x):
        """
        :param x: size [bs,C,H,W]
        :return:
        """
        bs, C, W, H = x.size()
        assert W % self.num_patch == 0 or  H % self.num_patch == 0
        # print(x.shape,"ok")

        # segment patches
        x = make_window(x.permute(0, 2, 3, 1).contiguous(), W//self.num_patch, H//self.num_patch, 0).permute(0, 3, 1, 2) # B * 9, C, ws, ws
        # print(x.shape,"x+x")
        num_windows = x.size(0) // bs


        x = self.in_conv(x)



        skips = []
        # encoding
        for i in range(self.level):
            B_, c, w, h = x.size()
            if i < self.initial_combine:
                sc = window_reverse(x.permute(0, 2, 3, 1).contiguous(), w, w * self.num_patch, w * self.num_patch).permute(0, 3, 1, 2)
                # print(sc.shape,"sc")
                skips.append(sc)
            else:
                # print(x.shape,"x111")
                skips.append(x)

            x = self.down_blocks[i](x)
            # print(x.shape,"x_d",i)



        B_, c, w, h = x.shape

        # this is useless currently, but could be useful in the future
        embedding = window_reverse(x.permute(0, 2, 3, 1).contiguous(), w, w * self.num_patch, w * self.num_patch).permute(0, 3, 1, 2).contiguous()
        # print(embedding.shape,"embedding.shape")


        t_x = x.clone().detach()






        x1=self.CBAM_SimAM(x)
        x2=self.CS((x1,x))
        x=self.SAM(x2,x1)


        self_dist_loss = []
        # decoding
        for i in range(self.level):
            #print(x.shape)
            # combine patches?
            if self.initial_combine is not None and self.initial_combine == (self.level - i):
                B_, c, w, h = x.shape
                x = window_reverse(x.permute(0, 2, 3, 1).contiguous(), w, w * self.num_patch, w * self.num_patch).permute(0, 3, 1, 2)
                #print(x.shape,'??')
                t_x = window_reverse(t_x.permute(0, 2, 3, 1).contiguous(), w, w * self.num_patch, w * self.num_patch).permute(0, 3, 1, 2)

            x = self.up_blocks[i](x, skips[-1-i])
            # print(x.shape,"x",i)

            # additional decoder memory matrix
            x = self.memory_blocks[-1-i](x)




            if self.dist:
                t_x = self.teacher_ups[i](t_x, skips[-1-i].detach().clone())
                # do we need sg here? maybe not
                self_dist_loss.append(self.mse_loss(x, t_x))

        # forward teacher decoder
        if self.dist:
            self_dist_loss = torch.sum(torch.stack(self_dist_loss))
            t_x = self.teacher_out(t_x)
            t_x = torch.sigmoid(t_x)
            B_, c, w, h = t_x.shape
            if self.initial_combine is None:
                t_x = window_reverse(t_x.permute(0, 2, 3, 1).contiguous(), w, w * self.num_patch, w * self.num_patch).permute(0, 3, 1, 2)


        x = self.out_conv(x)
        x = torch.sigmoid(x)


        B_, c, w, h = x.shape

        if self.initial_combine is None:
            whole_recon = window_reverse(x.permute(0, 2, 3, 1).contiguous(), w, w * self.num_patch, w * self.num_patch).permute(0, 3, 1, 2)
        else:
            whole_recon = x
            x = make_window(x.permute(0, 2, 3, 1).contiguous(), W//self.num_patch, H//self.num_patch, 0).permute(0, 3, 1, 2)

        outs = dict(recon=whole_recon, patch=x, embedding=embedding, teacher_recon=t_x, dist_loss=self_dist_loss)
        return outs
