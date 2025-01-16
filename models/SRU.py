import torch
import torch.nn.functional as F
import torch.nn as nn

class GroupBatchnorm2d(nn.Module):
    def __init__(self, c_num: int,
                 group_num: int = 16,
                 eps: float = 1e-10
                 ):
        super(GroupBatchnorm2d, self).__init__()
        assert c_num >= group_num
        self.group_num = group_num
        self.weight = nn.Parameter(torch.randn(c_num, 1, 1))
        self.bias = nn.Parameter(torch.zeros(c_num, 1, 1))
        self.eps = eps

    def forward(self, x):
        N, C, H, W = x.size()
        x = x.view(N, self.group_num, -1)
        mean = x.mean(dim=2, keepdim=True)
        std = x.std(dim=2, keepdim=True)
        x = (x - mean) / (std + self.eps)
        x = x.view(N, C, H, W)
        return x * self.weight + self.bias

class CRU(nn.Module):
    '''
    alpha: 0<alpha<1
    '''

    def __init__(self,
                 op_channel: int,
                 alpha: float = 1,
                 squeeze_radio: int = 1,
                 group_size: int = 1,
                 group_kernel_size: int = 3,
                 ):
        super().__init__()
        self.up_channel = up_channel = int(alpha * op_channel)
        self.low_channel = low_channel = op_channel - up_channel
        self.squeeze1 = nn.Conv2d(up_channel, up_channel // squeeze_radio, kernel_size=1, bias=False)
        self.squeeze2 = nn.Conv2d(low_channel, low_channel // squeeze_radio, kernel_size=1, bias=False)
        # up
        self.GWC = nn.Conv2d(up_channel // squeeze_radio, up_channel, kernel_size=group_kernel_size, stride=1,
                             padding=group_kernel_size // 2, groups=group_size)
        self.PWC1 = nn.Conv2d(up_channel // squeeze_radio, up_channel, kernel_size=1, bias=False)
        # low
        self.PWC2 = nn.Conv2d(up_channel // squeeze_radio, up_channel , kernel_size=1,
                              bias=False)
        self.advavg = nn.AdaptiveAvgPool2d(1)


    def forward(self, x):
        # Split

        Y1 = self.GWC(x)


        Y2=self.PWC2(x)


        return Y1,Y2

if __name__ == '__main__':
    # x = torch.randn(1, 32, 16, 16)
    x = torch.randn(16, 4, 16, 256)
    model = CRU(4)
    x1 = model(x,x)
    # x = torch.unsqueeze(x[:, 0], 1)
    # print(type(x))
    print(x.shape)
