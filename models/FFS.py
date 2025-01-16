import torch
import torch.nn as nn

import torch.nn.functional as F



class DepthWiseConv2d(nn.Module):
    def __init__(self, dim_in, dim_out, kernel_size=3, padding=1, stride=1, dilation=1):
        super().__init__()

        self.conv1 = nn.Conv2d(dim_in, dim_in, kernel_size=kernel_size, padding=padding,
                               stride=stride, dilation=dilation, groups=dim_in)
        self.norm_layer = nn.GroupNorm(4, dim_in)
        self.conv2 = nn.Conv2d(dim_in, dim_out, kernel_size=1)

    def forward(self, x):
        return self.conv2(self.norm_layer(self.conv1(x)))

class CS(nn.Module):
    def __init__(self, features, WH, r, L=32, in_c=1024, out_c=512,kernel_size=3):
        super(CS, self).__init__()
        d = max(int(features/r), L)
        self.gap = nn.AvgPool2d(int(WH))
        self.fcs = nn.ModuleList([])
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.fc = nn.Linear(int(features), d)
        for i in range(2):
            self.fcs.append(
                nn.Linear(d, features)
            )
        self.softmax = nn.Softmax(dim=1)
        self.w1 = nn.Sequential(
            DepthWiseConv2d(in_c, in_c, kernel_size, padding=(kernel_size) // 2),
            nn.Sigmoid()
        )

        self.w2 = nn.Sequential(
            DepthWiseConv2d(in_c, in_c, kernel_size + 2, padding=(kernel_size + 2) // 2),
            nn.GELU()
        )
        self.wo = nn.Sequential(
            DepthWiseConv2d(in_c, out_c, kernel_size),
            nn.GELU()
        )

        self.cw = nn.Conv2d(in_c, out_c, 1)
    def forward(self, x):
        x1, x2 = x
        # print(x1.shape,x2.shape,"x1.shape,x2.shape")

        # x = x1 + x2
        x=torch.cat((x1,x2),dim=1)
        x3,x4=self.w1(x),self.w2(x)
        x=self.wo(x3 * x4)

        # print(x1.shape,x2.shape,"x11.shape,x22.shape")
        #pdb.set_trace()
        fea_s = self.gap(x).squeeze_()
        # print(fea_s.shape,"fea_s")
        fea_z = self.fc(fea_s)
        # print(fea_z.shape,"fea_z")
        for i, fc in enumerate(self.fcs):
            vector = fc(fea_z).unsqueeze_(dim=1)
            if i == 0:
                attention_vec = vector
            else:
                attention_vec = torch.cat([attention_vec, vector], dim=1)
        attention_vec = self.softmax(attention_vec)
        # print(attention_vec.shape,"attention_vec.shape")
        attention_vec = attention_vec.unsqueeze(-1).unsqueeze(-1)
        #pdb.set_trace()
        attention_vec = attention_vec.transpose(0,1)
        # print(x1.shape,attention_vec[0].shape,"x1.shape,attention_vec[0].shape")
        out_x1 = x1 * attention_vec[0]
        # print(out_x1.shape,"out_x1.shape")
        out_x2 = x2 * attention_vec[1]
        out=torch.cat((out_x1,out_x2),dim=1)
        out=self.wo(out)
        return out


if __name__ == '__main__':
        # tran1 = TransformerBlock(4, 512)
        efm =CS(512,4,2)
        input1 = torch.rand(1, 512, 4, 4)
        input2 = torch.rand(1, 512, 4, 4)
        output1= efm((input1, input2))
        # output1,output = efm(input)
        # output=tran1(input)
        print(input1.shape)
        print(output1.shape)
        # print(output)