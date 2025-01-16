import torch
import torch.nn as nn
import torch.nn.functional as F

class GCN(nn.Module):
    def __init__(self, num_state, num_node, bias=False):
        super(GCN, self).__init__()
        self.conv1 = nn.Conv1d(num_node, num_node, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(num_state, num_state, kernel_size=1, bias=bias)

    def forward(self, x):
        h = self.conv1(x.permute(0, 2, 1)).permute(0, 2, 1)
        h = h - x
        h = self.relu(self.conv2(h))
        return h


class SAM(nn.Module):
    def __init__(self, num_in=512, edge_dim_in=512, plane_mid=16, mids=16, normalize=False):
        super(SAM, self).__init__()

        self.normalize = normalize
        self.num_s = int(plane_mid)
        self.num_n = (mids) * (mids)
        self.priors = nn.AdaptiveAvgPool2d(output_size=(mids + 2, mids + 2))

        self.conv_state = nn.Conv2d(num_in, self.num_s, kernel_size=1)
        self.conv_proj = nn.Conv2d(num_in, self.num_s, kernel_size=1)
        self.conv_edge_state = nn.Conv2d(edge_dim_in, self.num_s, kernel_size=1)

        self.gcn = GCN(num_state=self.num_s, num_node=self.num_n)
        self.conv_extend = nn.Conv2d(self.num_s*2, num_in, kernel_size=1, bias=False)

    def forward(self, x, edge):
        edge1 = F.interpolate(edge, (x.size()[-2], x.size()[-1]))

        n, c, h, w = x.size()
        edge = torch.nn.functional.softmax(edge1, dim=1)[:, 1, :, :].unsqueeze(1)

        x_state_reshaped = self.conv_state(x).view(n, self.num_s, -1)

        x_proj = self.conv_proj(x)
        x_mask = x_proj * edge

        x_anchor1 = self.priors(x_mask)
        x_anchor2 = self.priors(x_mask)[:, :, 1:-1, 1:-1].reshape(n, self.num_s, -1)
        x_anchor = self.priors(x_mask)[:, :, 1:-1, 1:-1].reshape(n, self.num_s, -1)

        x_proj_reshaped = torch.matmul(x_anchor.permute(0, 2, 1), x_proj.reshape(n, self.num_s, -1))
        x_proj_reshaped = torch.nn.functional.softmax(x_proj_reshaped, dim=1)


        edge_state = self.conv_edge_state(edge1).view(n, self.num_s, -1)

        x_rproj_reshaped = x_proj_reshaped

        x_edge_state = torch.matmul(edge_state, x_proj_reshaped.permute(0, 2, 1))

        x_n_state = torch.matmul(x_state_reshaped, x_proj_reshaped.permute(0, 2, 1))
        if self.normalize:
            x_n_state = x_n_state * (1. / x_state_reshaped.size(2))
        x_n_rel = self.gcn(x_n_state)
        x_edge_rel = self.gcn(x_edge_state)

        x_state_reshaped = torch.matmul(x_n_rel, x_rproj_reshaped)
        edge_state = torch.matmul(x_edge_rel, x_rproj_reshaped)

        x_state = x_state_reshaped.view(n, self.num_s, *x.size()[2:])
        edge_state = edge_state.view(n, self.num_s, *x.size()[2:])
        # print("x", x_state.shape)
        # print("edge", edge_state.shape)
        out =torch.cat((x_state,edge_state),dim=1)
        #print("out",out.shape)



        out = x + (self.conv_extend(out))

        return out

