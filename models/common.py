import torch
import torch.nn as nn
import torch.nn.functional as F
class GraphReasoning(nn.Module):
    """ Graph Reasoning Module for information aggregation. """

    def __init__(self, va_in, va_out, vb_in, vb_out, vc_in, vc_out, spatial_ratio, drop_rate):
        super(GraphReasoning, self).__init__()
        self.ratio = spatial_ratio
        self.va_embedding = nn.Sequential(
            nn.Conv2d(va_in, va_out, 1, bias=False),
            nn.ReLU(True),
            nn.Conv2d(va_out, va_out, 1, bias=False),
        )
        self.va_gated_b = nn.Sequential(
            nn.Conv2d(va_in, va_out, 1, bias=False),
            nn.Sigmoid()
        )
        self.va_gated_c = nn.Sequential(
            nn.Conv2d(va_in, va_out, 1, bias=False),
            nn.Sigmoid()
        )
        self.vb_embedding = nn.Sequential(
            nn.Linear(vb_in, vb_out, bias=False),
            nn.ReLU(True),
            nn.Linear(vb_out, vb_out, bias=False),
        )
        self.vc_embedding = nn.Sequential(
            nn.Linear(vc_in, vc_out, bias=False),
            nn.ReLU(True),
            nn.Linear(vc_out, vc_out, bias=False),
        )
        self.unfold_b = nn.Unfold(kernel_size=spatial_ratio[0], stride=spatial_ratio[0])
        self.unfold_c = nn.Unfold(kernel_size=spatial_ratio[1], stride=spatial_ratio[1])
        self.reweight_ab = nn.Sequential(
            nn.Linear(va_out + vb_out, 1, bias=False),
            nn.ReLU(True),
            nn.Softmax(dim=1)
        )
        self.reweight_ac = nn.Sequential(
            nn.Linear(va_out + vc_out, 1, bias=False),
            nn.ReLU(True),
            nn.Softmax(dim=1)
        )
        self.reproject = nn.Sequential(
            nn.Conv2d(va_out + vb_out + vc_out, va_in, kernel_size=1, bias=False),
            nn.ReLU(True),
            nn.Conv2d(va_in, va_in, kernel_size=1, bias=False),
            nn.Dropout(drop_rate) if drop_rate is not None else nn.Identity(),
        )

    def forward(self, vert_a, vert_b, vert_c):
        emb_vert_a = self.va_embedding(vert_a)
        emb_vert_a = emb_vert_a.reshape([emb_vert_a.shape[0], emb_vert_a.shape[1], -1])

        gate_vert_b = 1 - self.va_gated_b(vert_a)
        gate_vert_b = gate_vert_b.reshape(*emb_vert_a.shape)
        gate_vert_c = 1 - self.va_gated_c(vert_a)
        gate_vert_c = gate_vert_c.reshape(*emb_vert_a.shape)

        vert_b = self.unfold_b(vert_b).reshape(
            [vert_b.shape[0], vert_b.shape[1], self.ratio[0] * self.ratio[0], -1])
        vert_b = vert_b.permute([0, 2, 3, 1])
        emb_vert_b = self.vb_embedding(vert_b)

        vert_c = self.unfold_c(vert_c).reshape(
            [vert_c.shape[0], vert_c.shape[1], self.ratio[1] * self.ratio[1], -1])
        vert_c = vert_c.permute([0, 2, 3, 1])
        emb_vert_c = self.vc_embedding(vert_c)

        agg_vb = list()
        agg_vc = list()
        for j in range(emb_vert_a.shape[-1]):
            # ab propagating
            emb_v_a = torch.stack([emb_vert_a[:, :, j]] * (self.ratio[0] ** 2), dim=1)
            emb_v_b = emb_vert_b[:, :, j, :]
            emb_v_ab = torch.cat([emb_v_a, emb_v_b], dim=-1)
            w = self.reweight_ab(emb_v_ab)
            agg_vb.append(torch.bmm(emb_v_b.transpose(1, 2), w).squeeze() * gate_vert_b[:, :, j])

            # ac propagating
            emb_v_a = torch.stack([emb_vert_a[:, :, j]] * (self.ratio[1] ** 2), dim=1)
            emb_v_c = emb_vert_c[:, :, j, :]
            emb_v_ac = torch.cat([emb_v_a, emb_v_c], dim=-1)
            w = self.reweight_ac(emb_v_ac)
            agg_vc.append(torch.bmm(emb_v_c.transpose(1, 2), w).squeeze() * gate_vert_c[:, :, j])

        agg_vert_b = torch.stack(agg_vb, dim=-1)
        agg_vert_c = torch.stack(agg_vc, dim=-1)
        agg_vert_bc = torch.cat([agg_vert_b, agg_vert_c], dim=1)
        agg_vert_abc = torch.cat([agg_vert_bc, emb_vert_a], dim=1)
        agg_vert_abc = torch.sigmoid(agg_vert_abc)
        agg_vert_abc = agg_vert_abc.reshape(vert_a.shape[0], -1, vert_a.shape[2], vert_a.shape[3])
        return self.reproject(agg_vert_abc)
class GuidedAttention(nn.Module):
    """ Reconstruction Guided Attention. """

    def __init__(self, depth=512, drop_rate=0.2):
        super(GuidedAttention, self).__init__()
        self.depth = depth
        self.gated = nn.Sequential(
            nn.Conv2d(512, 3, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(True),
            nn.Conv2d(3, 1, 1, bias=False),
            nn.Sigmoid()
        )
        self.h = nn.Sequential(
            nn.Conv2d(depth, depth, 1, 1, bias=False),
            nn.BatchNorm2d(depth),
            nn.ReLU(True),
        )
        self.dropout = nn.Dropout(drop_rate)

    def forward(self, x, pred_x, embedding):
        # print(x.shape,pred_x.shape,embedding
        #       .shape)
        residual_full = torch.abs(x - pred_x)
        # print(residual_full.shape)
        residual_x = F.interpolate(residual_full, size=embedding.shape[-2:],
                                   mode='bilinear', align_corners=True)
        # print(residual_x.shape)
        res_map = self.gated(residual_x)
        # print(res_map.shape,"ok")
        return res_map * self.h(embedding) + self.dropout(embedding)