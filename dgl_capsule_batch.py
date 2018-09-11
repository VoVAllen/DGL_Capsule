import dgl
import torch
import torch.nn.functional as F
from torch import nn

from capsule_layer import CapsuleLayer


class DGLFeature():
    def __init__(self, tensor, pad_to):
        # self.tensor = tensor
        self.node_num = tensor.size(0)
        self.flat_tensor = tensor.contiguous().view(self.node_num, -1)
        self.node_feature_dim = self.flat_tensor.size(1)
        self.flat_pad_tensor = F.pad(self.flat_tensor, (0, pad_to - self.flat_tensor.size(1)))
        self.shape = tensor.shape

    @property
    def of(self):
        return self.flat_tensor.index_select(1, torch.arange(0, self.node_feature_dim).to("cuda")).view(self.shape)

    @property
    def nf(self):
        return self.flat_pad_tensor

    def refill_flat(self, tensor):
        self.flat_pad_tensor = tensor


class DGLBatchCapsuleLayer(CapsuleLayer):
    def __init__(self, in_unit, in_channel, num_unit, unit_size, use_routing,
                 num_routing, cuda_enabled):
        super(DGLBatchCapsuleLayer, self).__init__(in_unit, in_channel, num_unit, unit_size, use_routing,
                                                   num_routing, cuda_enabled)
        self.unit_size = unit_size
        self.weight = nn.Parameter(torch.randn(in_channel, num_unit, unit_size, in_unit))

    def routing(self, x):

        batch_size = x.size(0)

        self.g = dgl.DGLGraph()

        self.g.add_nodes_from([i for i in range(self.in_channel)])
        self.g.add_nodes_from([i + self.in_channel for i in range(self.num_unit)])
        for i in range(self.in_channel):
            for j in range(self.num_unit):
                index_j = j + self.in_channel
                self.g.add_edge(i, index_j)

        self.edge_features = torch.zeros(self.in_channel, self.num_unit).to('cuda')

        x_ = x.transpose(0, 2)
        x_ = DGLFeature(x_, batch_size * 16)

        x = x.transpose(1, 2)
        x = torch.stack([x] * self.num_unit, dim=2).unsqueeze(4)
        W = torch.cat([self.weight.unsqueeze(0)] * batch_size, dim=0)
        u_hat = torch.matmul(W, x).permute(1, 2, 0, 3, 4).squeeze().contiguous()

        self.node_feature = DGLFeature(torch.zeros(self.num_unit, batch_size, self.unit_size).to('cuda'), batch_size * self.unit_size)
        nf = torch.cat([x_.nf, self.node_feature.nf], dim=0)

        self.g.set_e_repr({'b_ij': self.edge_features.view(-1)})
        self.g.set_n_repr({'h': nf})
        self.g.set_e_repr({'u_hat': u_hat.view(-1, batch_size, self.unit_size)})

        for _ in range(self.num_routing):
            self.g.update_all(self.capsule_msg, self.capsule_reduce, lambda x: x, batchable=True)
            self.g.set_n_repr({'h': self.g.get_n_repr()['__REPR__']})
            self.update_edge()

        self.node_feature = DGLFeature(
            self.g.get_n_repr()['__REPR__'].index_select(0, torch.arange(self.in_channel,
                                                                         self.in_channel + self.num_unit).to("cuda")),
            batch_size * self.unit_size)
        return self.node_feature.of.transpose(0, 1).unsqueeze(1).unsqueeze(4).squeeze(1)

    def update_edge(self):
        self.g.update_edge(dgl.base.ALL, dgl.base.ALL,
                           lambda u, v, edge: edge['b_ij'] + torch.sum(v['h'] * edge['u_hat']),
                           batchable=True)

    @staticmethod
    def capsule_msg(src, edge):
        return {'b_ij': edge['b_ij'], 'h': src['h'], 'u_hat': edge['u_hat']}

    def capsule_reduce(self, node, msg):

        b_ij_c, h_c, u_hat_c = msg['b_ij'], msg['h'], msg['u_hat']
        u_hat = u_hat_c
        c_i = F.softmax(b_ij_c, dim=0)
        s_j = (c_i.unsqueeze(2).unsqueeze(3) * u_hat).sum(dim=1)
        v_j = self.squash(s_j)
        return v_j

    @staticmethod
    def squash(s):
        # This is equation 1 from the paper.
        mag_sq = torch.sum(s ** 2, dim=2, keepdim=True)
        mag = torch.sqrt(mag_sq)
        s = (mag_sq / (1.0 + mag_sq)) * (s / mag)
        return s
