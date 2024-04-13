import torch.nn as nn
import torch.nn.functional as F
import torch
from ppnp.pytorch.utils import MixedDropout, MixedLinear
from torch.nn.parameter import Parameter

class HyperAdaGCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, dropout_adj,view=4):
        super().__init__()
        fcs = [MixedLinear(nfeat, nhid, bias=False), MixedLinear(nfeat, nhid, bias=False),
               nn.Linear(nhid, nclass, bias=False)]
        self.gate1 =  nn.Linear(nhid, nhid, bias=False)
        self.gate2 =  nn.Linear(nhid, nhid, bias=False)
        self.gate_transform = nn.Linear(2*nhid, nhid, bias=False)
        self.fcs = nn.ModuleList(fcs)
        self.reg_params = list(self.fcs[0:3].parameters())
        self.x_orin = None
        self.simple_features = None
        self.x_hyper_adj = None
        self.x_simple_adj = None
        self.w1 = Parameter(torch.Tensor([0.5]), requires_grad=True)
        self.nhid = nhid
        self.node_features = None
        self.edge_features = None

        if dropout == 0:
            self.dropout = lambda x: x
        else:
            self.dropout = MixedDropout(dropout)  # p: drop rate
        if dropout_adj == 0:
            self.dropout_adj = lambda x: x
        else:
            self.dropout_adj = MixedDropout(dropout_adj)  # p: drop rate
        self.act_fn = nn.ReLU()
        self.reset_parameters()

    def reset_parameters(self):
        self.w1.data.fill_(0.5)

    def normalize_l2(self, X):
        """Row-normalize  matrix"""
        rownorm = X.detach().norm(dim=1, keepdim=True)
        scale = rownorm.pow(-1)
        scale[torch.isinf(scale)] = 0.
        X = X * scale
        return X

    def _transform_features(self, x):
        x = self.normalize_l2(x)
        # second contribution, utilized the simple feature information, a 2 view hyperGCN
        self.simple_features = self.normalize_l2(self.simple_features)
        layer_inner = self.act_fn(self.fcs[0](self.dropout(x)))
        lay_simple_inner = self.act_fn(self.fcs[1](self.dropout(self.simple_features)))
        gate1 = self.gate1(layer_inner)
        gate2 = self.gate2(lay_simple_inner)
        # gate = torch.tanh(gate1 + gate2)
        gate = torch.sigmoid(self.gate_transform(torch.concatenate([gate1, gate2], dim=1)))

        fix_inner = (1-gate) * layer_inner + (gate * lay_simple_inner)
        self.simple_x = lay_simple_inner.detach().to('cpu').numpy()
        self.hyper_x = layer_inner.detach().to('cpu').numpy()
        self.fix_inner = fix_inner.detach().to('cpu').numpy()
        # fix_inner =  layer_inner
        fix_inner = self.normalize_l2(fix_inner)
        res = self.act_fn(self.fcs[-1](self.dropout_adj(fix_inner)))
        return res

    def forward(self, x, idx):  # X, A
        logits = self._transform_features(x)
        return F.log_softmax(logits, dim=-1)[idx]

