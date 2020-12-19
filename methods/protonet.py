# This code is modified from https://github.com/jakesnell/prototypical-networks 

import torch
import torch.nn as nn
import numpy as np
from methods.meta_template import MetaTemplate


class ProtoNet(MetaTemplate):
    def __init__(self, model_func, n_way, n_support, use_cuda=True, adaptation=False):
        super(ProtoNet, self).__init__(model_func, n_way, n_support, use_cuda=use_cuda, adaptation=adaptation)
        self.loss_fn = nn.CrossEntropyLoss()

    def set_forward(self, x, is_feature=False):
        z_support, z_query = self.parse_feature(x, is_feature)
        z_proto = z_support.reshape(self.n_way, self.n_support, -1).mean(1)  # [N,d]
        z_query = z_query.reshape(self.n_way * self.n_query, -1)  # [N*Q,d]
        return -self.euclidean_dist(z_query, z_proto)

    def set_forward_loss(self, x):
        y_query = torch.from_numpy(np.repeat(range(self.n_way), self.n_query)).long()
        if self.use_cuda:
            y_query = y_query.cuda()
        scores = self.set_forward(x)
        return self.loss_fn(scores, y_query)

    def euclidean_dist(self, x, y):
        # x: m x d
        # y: n x d
        # return: m x n
        assert x.size(1) == y.size(1)
        x = x.unsqueeze(1).expand(x.size(0), y.size(0), x.size(1))  # [m,1*n,d]
        y = y.unsqueeze(0).expand(x.shape)  # [1*m,n,d]
        return torch.pow(x - y, 2).sum(2)
