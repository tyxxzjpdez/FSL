# This code is modified from https://github.com/facebookresearch/low-shot-shrink-hallucinate

import torch
import torch.nn as nn
import numpy as np
from methods.meta_template import MetaTemplate
import utils


class MatchingNet(MetaTemplate):
    def __init__(self, model_func, n_way, n_support, use_cuda=True, adaptation=False):
        super(MatchingNet, self).__init__(model_func, n_way, n_support, use_cuda=use_cuda, adaptation=adaptation)
        self.loss_fn = nn.NLLLoss()
        self.FCE = FullyContextualEmbedding(self.feat_dim).cuda()
        self.G_encoder = nn.LSTM(input_size=self.feat_dim, hidden_size=self.feat_dim, num_layers=1, batch_first=True,
                                 bidirectional=True)

    def set_forward(self, x, is_feature=False):
        z_support, z_query = self.parse_feature(x, is_feature)
        z_support = z_support.reshape(self.n_way * self.n_support, -1)  # [N*S,d]
        z_query = z_query.reshape(self.n_way * self.n_query, -1)  # [N*Q,d]

        out_G = self.G_encoder(z_support.unsqueeze(0))[0].squeeze()  # S.unsqueeze(0):[1,N*S,d], out_G:[N*S,2d]
        G = z_support + out_G[:, :z_support.size(1)] + out_G[:, z_support.size(1):]  # [N*S,d]
        G_normalized = torch.nn.functional.normalize(G, p=2, dim=1)

        y_s = torch.from_numpy(np.repeat(range(self.n_way), self.n_support))  # [N*S]
        Y_S = utils.one_hot(y_s, self.n_way).cuda()  # [N*S,N]
        F = self.FCE(z_query, G)  # [N*Q,d]
        F_normalized = torch.nn.functional.normalize(F, p=2, dim=1)  # [N*Q,d]

        # The original paper use cosine simlarity, but here we scale it by 100 to strengthen highest probability after softmax
        scores = torch.relu(F_normalized.mm(G_normalized.transpose(0, 1))) * 100  # [N*Q,N*S]
        logprobs = (torch.softmax(scores, dim=1).mm(Y_S) + 1e-6).log()  # [N*Q,N]
        return logprobs

    def set_forward_loss(self, x):
        y_query = torch.from_numpy(np.repeat(range(self.n_way), self.n_query)).cuda()
        logprobs = self.set_forward(x)
        return self.loss_fn(logprobs, y_query.long())


class FullyContextualEmbedding(nn.Module):
    def __init__(self, feat_dim):
        super(FullyContextualEmbedding, self).__init__()
        self.lstmcell = nn.LSTMCell(feat_dim * 2, feat_dim)
        self.c_0 = torch.zeros(1, feat_dim).cuda()

    def forward(self, f, G):
        h = f  # [N*Q,d]
        c = self.c_0.expand_as(f)  # [N*Q,d]
        G_T = G.transpose(0, 1)  # [d,N*S]
        for k in range(G.size(0)):
            logit_a = h.mm(G_T)  # [N*Q,N*S]
            a = torch.softmax(logit_a, dim=1)  # [N*Q,N*S]
            r = a.mm(G)  # [N*Q, d]
            x = torch.cat((f, r), 1)  # [N*Q,2d]
            h, c = self.lstmcell(x, (h, c))
            h = h + f
        return h
