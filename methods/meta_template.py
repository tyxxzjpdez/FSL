import torch
import torch.nn as nn
import numpy as np
from abc import abstractmethod


class MetaTemplate(nn.Module):
    def __init__(self, model_func, n_way, n_support, verbose=False, change_way=True, use_cuda=True, adaptation=False):
        super(MetaTemplate, self).__init__()
        self.n_way = n_way  # N, n_classes
        self.n_support = n_support  # S, sample num of support set
        self.n_query = -1  # Q, sample num of query set(change depends on input)
        self.feature_extractor = model_func()  # feature extractor
        self.feat_dim = self.feature_extractor.final_feat_dim
        self.verbose = verbose
        self.change_way = change_way  # some methods allow different_way classification during training and test
        self.use_cuda = use_cuda
        self.adaptation = adaptation

    @abstractmethod
    def set_forward(self, x, is_feature):
        # x -> predicted score
        pass

    @abstractmethod
    def set_forward_loss(self, x):
        # x -> loss value
        pass

    def forward(self, x):
        # x-> feature embedding
        out = self.feature_extractor.forward(x)
        return out

    def parse_feature(self, x, is_adaptation=False):
        x = x.requires_grad_(True)
        x = x.reshape(self.n_way * (self.n_support + self.n_query), *x.size()[2:])
        z_all = self.feature_extractor(x)
        z_all = z_all.reshape(self.n_way, self.n_support + self.n_query, *z_all.shape[1:])  # [N, S+Q, d]
        if is_adaptation:
            z_all = z_all.detach()
        z_support = z_all[:, :self.n_support]  # [N, S, d]
        z_query = z_all[:, self.n_support:]  # [N, Q, d]
        return z_support, z_query

    def correct(self, x):
        if self.adaptation:
            scores = self.set_forward_adaptation(x)
        else:
            scores = self.set_forward(x)
        y_query = np.repeat(range(self.n_way), self.n_query)  # [0 0 0 1 1 1 2 2 2 3 3 3 4 4 4]
        topk_scores, topk_labels = scores.data.topk(1, 1, True, True)  # top1, dim=1, largest, sorted
        topk_ind = topk_labels.cpu().numpy()  # index of topk
        top1_correct = np.sum(topk_ind[:, 0] == y_query)
        return float(top1_correct), len(y_query)

    def train_loop(self, epoch, train_loader, optimizer):
        print_freq = 10
        avg_loss = 0
        for i, (x, _) in enumerate(train_loader):
            if self.use_cuda:
                x = x.cuda()
            self.n_query = x.size(1) - self.n_support  # x:[N, S+Q, n_channel, h, w]
            if self.change_way:
                self.n_way = x.size(0)
            optimizer.zero_grad()
            loss = self.set_forward_loss(x)
            loss.backward()
            optimizer.step()
            avg_loss = avg_loss + loss.item()
            if self.verbose and (i % print_freq) == 0:
                print('Epoch {:d} | Batch {:d}/{:d} | Loss {:f}'.format(epoch, i, len(train_loader),
                                                                        avg_loss / float(i + 1)))
        if not self.verbose:
            print('Epoch {:d} | Loss {:f}'.format(epoch, avg_loss / float(i + 1)))
        return avg_loss

    def train_loop_with_acc(self, epoch, train_loader, optimizer):
        print_freq = 10
        avg_loss = 0
        acc_all = []
        iter_num = len(train_loader)
        for i, (x, _) in enumerate(train_loader):
            if self.use_cuda:
                x = x.cuda()
            self.n_query = x.size(1) - self.n_support  # x:[N, S+Q, n_channel, h, w]
            if self.change_way:
                self.n_way = x.size(0)
            optimizer.zero_grad()
            loss = self.set_forward_loss(x)
            loss.backward()
            optimizer.step()
            avg_loss = avg_loss + loss.item()
            if self.verbose and (i % print_freq) == 0:
                print('Epoch {:d} | Batch {:d}/{:d} | Loss {:f}'.format(epoch, i, len(train_loader),
                                                                        avg_loss / float(i + 1)))
            correct_this, count_this = self.correct(x)
            acc_all.append(correct_this / count_this * 100)
        acc_all = np.asarray(acc_all)
        acc_mean = np.mean(acc_all)
        acc_std = np.std(acc_all)
        if self.verbose:
            print('%d Test Acc = %4.2f%% +- %4.2f%%' % (iter_num, acc_mean, 1.96 * acc_std / np.sqrt(iter_num)))
        if not self.verbose:
            print('Epoch {:d} | Loss {:f}'.format(epoch, avg_loss / float(i + 1)))
        return acc_mean, avg_loss

    def test_loop(self, test_loader, record=None, return_std=False):
        acc_all = []
        iter_num = len(test_loader)
        for i, (x, _) in enumerate(test_loader):
            if self.use_cuda:
                x = x.cuda()
            self.n_query = x.size(1) - self.n_support  # x:[N, S+Q, n_channel, h, w]
            if self.change_way:
                self.n_way = x.size(0)
            correct_this, count_this = self.correct(x)
            acc_all.append(correct_this / count_this * 100)
        acc_all = np.asarray(acc_all)
        acc_mean = np.mean(acc_all)
        acc_std = np.std(acc_all)
        if self.verbose:
            print('%d Test Acc = %4.2f%% +- %4.2f%%' % (iter_num, acc_mean, 1.96 * acc_std / np.sqrt(iter_num)))
        if return_std:
            return acc_mean, acc_std
        else:
            return acc_mean

    def set_forward_adaptation(self, x):
        # further adaptation, default is fixing feature and train a new softmax clasifier
        z_support, z_query = self.parse_feature(x, is_adaptation=True)
        z_support = z_support.reshape(self.n_way * self.n_support, -1)
        z_query = z_query.reshape(self.n_way * self.n_query, -1)
        y_support = torch.from_numpy(np.repeat(range(self.n_way), self.n_support))  # [0 0 0 1 1 1 2 2 2 3 3 3 4 4 4]
        y_support = y_support.cuda().float().requires_grad_(True)
        linear_clf = nn.Linear(self.feat_dim, self.n_way)
        linear_clf = linear_clf.cuda()
        set_optimizer = torch.optim.SGD(linear_clf.parameters(), lr=0.01, momentum=0.9, dampening=0.9,
                                        weight_decay=0.001)
        batch_size = 4
        support_size = self.n_way * self.n_support
        for epoch in range(100):
            rand_id = np.random.permutation(support_size)
            for i in range(0, support_size, batch_size):
                set_optimizer.zero_grad()
                selected_id = torch.from_numpy(rand_id[i: min(i + batch_size, support_size)]).cuda().long()
                z_batch = z_support[selected_id]
                y_batch = y_support[selected_id]
                scores = linear_clf(z_batch)
                loss = nn.CrossEntropyLoss()(scores, y_batch.long())
                loss.backward()
                set_optimizer.step()
        scores = linear_clf(z_query)
        return scores
