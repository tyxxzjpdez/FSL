import backbone
import torch
import torch.nn as nn
import numpy as np
from methods.meta_template import MetaTemplate


class Baseline(MetaTemplate):
    def __init__(self, model_func, n_way, n_support, num_class, loss_type="softmax"):
        super(Baseline, self).__init__(model_func, n_way, n_support)
        self.feature_extractor = model_func()
        if loss_type == 'softmax':
            self.classifier = nn.Linear(self.feature_extractor.final_feat_dim, num_class)
            self.classifier.bias.data.fill_(0)
        elif loss_type == 'dist':  # Baseline ++
            self.classifier = backbone.distLinear(self.feature_extractor.final_feat_dim, num_class)
        self.loss_type = loss_type  # 'softmax' #'dist'
        self.num_class = num_class
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, x):
        x = x.cuda()
        out = self.feature_extractor.forward(x)
        scores = self.classifier.forward(out)
        return scores

    def forward_loss(self, x, y):
        scores = self.forward(x)
        return self.loss_fn(scores, y.cuda())

    def train_loop(self, epoch, train_loader, optimizer):
        for i, (x, y) in enumerate(train_loader):
            optimizer.zero_grad()
            loss = self.forward_loss(x, y)
            loss.backward()
            optimizer.step()

    def test_loop(self, test_loader, record=None):
        # correct = 0
        # count = 0
        acc_all = []
        iter_num = len(test_loader)
        for i, (x, _) in enumerate(test_loader):
            self.n_query = x.size(1) - self.n_support  # x:[N, S+Q, n_channel, h, w]
            if self.change_way:
                self.n_way = x.size(0)
            # ---------------------------
            # TODO temporally replaced the call to correct() with the code
            # correct_this, count_this = self.correct(x)
            scores = self.set_forward(x)
            y_query = np.repeat(range(self.n_way), self.n_query)  # [0 0 0 1 1 1 2 2 2 3 3 3 4 4 4]
            topk_scores, topk_labels = scores.data.topk(1, 1, True, True)
            topk_ind = topk_labels.cpu().numpy()
            top1_correct = np.sum(topk_ind[:, 0] == y_query)
            correct_this = float(top1_correct)
            count_this = len(y_query)
            # ---------------------------
            acc_all.append(correct_this / count_this * 100)
        acc_all = np.asarray(acc_all)
        acc_mean = np.mean(acc_all)
        acc_std = np.std(acc_all)
        print('%d Test Acc = %4.2f%% +- %4.2f%%' % (iter_num, acc_mean, 1.96 * acc_std / np.sqrt(iter_num)))
        return acc_mean

    def set_forward(self, x, is_feature=True):
        return self.set_forward_adaptation(x, is_feature)  # Baseline always do adaptation

    def set_forward_adaptation(self, x, is_feature=True):
        assert is_feature == True, 'Baseline only support testing with feature'
        z_support, z_query = self.parse_feature(x, is_feature)
        z_support = z_support.reshape(self.n_way * self.n_support, -1)
        z_query = z_query.contiguous().view(self.n_way * self.n_query, -1)

        y_support = torch.from_numpy(np.repeat(range(self.n_way), self.n_support))
        y_support = y_support.cuda()

        if self.loss_type == 'softmax':
            linear_clf = nn.Linear(self.feat_dim, self.n_way)
        elif self.loss_type == 'dist':
            linear_clf = backbone.distLinear(self.feat_dim, self.n_way)
        linear_clf = linear_clf.cuda()

        set_optimizer = torch.optim.SGD(linear_clf.parameters(), lr=0.01, momentum=0.9, dampening=0.9,
                                        weight_decay=0.001)
        loss_function = nn.CrossEntropyLoss()
        loss_function = loss_function.cuda()
        batch_size = 4
        support_size = self.n_way * self.n_support
        for epoch in range(100):
            rand_id = np.random.permutation(support_size)
            for i in range(0, support_size, batch_size):
                set_optimizer.zero_grad()
                selected_id = torch.from_numpy(rand_id[i: min(i + batch_size, support_size)]).cuda()
                z_batch = z_support[selected_id.long()]
                y_batch = y_support[selected_id.long()]
                scores = linear_clf(z_batch)
                loss = loss_function(scores, y_batch.long())
                loss.backward()
                set_optimizer.step()
        scores = linear_clf(z_query)
        return scores

    def set_forward_loss(self, x):
        raise ValueError('Baseline predict on pretrained feature and do not support finetune backbone')
