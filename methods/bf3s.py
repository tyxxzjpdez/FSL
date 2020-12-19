import backbone
import torch
import torch.nn as nn
import numpy as np
from methods.meta_template import MetaTemplate

def apply_2d_rotation(input_tensor, rotation):
    """Apply a 2d rotation of 0, 90, 180, or 270 degrees to a tensor.
    The code assumes that the spatial dimensions are the last two dimensions,
    e.g., for a 4D tensors, the height dimension is the 3rd one, and the width
    dimension is the 4th one.
    """
    assert input_tensor.dim() >= 2

    height_dim = input_tensor.dim() - 2
    width_dim = height_dim + 1

    flip_upside_down = lambda x: torch.flip(x, dims=(height_dim,))
    flip_left_right = lambda x: torch.flip(x, dims=(width_dim,))
    spatial_transpose = lambda x: torch.transpose(x, height_dim, width_dim)

    if rotation == 0:  # 0 degrees rotation
        return input_tensor
    elif rotation == 90:  # 90 degrees rotation
        return flip_upside_down(spatial_transpose(input_tensor))
    elif rotation == 180:  # 90 degrees rotation
        return flip_left_right(flip_upside_down(input_tensor))
    elif rotation == 270:  # 270 degrees rotation / or -90
        return spatial_transpose(flip_upside_down(input_tensor))
    else:
        raise ValueError(
            "rotation should be 0, 90, 180, or 270 degrees; input value {}".format(rotation)
        )

class Selfsupervision_rot(nn.Module):
    def __init__(self, model_func ='Conv4', num_classes=4):
        super(Selfsupervision_rot, self).__init__()

        assert model_func == 'Conv4', "Only support Conv4"
        if model_func == 'Conv4':
            self.feature_extractor = model_func().cuda()
        self.classifier = backbone.distLinear(self.feature_extractor.final_feat_dim, num_classes).cuda()
        self.num_classes = num_classes

    def forward(self, x):
        x_90 = apply_2d_rotation(x,90)
        x_180 = apply_2d_rotation(x, 180)
        x_270 = apply_2d_rotation(x, 270)
        x = torch.cat([x, x_90, x_180, x_270], dim=0)
        out = self.feature_extractor.forward(x)
        scores = self.classifier.forward(out)
        return scores

    def loss(self, x):

        loss_function = nn.CrossEntropyLoss()
        loss_function = loss_function.cuda()

        x = x.view(-1, self.num_classes)
        y = torch.from_numpy(np.repeat(range(self.num_classes), x.size(0)/self.num_classes)).cuda()

        loss = loss_function(x, y.long())
        return loss


class Bf3s(MetaTemplate):
    def __init__(self, model_func, n_way, n_support, num_class, alpha=1.0 ,loss_type="dist"):
        super(Bf3s, self).__init__(model_func, n_way, n_support)
        self.feature_extractor = model_func().cuda()
        if loss_type == 'softmax':
            self.classifier = nn.Linear(self.feature_extractor.final_feat_dim, num_class)
            self.classifier.bias.data.fill_(0)
        elif loss_type == 'dist':  # Baseline ++
            self.classifier = backbone.distLinear(self.feature_extractor.final_feat_dim, num_class).cuda()
        self.loss_type = loss_type  # 'softmax' #'dist'
        self.num_class = num_class
        self.loss_fn = nn.CrossEntropyLoss().cuda()
        self.alpha = alpha
        self.self_supervision_net = Selfsupervision_rot().cuda()

    def set_forward(self, x, is_feature):
        scores_fewshot = self.feature_extractor.forward(x)
        scores_fewshot = self.classifier.forward(scores_fewshot)

        socres_selfsupervision = self.self_supervision_net(x)
        return scores_fewshot,socres_selfsupervision

    def set_forward_loss(self, x):
        scores_fewshot, scores_selfsupervision = self.set_forward(x)
        y = torch.from_numpy(np.repeat(range(self.n_way), self.n_query)).long()
        return self.loss_fn(scores_fewshot, y.cuda()) + \
               self.alpha * self.self_supervision_net.loss(scores_selfsupervision)

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

    def test_loop(self, test_loader, record=None, return_std=False):
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