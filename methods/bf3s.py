import backbone
import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable
from methods.meta_template import MetaTemplate

def apply_2d_rotation(input_tensor, rotation):
    """Apply a 2d rotation of 0, 90, 180, or 270 degrees to a tensor.
    The code assumes that the spatial dimensions are the last two dimensions,
    e.g., for a 4D tensors, the height dimension is the 3rd one, and the width
    dimension is the 4th one.
    """
    assert input_tensor.dim() >= 2

    shape = input_tensor.shape

    height_dim = input_tensor.dim() - 2
    width_dim = height_dim + 1

    flip_upside_down = lambda x: torch.flip(x, dims=(height_dim,))
    flip_left_right = lambda x: torch.flip(x, dims=(width_dim,))
    spatial_transpose = lambda x: torch.transpose(x, height_dim, width_dim)

    if rotation == 0:  # 0 degrees rotation
        return input_tensor.view(shape)
    elif rotation == 90:  # 90 degrees rotation
        return flip_upside_down(spatial_transpose(input_tensor)).view(shape)
    elif rotation == 180:  # 90 degrees rotation
        return flip_left_right(flip_upside_down(input_tensor)).view(shape)
    elif rotation == 270:  # 270 degrees rotation / or -90
        return spatial_transpose(flip_upside_down(input_tensor)).view(shape)
    else:
        raise ValueError(
            "rotation should be 0, 90, 180, or 270 degrees; input value {}".format(rotation)
        )
def tranform_shape(x, n_support):
    ways = x.size(0)
    n_views = x.size(1)
    shots = n_support
    query_shots = n_views - shots
    x_support = x[:, :shots].reshape((ways * shots, *x.shape[-3:]))
    x_support = Variable(x_support.cuda())
    x_query = x[:, shots:].reshape((ways * query_shots, *x.shape[-3:]))
    x_query = Variable(x_query.cuda())

    # Extract features
    x_both = torch.cat([x_support, x_query], 0)
    return x_both

class Selfsupervision_rot(nn.Module):
    def __init__(self, model_func, n_support,num_classes=4):
        super(Selfsupervision_rot, self).__init__()

        assert model_func is not None
        self.feature_extractor = model_func().cuda()
        self.classifier = backbone.distLinear(self.feature_extractor.final_feat_dim, num_classes).cuda()
        self.num_classes = num_classes
        self.n_support = n_support

    def forward(self, x):
        n_support = self.n_support
        tx = tranform_shape(x, n_support)

        x_90 = tranform_shape(apply_2d_rotation(x, 90), n_support)
        x_180 = tranform_shape(apply_2d_rotation(x, 180), n_support)
        x_270 = tranform_shape(apply_2d_rotation(x, 270), n_support)
        x = torch.cat([tx, x_90, x_180, x_270], dim=0)
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
        self.self_supervision_net = Selfsupervision_rot(model_func, n_support).cuda()

    def set_forward(self, x, is_feature=False):

        scores_fewshot = self.feature_extractor(tranform_shape(x, self.n_support))

        scores_fewshot = self.classifier.forward(scores_fewshot)

        socres_selfsupervision = self.self_supervision_net(x)
        return scores_fewshot,socres_selfsupervision

    def set_forward_loss(self, x):
        scores_fewshot, scores_selfsupervision = self.set_forward(x)
        y = torch.from_numpy(np.repeat(range(self.n_way), self.n_query+self.n_support)).long()
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
            x = self.feature_extractor(tranform_shape(x,self.n_support)).view(*x.shape[:2],-1)
            z_support, z_query = x[:,:self.n_support], x[:,self.n_support:]# [N, S, d], [N, Q, d]
            assert False

            eps = 0.00001
            prototype_weight = z_support / (torch.norm(z_support.mean(dim=1), p=2, dim=1, keepdim=True).expand_as(z_support) + eps)
            prototype_weight = prototype_weight.transpose(0,1)
            z_query = z_query.view(-1, z_query.size(-1)) # [N*Q,d]
            z_query = z_query / (torch.norm(z_query.mean(dim=1), p=2, dim=1, keepdim=True).expand_as(z_support) + eps)
            scores = torch.mm(z_query, prototype_weight) * self.classifier.scale_factor # [N*Q,d] * [d, N] -> [N*Q, N]

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
        if return_std:
            return acc_mean, acc_std
        return acc_mean
