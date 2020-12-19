import torch
import numpy as np
import backbone
import os
import random
import glob
from methods.baseline import Baseline
from methods.bf3s import Bf3s
from methods.protonet import ProtoNet
from methods.matchingnet import MatchingNet
from methods.relationnet import RelationNet
from methods.maml import MAML
from data.datamgr import SimpleDataManager, SetDataManager
import h5py

# region common parameters

base_path = os.path.dirname(__file__).replace('\\', '/')

model_dict = dict(
    Conv4=backbone.Conv4,
    Conv4S=backbone.Conv4S,
    Conv6=backbone.Conv6,
    ResNet10=backbone.ResNet10,
    ResNet18=backbone.ResNet18,
    ResNet34=backbone.ResNet34,
    ResNet50=backbone.ResNet50,
    ResNet101=backbone.ResNet101
)

data_dir = dict(
    CUB='/seu_share/home/xuehui/PALM_HXUE/ayx/FSL/filelists/CUB/',
    miniImagenet='/seu_share/home/xuehui/PALM_HXUE/ayx/FSL/filelists/miniImagenet/',
    omniglot='/seu_share/home/xuehui/PALM_HXUE/ayx/FSL/filelists/omniglot/',
    emnist='/seu_share/home/xuehui/PALM_HXUE/ayx/FSL/filelists/emnist/',
    cifar='/seu_share/home/xuehui/PALM_HXUE/ayx/FSL/filelists/cifar/'
)

start_epoch = 0  # Starting epoch
# stop_epoch = 1  # Stopping epoch
save_freq = 50  # Save frequency
train_n_way = 5  # class num to classify for training
test_n_way = 5  # class num to classify for testing (validation)
test_iter_num = 10  # Only for test the code, test iteration num
num_workers = 0  # Only for test the code
verbose = False


# endregion

def one_hot(y, num_class):
    return torch.zeros((len(y), num_class)).scatter_(1, y.unsqueeze(1).long(), 1)


def DBindex(cl_data_file):
    class_list = cl_data_file.keys()
    cl_num = len(class_list)
    cl_means = []
    stds = []
    DBs = []
    for cl in class_list:
        cl_means.append(np.mean(cl_data_file[cl], axis=0))
        stds.append(np.sqrt(np.mean(np.sum(np.square(cl_data_file[cl] - cl_means[-1]), axis=1))))

    mu_i = np.tile(np.expand_dims(np.array(cl_means), axis=0), (len(class_list), 1, 1))
    mu_j = np.transpose(mu_i, (1, 0, 2))
    mdists = np.sqrt(np.sum(np.square(mu_i - mu_j), axis=2))

    for i in range(cl_num):
        DBs.append(np.max([(stds[i] + stds[j]) / mdists[i, j] for j in range(cl_num) if j != i]))
    return np.mean(DBs)


def sparsity(cl_data_file):
    class_list = cl_data_file.keys()
    cl_sparsity = []
    for cl in class_list:
        cl_sparsity.append(np.mean([np.sum(x != 0) for x in cl_data_file[cl]]))
    return np.mean(cl_sparsity)


def set_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_image_size(model_name, dataset):
    if 'Conv' in model_name:
        if dataset in ['omniglot', 'cross_char']:
            image_size = 28
        else:
            image_size = 84
    else:
        image_size = 224
    return image_size


def get_train_files(dataset):
    if dataset == 'cross':
        base_file = data_dir['miniImagenet'] + 'all.json'
        val_file = data_dir['CUB'] + 'val.json'
    elif dataset == 'cross_char':
        base_file = data_dir['omniglot'] + 'noLatin.json'
        val_file = data_dir['emnist'] + 'val.json'
    else:
        base_file = data_dir[dataset] + 'base.json'
        val_file = data_dir[dataset] + 'val.json'
    return base_file, val_file


def get_train_loader(algorithm, image_size, base_file, val_file, train_n_way, test_n_way, n_shot, noise_rate=0.,
                     val_noise=True, num_workers=4):
    if algorithm in ['baseline', 'baseline++']:
        base_datamgr = SimpleDataManager(image_size, batch_size=16)
        base_loader = base_datamgr.get_data_loader(base_file, aug=True)
        val_datamgr = SimpleDataManager(image_size, batch_size=64)
        val_loader = val_datamgr.get_data_loader(val_file, aug=False)
    else:
        n_query = max(1, int(
            16 * test_n_way / train_n_way))  # if test_n_way <train_n_way, reduce n_query to keep batch size small
        base_datamgr = SetDataManager(image_size, n_query=n_query, n_way=train_n_way, n_support=n_shot,
                                      noise_rate=noise_rate, num_workers=num_workers)  # n_eposide=100
        base_loader = base_datamgr.get_data_loader(base_file, aug=True)
        if val_noise:
            val_datamgr = SetDataManager(image_size, n_query=n_query, n_way=test_n_way, n_support=n_shot,
                                         noise_rate=noise_rate, num_workers=num_workers)
        else:
            val_datamgr = SetDataManager(image_size, n_query=n_query, n_way=test_n_way, n_support=n_shot,
                                         noise_rate=0., num_workers=num_workers)
        val_loader = val_datamgr.get_data_loader(val_file, aug=False)
        # a batch for SetDataManager: a [n_way, n_support + n_query, dim, w, h] tensor
    return base_loader, val_loader


def get_novel_file(dataset, split='novel'):
    if dataset == 'cross':
        if split == 'base':
            loadfile = data_dir['miniImagenet'] + 'all.json'
        else:
            loadfile = data_dir['CUB'] + split + '.json'
    elif dataset == 'cross_char':
        if split == 'base':
            loadfile = data_dir['omniglot'] + 'noLatin.json'
        else:
            loadfile = data_dir['emnist'] + split + '.json'
    else:
        loadfile = data_dir[dataset] + split + '.json'
    return loadfile


def get_model_name(model_name, dataset):
    if dataset in ['omniglot', 'cross_char']:
        assert model_name == 'Conv4', 'omniglot only support Conv4 without augmentation'
        model_name = 'Conv4S'
    return model_name


def get_stop_epoch(algorithm, dataset, n_shot=5):
    if algorithm in ['baseline', 'baseline++']:
        if dataset in ['omniglot', 'cross_char']:
            stop_epoch = 5
        elif dataset in ['CUB']:
            stop_epoch = 200  # This is different as stated in the open-review paper. However, using 400 epoch in baseline actually lead to over-fitting
        elif dataset in ['miniImagenet', 'cross']:
            stop_epoch = 400
        else:
            stop_epoch = 400  # default
    else:  # meta-learning methods
        if n_shot == 1:
            stop_epoch = 600
        elif n_shot == 5:
            stop_epoch = 400
        else:
            stop_epoch = 600  # default
    return stop_epoch
    # return 1 # Only for test the code


def get_assigned_file(checkpoint_dir, num):
    assign_file = os.path.join(checkpoint_dir, '{:d}.tar'.format(num))
    return assign_file


def get_resume_file(checkpoint_dir, epoch=None):
    filelist = glob.glob(os.path.join(checkpoint_dir, '*.tar'))
    if len(filelist) == 0:
        return None
    if epoch is not None:
        resume_file = os.path.join(checkpoint_dir, '{:d}.tar'.format(epoch))
        return resume_file
    filelist = [x for x in filelist if os.path.basename(x) != 'best_model.tar']
    epochs = np.array([int(os.path.splitext(os.path.basename(x))[0]) for x in filelist])
    max_epoch = np.max(epochs)
    resume_file = os.path.join(checkpoint_dir, '{:d}.tar'.format(max_epoch))
    return resume_file


def get_best_file(checkpoint_dir):
    best_file = os.path.join(checkpoint_dir, 'best_model.tar')
    if os.path.isfile(best_file):
        return best_file
    else:
        return get_resume_file(checkpoint_dir)


def get_model(algorithm, model_name, dataset, n_way, n_shot):
    if dataset == 'omniglot':
        num_classes = 4112  # total number of classes in softmax, only used in baseline
    elif dataset == 'cross_char':
        num_classes = 1597
    else:
        num_classes = 200
    if algorithm == 'baseline':
        model = Baseline(model_dict[model_name], n_way=n_way, n_support=n_shot, num_class=num_classes,
                         loss_type='softmax')
    elif algorithm == 'baseline++':
        model = Baseline(model_dict[model_name], n_way=n_way, n_support=n_shot, num_class=num_classes, loss_type='dist')
    elif algorithm == 'protonet':
        model = ProtoNet(model_dict[model_name], n_way=n_way, n_support=n_shot)
    elif algorithm == 'matchingnet':
        model = MatchingNet(model_dict[model_name], n_way=n_way, n_support=n_shot)
    elif algorithm in ['relationnet', 'relationnet_softmax']:
        if model_name == 'Conv4':
            feature_model = backbone.Conv4NP
        elif model_name == 'Conv6':
            feature_model = backbone.Conv6NP
        elif model_name == 'Conv4S':
            feature_model = backbone.Conv4SNP
        else:
            feature_model = lambda: model_dict[model](flatten=False)
        loss_type = 'mse' if algorithm == 'relationnet' else 'softmax'
        model = RelationNet(feature_model, loss_type=loss_type, n_way=n_way, n_support=n_shot)
    elif algorithm in ['maml', 'maml_approx']:
        backbone.ConvBlock.maml = True
        backbone.SimpleBlock.maml = True
        backbone.BottleneckBlock.maml = True
        backbone.ResNet.maml = True
        model = MAML(model_dict[model_name], approx=(algorithm == 'maml_approx'), n_way=n_way, n_support=n_shot)
        if dataset in ['omniglot', 'cross_char']:  # maml use different parameter in omniglot
            model.n_task = 32
            model.task_update_num = 1
            model.train_lr = 0.1
    else:
        raise ValueError('Unknown algorithm')
    return model


def get_checkpoint_dir(algorithm, model_name, dataset, train_n_way, n_shot, addition=None):
    if addition is None:
        checkpoint_dir = base_path + '/save/checkpoints/%s/%s_%s' % (dataset, model_name, algorithm)
    else:
        checkpoint_dir = base_path + '/save/checkpoints/%s/%s_%s_%s' % (dataset, model_name, algorithm, str(addition))
    if not algorithm in ['baseline', 'baseline++']:
        checkpoint_dir += '_%dway_%dshot' % (train_n_way, n_shot)
    if not os.path.isdir(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    return checkpoint_dir


def save_features(feature_extractor, data_loader, outfile):
    f = h5py.File(outfile, 'w')
    max_count = len(data_loader) * data_loader.batch_size
    all_labels = f.create_dataset('all_labels', (max_count,), dtype='i')
    all_feats = None
    count = 0
    for i, (x, y) in enumerate(data_loader):
        if i % 10 == 0:
            print('{:d}/{:d}'.format(i, len(data_loader)))
        x = x.cuda()
        feats = feature_extractor(x)
        if all_feats is None:
            all_feats = f.create_dataset('all_feats', [max_count] + list(feats.size()[1:]), dtype='f')
        all_feats[count:count + feats.size(0)] = feats.data.cpu().numpy()
        all_labels[count:count + feats.size(0)] = y.cpu().numpy()
        count = count + feats.size(0)
    count_var = f.create_dataset('count', (1,), dtype='i')
    count_var[0] = count
    f.close()


def feature_evaluation(cl_data_file, model, n_way=5, n_support=5, n_query=15, adaptation=False):
    class_list = cl_data_file.keys()
    select_class = random.sample(class_list, n_way)
    z_all = []
    for cl in select_class:
        img_feat = cl_data_file[cl]
        perm_ids = np.random.permutation(len(img_feat)).tolist()
        z_all.append([np.squeeze(img_feat[perm_ids[i]]) for i in range(n_support + n_query)])  # stack each batch
    z_all = torch.from_numpy(np.array(z_all))
    model.n_query = n_query
    if adaptation:
        scores = model.set_forward_adaptation(z_all, is_feature=True)
    else:
        scores = model.set_forward(z_all, is_feature=True)
    pred = scores.data.cpu().numpy().argmax(axis=1)
    y = np.repeat(range(n_way), n_query)
    acc = np.mean(pred == y) * 100
    return acc
