from utils import *
#from methods.gpshot import GPShot

datasets = ['omniglot', 'cross_char', 'CUB', 'miniImagenet', 'cross',
            'cifar']  # CUB/omniglot/miniImagenet/cross/cross_char/cifar


def get_model(algorithm, model_name, dataset, n_way, n_shot, adaptation):
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
        model = ProtoNet(model_dict[model_name], n_way=n_way, n_support=n_shot, adaptation=adaptation,
                         use_cuda=use_cuda)
    elif algorithm == 'matchingnet':
        model = MatchingNet(model_dict[model_name], n_way=n_way, n_support=n_shot, adaptation=adaptation,
                            use_cuda=use_cuda)
    #elif algorithm == 'gpshot':
    #    model = GPShot(model_dict[model_name], n_way=n_way, n_support=n_shot)
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
        model = RelationNet(feature_model, loss_type=loss_type, n_way=n_way, n_support=n_shot, adaptation=adaptation,
                            use_cuda=use_cuda)
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
    elif algorithm == "bf3s":
        model = Bf3s(model_dict[model_name], n_way=n_way, n_support=n_shot, num_class=num_classes,
                         loss_type='dist')
    else:
        raise ValueError('Unknown algorithm')
    return model

# just test cifar
datasets = ['cifar']  # CUB/omniglot/miniImagenet/cross/cross_char/cifar

for dataset in datasets:
    print(dataset)
    # region parameters
    algorithm = 'bf3s'  # protonet/matchingnet/relationnet
    model_name = 'Conv4'  # Conv{4|6} / ResNet{10|18|34|50|101}
    train_n_way = 5  # class num to classify for training
    test_n_way = 5  # class num to classify for testing (validation)
    n_shot = 5  # number of labeled data in each class, same as n_support
    noise_rate = 0.
    seed = 0
    set_seed(seed)
    stop_epoch = - 1
    use_cuda = True
    if_train = True
    if_test = True
    adaptation = False
    num_workers = 0
    # endregion

    image_size = get_image_size(model_name=model_name, dataset=dataset)
    model_name = get_model_name(model_name=model_name, dataset=dataset)
    if stop_epoch == -1:
        stop_epoch = get_stop_epoch(algorithm=algorithm, dataset=dataset)
    checkpoint_dir = get_checkpoint_dir(algorithm=algorithm, model_name=model_name, dataset=dataset,
                                        train_n_way=train_n_way, n_shot=n_shot, addition='%f' % noise_rate)
    base_file, val_file = get_train_files(dataset=dataset)
    base_loader, val_loader = get_train_loader(algorithm=algorithm, image_size=image_size, base_file=base_file,
                                               val_file=val_file, train_n_way=train_n_way, test_n_way=test_n_way,
                                               n_shot=n_shot, noise_rate=noise_rate, val_noise=True,
                                               num_workers=num_workers)

    # region train
    if if_train:
        print('Start training!')
        model = get_model(algorithm=algorithm, model_name=model_name, dataset=dataset, n_way=train_n_way,
                          n_shot=n_shot, adaptation=adaptation)
        if use_cuda:
            model = model.cuda()
        if algorithm == 'relationnet':
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        else:
            optimizer = torch.optim.Adam(model.parameters())
        if algorithm in ['maml', 'maml_approx']:
            stop_epoch = stop_epoch * model.n_task  # maml use multiple tasks in one update
        max_acc = 0
        for epoch in range(start_epoch, stop_epoch):
            model.train()
            model.train_loop(epoch, base_loader, optimizer)  # model are called by reference, no need to return
            model.eval()
            acc = model.test_loop(val_loader)
            if acc > max_acc:  # for baseline and baseline++, we don't use validation here so we let acc = -1
                print("--> Best model! save...", acc)
                max_acc = acc
                outfile = os.path.join(checkpoint_dir, 'best_model.tar')
                if 'RCL' in algorithm:
                    model.save(outfile)
                else:
                    torch.save({'epoch': epoch, 'state': model.state_dict()}, outfile)
            if not os.path.isdir(checkpoint_dir):
                os.makedirs(checkpoint_dir)
            if (epoch % save_freq == 0) or (epoch == stop_epoch - 1):
                outfile = os.path.join(checkpoint_dir, '{:d}.tar'.format(epoch))
                torch.save({'epoch': epoch, 'state': model.state_dict()}, outfile)
            assert False
    # endregion

    # region test
    if if_test:
        print('Start testing!')
        model = get_model(algorithm=algorithm, model_name=model_name, dataset=dataset, n_way=train_n_way,
                          n_shot=n_shot, adaptation=adaptation)
        if use_cuda:
            model = model.cuda()
        modelfile = get_best_file(checkpoint_dir)
        # modelfile = get_resume_file(checkpoint_dir)
        assert modelfile is not None
        tmp = torch.load(modelfile)
        model.load_state_dict(tmp['state'])
        loadfile = get_novel_file(dataset=dataset, split='novel')
        datamgr = SetDataManager(image_size, n_eposide=test_iter_num, n_query=15, n_way=test_n_way, n_support=n_shot,
                                 noise_rate=0., num_workers=num_workers)
        novel_loader = datamgr.get_data_loader(loadfile, aug=False)
        model.eval()
        acc_mean, acc_std = model.test_loop(novel_loader, return_std=True)
        print('%d Test Acc = %4.2f%% +- %4.2f%%' % (test_iter_num, acc_mean, 1.96 * acc_std / np.sqrt(test_iter_num)))
    # endregion
