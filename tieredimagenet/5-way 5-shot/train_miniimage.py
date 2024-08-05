# coding=utf-8
import gpytorch
from torch.nn import functional as F
from prototypical_batch_sampler import PrototypicalBatchSampler
from prototypical_loss import prototypical_loss as loss_fn
from miniImageNet_dataset import MiniDataset
from protonet import ProtoNet
from backbone import Conv64F_Local
from backbone import Conv64F
from backbone import ResNet12
from parser_util import get_parser
# from pronet import GPRegressionModel
from few_shot_splits_samples import split_samples
from Fc_Global import cbam_block
from Fc_Global import fc_block
from Getconvgp import convgp
from tqdm import tqdm
import numpy as np
import torch
import os
import random


def init_seed(opt):
    '''
    Disable cudnn to maximize reproducibility
    通过设置这些随机种子，可以在每次运行代码时获得相同的随机数序列。
    这对于实验的可重复性和结果的一致性非常重要，因为在相同的随机种子下，相同的初始化、
    数据处理和模型训练过程将产生相同的结果。这对于调试、复现实验结果以及比较不同模型或算法的性能非常有用。
    '''
    random.seed(opt.manual_seed)
    torch.cuda.cudnn_enabled = False
    np.random.seed(opt.manual_seed)
    torch.manual_seed(opt.manual_seed)
    torch.cuda.manual_seed(opt.manual_seed)
    torch.backends.cudnn.deterministic = True


def init_dataset(opt, mode):
    dataset = MiniDataset(mode=mode, root=opt.dataset_root)
    n_classes = len(np.unique(dataset.label))
    if n_classes < opt.classes_per_it_tr or n_classes < opt.classes_per_it_val:
        raise(Exception('There are not enough classes in the dataset in order ' +
                        'to satisfy the chosen classes_per_it. Decrease the ' +
                        'classes_per_it_{tr/val} option and try again.'))
    return dataset


def init_sampler(opt, labels, mode):
    if 'train' in mode:
        classes_per_it = opt.classes_per_it_tr
        num_samples = opt.num_support_tr + opt.num_query_tr #10
    else:
        classes_per_it = opt.classes_per_it_val
        num_samples = opt.num_support_val + opt.num_query_val #10

    return PrototypicalBatchSampler(labels=labels,
                                    classes_per_it=classes_per_it,
                                    num_samples=num_samples,
                                    iterations=opt.iterations)


def init_dataloader(opt, mode):
    dataset = init_dataset(opt, mode)
    sampler = init_sampler(opt, dataset.label, mode)
    dataloader = torch.utils.data.DataLoader(dataset, batch_sampler=sampler,num_workers = 8)  #batch_sampler 每次返回一批索引的策略
    return dataloader


def init_protonet(opt):
    '''
    Initialize the ProtoNet
    '''
    device = 'cuda:0' if torch.cuda.is_available() and opt.cuda else 'cpu'
    model = ResNet12().to(device)
    return model


def load_model(model,dir):
    model_dict = model.state_dict()
    print('loading model from :', dir)
    pretrained_dict = torch.load(dir)['params']
    if 'encoder' in list(pretrained_dict.keys())[0]:  # load from a parallel meta-trained model
        if 'module' in list(pretrained_dict.keys())[0]:
            pretrained_dict = {k[7:]: v for k, v in pretrained_dict.items()}
        else:
            pretrained_dict = {k: v for k, v in pretrained_dict.items()}
    else:
        pretrained_dict = {'encoder.' + k: v for k, v in pretrained_dict.items()}  # load from a pretrained model
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)  # update the param in encoder, remain others still
    model.load_state_dict(model_dict)
    return model


def init_optim(opt, model, global_model,attention_model):
    '''
    Initialize optimizer
    '''
    return torch.optim.Adam([
                            {'params': model.parameters(),'lr' :opt.learning_rate},
                            # {'params': global_model.parameters()},
                            {'params': attention_model.parameters(),'lr' : 1e-4}]
                            ,weight_decay=0.005)
                            # ,lr=opt.learning_rate,weight_decay=0.005) #,weight_decay= 1e-4


def init_lr_scheduler(opt, optim):
    '''
    Initialize the learning rate scheduler
    '''
    return torch.optim.lr_scheduler.StepLR(optimizer=optim,
                                           gamma=opt.lr_scheduler_gamma,
                                           step_size=opt.lr_scheduler_step)


def save_list_to_file(path, thelist):
    with open(path, 'w') as f:
        for item in thelist:
            f.write("%s\n" % item)


def train(opt, tr_dataloader, model, optim, lr_scheduler, val_dataloader=None, global_model=None,attention_model=None):
    '''
    Train the model with the prototypical learning algorithm
    '''

    device = 'cuda:0' if torch.cuda.is_available() and opt.cuda else 'cpu'

    if val_dataloader is None:
        best_state = None
    train_loss = []
    train_acc = []
    val_loss = []
    val_acc = []
    best_acc = 0

    best_model_path = os.path.join(opt.experiment_root, 'best_model.pth')
    best_attention_model_path = os.path.join(opt.experiment_root, 'best_attention_model.pth')
    last_attention_model_path = os.path.join(opt.experiment_root, 'last_attention_model.pth')
    last_model_path = os.path.join(opt.experiment_root, 'last_model.pth')

    label = 0

    for epoch in range(opt.epochs):
        print('=== Epoch: {} ==='.format(epoch))
        tr_iter = iter(tr_dataloader)
        for batch in tqdm(tr_iter):  #tqdm显示进度条
            x, y = batch
            x, y = x.to(device), y.to(device)
            model.train()
            global_model.train()
            attention_model.train()
            model_output = model(x)
            prototypes,query_samples,num_classes,n_query = split_samples(model_output,target=y, n_support=opt.num_support_tr)
            target_inds = torch.arange(0, num_classes).cuda()
            target_inds = target_inds.view(num_classes, 1)  # 60x1
            target_inds = target_inds.expand(num_classes, 5).long().reshape(-1)
            # target_inds = torch.zeros(num_classes*1,num_classes).cuda().scatter_(1,target_inds.unsqueeze(1),1)
            target_inds = torch.full([num_classes*5,num_classes],0.025).cuda().scatter_(1,target_inds.unsqueeze(1),0.9)


            covar = convgp(prototypes,global_model=global_model,attention_model=attention_model)
            #在covar的对角线上加上噪声
            covar = covar + torch.eye(covar.shape[0]).cuda() * (1e-4)
            covar_inv = covar.inverse()
            k = convgp(prototypes,query_samples,global_model=global_model,attention_model=attention_model)
            kt = k.transpose(0,1).cuda()
            #求出均值
            mean = torch.matmul(torch.matmul(kt,covar_inv),target_inds)

            #求出方差
            c = convgp(query_samples,global_model=global_model,attention_model=attention_model)
            #在c的对角线上加上噪声
            # c = c + torch.eye(c.shape[0]).cuda() * (1e-4)
            
            var = c.diag() - torch.matmul(torch.matmul(kt,covar_inv),k).diag()

            optim.zero_grad()
            # optim2.zero_grad()
            loss, acc = loss_fn(num_classes, n_query, mean, var)

            loss.backward()
            optim.step()
            # optim2.step()
            train_loss.append(loss.item())
            train_acc.append(acc.item())

        avg_loss = np.mean(train_loss[-opt.iterations: ])#取train_loss和train_acc列表的最后opt.iterations个元素来计算平均值。
        avg_acc = np.mean(train_acc[-opt.iterations:])
        train_acc.append(avg_acc.item())
        train_loss.append(avg_loss.item())
        print('Avg model Train Loss: {}, Avg Train Acc: {}'.format(avg_loss, avg_acc))
        lr_scheduler.step()
        if val_dataloader is None:
            continue
        val_iter = iter(val_dataloader)
        model.eval()
        global_model.eval()
        attention_model.eval()

        with torch.no_grad():
            for batch in val_iter:
                x, y = batch
                x, y = x.to(device), y.to(device)
                model_output = model(x)

                prototypes,query_samples,num_classes,n_query = split_samples(model_output,target=y, n_support=opt.num_support_val)
                target_inds = torch.arange(0, num_classes).cuda()
                target_inds = target_inds.view(num_classes, 1)  # 60x1x1
                target_inds = target_inds.expand(num_classes, 5).long().reshape(-1)
                # target_inds = torch.zeros(num_classes * 1, num_classes).cuda().scatter_(1,target_inds.unsqueeze(1),1)
                target_inds = torch.full([num_classes * 5, num_classes], 0.025).cuda().scatter_(1,target_inds.unsqueeze(1), 0.9)

                covar = convgp(prototypes,global_model=global_model, attention_model=attention_model)
                # 在covar的对角线上加上噪声
                covar = covar + torch.eye(covar.shape[0]).cuda() * (1e-4)
                covar_inv = covar.inverse()
                k = convgp(prototypes, query_samples,global_model=global_model, attention_model=attention_model)
                kt = k.transpose(0, 1).cuda()
                # 求出均值
                mean = torch.matmul(torch.matmul(kt, covar_inv), target_inds)
                # 求出方差
                c = convgp(query_samples, global_model=global_model, attention_model=attention_model)
                # 在c的对角线上加上噪声
                # c = c + torch.eye(c.shape[0]).cuda() * (1e-4)

                var = c.diag() - torch.matmul(torch.matmul(kt, covar_inv), k).diag()

                loss, acc = loss_fn(num_classes, n_query, mean,var)
                val_loss.append(loss.item())
                val_acc.append(acc.item())

        avg_loss = np.mean(val_loss[-opt.iterations:])
        avg_acc = np.mean(val_acc[-opt.iterations:])
        val_acc.append(avg_acc.item())
        val_loss.append(avg_loss.item())
        postfix = ' (Best)' if avg_acc >= best_acc else ' (Best: {})'.format(
            best_acc)
        print('Avg Val Loss: {}, Avg Val Acc: {}{}'.format(
            avg_loss, avg_acc, postfix))
        if avg_acc >= best_acc:
            torch.save(model.state_dict(), best_model_path)
            torch.save(attention_model.state_dict(), best_attention_model_path)
            best_acc = avg_acc
            best_state = model.state_dict()

    torch.save(model.state_dict(), last_model_path)
    torch.save(attention_model.state_dict(), last_attention_model_path)
    val_acc.append(best_acc.item())

    for name in ['train_loss', 'train_acc', 'val_loss', 'val_acc']:
        save_list_to_file(os.path.join(opt.experiment_root,
                                       name + '.txt'), locals()[name])

    return best_state, best_acc, train_loss, train_acc, val_loss, val_acc


def test1(opt, test_dataloader, model,global_model,attention_model):
    '''
    Test the model trained with the prototypical learning algorithm
    '''
    device = 'cuda:0' if torch.cuda.is_available() and opt.cuda else 'cpu'
    avg_acc = list()
    model.eval()
    global_model.eval()
    attention_model.eval()
    with torch.no_grad():
        test_iter = iter(test_dataloader)
        for batch in test_iter:
            x, y = batch
            x, y = x.to(device), y.to(device)
            model_output = model(x)

            prototypes, query_samples, num_classes, n_query = split_samples(model_output, target=y,
                                                                            n_support=opt.num_support_val)

            target_inds = torch.arange(0, num_classes).cuda()
            target_inds = target_inds.view(num_classes, 1)  # 60x1x1
            target_inds = target_inds.expand(num_classes, 5).long().reshape(-1)
            # target_inds = torch.zeros(num_classes * 1, num_classes).cuda().scatter_(1,target_inds.unsqueeze(1),1)
            target_inds = torch.full([num_classes*5,num_classes],0.025).cuda().scatter_(1,target_inds.unsqueeze(1),0.9)

            covar = convgp(prototypes,global_model=global_model, attention_model=attention_model)
            # 在covar的对角线上加上噪声
            covar = covar + torch.eye(covar.shape[0]).cuda() * (1e-4)
            covar_inv = covar.inverse()
            k = convgp(prototypes, query_samples,global_model=global_model, attention_model=attention_model)
            kt = k.transpose(0, 1).cuda()
            # 求出均值
            mean = torch.matmul(torch.matmul(kt, covar_inv), target_inds)

            # 求出方差
            c = convgp(query_samples, global_model=global_model, attention_model=attention_model)
            # 在c的对角线上加上噪声
            # c = c + torch.eye(c.shape[0]).cuda() * (1e-4)

            var = c.diag() - torch.matmul(torch.matmul(kt, covar_inv), k).diag()

            _, acc = loss_fn(num_classes, n_query, mean, var)
            avg_acc.append(acc.item())

    avg_acc1 = np.mean(avg_acc)
    print('Test Acc: {}'.format(avg_acc1))
    avg_acc.append(avg_acc1.item())
    save_list_to_file(os.path.join(opt.experiment_root, 'test' + '.txt'), locals()['avg_acc'])


    return avg_acc


def eval(opt):
    '''
    Initialize everything and train
    '''
    options = get_parser().parse_args()

    if torch.cuda.is_available() and not options.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

    init_seed(options)
    test_dataloader = init_dataset(options)[-1]
    model = init_protonet(options)
    model_path = os.path.join(opt.experiment_root, 'best_model.pth')
    model.load_state_dict(torch.load(model_path))

    test1(opt=options,
         test_dataloader=test_dataloader,
         model=model)


def main():
    '''
    Initialize everything and train
    '''
    options = get_parser().parse_args()
    if not os.path.exists(options.experiment_root):
        os.makedirs(options.experiment_root)

    if torch.cuda.is_available() and not options.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

    init_seed(options)

    tr_dataloader = init_dataloader(options, 'train')
    val_dataloader = init_dataloader(options, 'val')
    # trainval_dataloader = init_dataloader(options, 'trainval')
    test_dataloader = init_dataloader(options, 'test')

    global_model = fc_block().cuda()
    attention_model = cbam_block(64).cuda()
    PATH = 'D:\lcb\code1\deepemd_pretrain_model\\tieredimagenet\\resnet12\max_acc.pth'
    model = init_protonet(options)
    premodel = torch.load(PATH)['params']
    model.load_state_dict(premodel)
    model = load_model(model,PATH)
    model.eval()


    optim = init_optim(options, model,global_model,attention_model)
    lr_scheduler = init_lr_scheduler(options, optim)
    res = train(opt=options,
                tr_dataloader=tr_dataloader,
                val_dataloader=val_dataloader,
                model=model,
                optim=optim,
                lr_scheduler=lr_scheduler,
                global_model = global_model,
                attention_model = attention_model)
    best_state, best_acc, train_loss, train_acc, val_loss, val_acc = res
    print('Testing with last model..')
    test1(opt=options,
         test_dataloader=test_dataloader,
         model=model,
         global_model = global_model,
         attention_model = attention_model)
    #
    model.load_state_dict(best_state)
    print('Testing with best model..')
    test1(opt=options,
         test_dataloader=test_dataloader,
         model=model,
         global_model = global_model,
         attention_model = attention_model)

if __name__ == '__main__':
    main()
