# coding=utf-8
import torch
from torch.nn import functional as F
from torch.nn.modules import Module


class PrototypicalLoss(Module):
    '''
    Loss class deriving from Module for the prototypical loss function defined below
    '''
    def __init__(self, n_support):
        super(PrototypicalLoss, self).__init__()
        self.n_support = n_support

    def forward(self, input, target):
        return prototypical_loss(input, target, self.n_support)

def count_acc(logits, label):
    pred = torch.argmax(logits, dim=1)
    if torch.cuda.is_available():
        return (pred == label).type(torch.cuda.FloatTensor).mean()
    else:
        return (pred == label).type(torch.FloatTensor).mean()




def prototypical_loss(n_classes,n_query,mean, var):
    '''
    Inspired by https://github.com/jakesnell/prototypical-networks/blob/master/protonets/models/few_shot.py

    Compute the barycentres by averaging the features of n_support
    samples for each class in target, computes then the distances from each
    samples' features to each one of the barycentres, computes the
    log_probability for each n_query samples for each one of the current
    classes, of appartaining to a class c, loss and accuracy are then computed
    and returned
    Args:
    - input: the model output for a batch of samples
    - target: ground truth for the above batch of samples
    - n_support: number of samples to keep in account when computing
      barycentres, for each one of the current classes
    '''
    T = 1000
    mean = mean.view(1,mean.size(0),-1)
    mean = mean.expand(T,mean.size(1),-1).cuda() #50 300 20
    # std = var
    std =  torch.sqrt(var) #300,  kaifangyixia
    # std = torch.where(var >= 0, torch.sqrt(var), torch.tensor(0.0))
    std = std.view(1, std.size(0), 1)

    std = std.expand(T, std.size(1), n_classes).cuda() # 50 300 20
    dist = torch.randn([T,std.size(1),n_classes], dtype=torch.float).cuda() #50 300 20

    mean = mean + std*dist
    mean1 = mean / 0.18
    mean1 = F.softmax(mean1, dim=2)
    mean_softmax = mean1.mean(dim=0)
    mean1 = torch.log(mean_softmax) #[样本数，类数]  75x5

    target_inds = torch.arange(0, n_classes).cuda() #5
    target_inds = target_inds.view(n_classes, 1)  # 5x1
    target_inds = target_inds.expand(n_classes, n_query).long().contiguous().view(-1) #5x15
    target_inds1 = torch.full([n_classes * n_query, n_classes], 0.025).cuda().scatter_(1, target_inds.unsqueeze(1), 0.9)#75x5

    # loss = F.nll_loss(mean1, target_inds)
    loss =-mean1*target_inds1 #75x5
    loss = loss.sum(dim=1) #75x1
    loss = loss.mean() #1
    acc = count_acc(mean_softmax, target_inds)
    return loss,  acc




    # log_p_y = F.softmax(mean, dim=2)  #  50 300 20  负数
    # log_p_y = log_p_y.mean(dim=0).view(n_classes, n_query, -1) #20 15  20
    #
    #
    # target_inds = torch.arange(0, n_classes).cuda()
    # target_inds = target_inds.view(n_classes, 1, 1)  # 60x1
    # target_inds = target_inds.expand(n_classes, n_query, 1).long().cuda()
    # #print(target_inds)
    #
    # loss_val = log_p_y.gather(2, target_inds).squeeze().view(-1) #属于第一类就取出关于第一类的概率值，就相当于交叉熵操作了！！ 我们只要当前正确的类的概率值作为损失。其他都是0]
    #
    #
    # loss_val = -torch.log(loss_val)
    # loss_val = loss_val.mean()
    # _, y_hat = log_p_y.max(2)   #挑出距离类中心最近的索引，即当前样本所属的类别
    #
    # acc_val = y_hat.eq(target_inds.squeeze(2)).float().mean()
    # return loss_val,  acc_val




# def prototypical_loss(n_classes,n_query,mean, var):
#     '''
#     Inspired by https://github.com/jakesnell/prototypical-networks/blob/master/protonets/models/few_shot.py
#
#     Compute the barycentres by averaging the features of n_support
#     samples for each class in target, computes then the distances from each
#     samples' features to each one of the barycentres, computes the
#     log_probability for each n_query samples for each one of the current
#     classes, of appartaining to a class c, loss and accuracy are then computed
#     and returned
#     Args:
#     - input: the model output for a batch of samples
#     - target: ground truth for the above batch of samples
#     - n_support: number of samples to keep in account when computing
#       barycentres, for each one of the current classes
#     '''
#     T = 1000
#     mean = mean.view(1,mean.size(0),-1)
#     mean = mean.expand(T,mean.size(1),-1).cuda() #50 300 20
#
#     # std = var #300,1
#     std =  torch.sqrt(var) #300,
#     std = std.view(1, std.size(0), 1)
#
#     std = std.expand(T, std.size(1), n_classes).cuda() # 50 300 20
#     dist = torch.randn([T,std.size(1),n_classes], dtype=torch.float).cuda() #50 300 20
#
#     mean = mean + std*dist
#     mean = mean / 0.1
#
#
#     log_p_y = F.log_softmax(mean, dim=2)  #  50 300 20  负数
#     log_p_y = log_p_y.mean(dim=0).view(n_classes, n_query, -1) #5 15  5
#
#     target_inds = torch.arange(0, n_classes)
#     target_inds = target_inds.view(n_classes, 1, 1)  # 60x1
#     target_inds = target_inds.expand(n_classes, n_query, 1).long().cuda()
#
#     loss_val = -log_p_y.gather(2, target_inds).squeeze().view(-1).mean() #属于第一类就取出关于第一类的概率值，就相当于交叉熵操作了！！ 我们只要当前正确的类的概率值作为损失。其他都是0]
#
#     _, y_hat = log_p_y.max(2)   #挑出距离类中心最近的索引，即当前样本所属的类别
#
#     acc_val = y_hat.eq(target_inds.squeeze(2)).float().mean()
#     return loss_val,  acc_val


    # log_p_y = F.log_softmax(mean, dim=1).view(n_classes, n_query, -1)  #5x5x5?  把距离概率化了一下，又reshape了一下
    #
    # target_inds = torch.arange(0, n_classes)
    # target_inds = target_inds.view(n_classes, 1, 1) #60x1x1
    # target_inds = target_inds.expand(n_classes, n_query, 1).long().cuda() #60x5x1  groud truth   [0,0,0,0,0][1,1,1,1,1]....
    #
    #
    # loss_val = -log_p_y.gather(2, target_inds).squeeze().view(-1).mean() #属于第一类就取出关于第一类的概率值，就相当于交叉熵操作了！！ 我们只要当前正确的类的概率值作为损失。其他都是0
    # _, y_hat = log_p_y.max(2)   #挑出距离类中心最近的索引，即当前样本所属的类别
    # acc_val = y_hat.eq(target_inds.squeeze(2)).float().mean()
    #
    # return loss_val,  acc_val
