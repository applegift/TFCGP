# coding=utf-8
import numpy as np
import torch


class PrototypicalBatchSampler(object):
    '''
    PrototypicalBatchSampler: yield a batch of indexes at each iteration.
    Indexes are calculated by keeping in account 'classes_per_it' and 'num_samples',
    In fact at every iteration the batch indexes will refer to  'num_support' + 'num_query' samples
    for 'classes_per_it' random classes.

    __len__ returns the number of episodes per epoch (same as 'self.iterations').
    '''

    def __init__(self, labels, classes_per_it, num_samples, iterations):
        '''
        Initialize the PrototypicalBatchSampler object
        Args:
        - labels: an iterable containing all the labels for the current dataset samples indexes will be infered from this iterable.
        - classes_per_it: number of random classes for each iteration
        - num_samples: number of samples for each iteration for each class (support + query)
        - iterations: number of iterations (episodes) per epoch
        '''
        super(PrototypicalBatchSampler, self).__init__()
        self.labels = labels

        self.classes_per_it = classes_per_it
        self.sample_per_class = num_samples
        self.iterations = iterations

        self.classes, self.counts = np.unique(self.labels, return_counts=True)
        #counts = 20  每个类中有20个样本
        self.classes = torch.LongTensor(self.classes)

        # create a matrix, indexes, of dim: classes X max(elements per class)
        # fill it with nans
        # for every class c, fill the relative row with the indices samples belonging to c
        # in numel_per_class we store the number of samples for each class/row
        self.idxs = range(len(self.labels))
        # print(self.idxs)
        # == Dataset: Found 82240 items
        # == Dataset: Found 4112 classes
        # range(0, 82240)

        self.indexes = np.empty((len(self.classes), max(self.counts)), dtype=int) * np.nan #类别数（4112）x每个类的元素个数（20）矩阵
        self.indexes = torch.Tensor(self.indexes)
        self.numel_per_class = torch.zeros_like(self.classes) #生成全是0的张量

        for idx, label in enumerate(self.labels):
            #idx 索引， label是每个样本的标签
            #indexes存储着这个类的20个样本的索引
            label_idx = np.argwhere(self.classes == label).item()  #返回所有self.classes == label的索引
            self.indexes[label_idx, np.where(np.isnan(self.indexes[label_idx]))[0][0]] = idx #np.where()会返回一个元组，里面是满足目标条件的索引，这里使用两个[0]的原因是:第一个[0]确定了第一个维度，第二个确定了当前维度的第一个值
            self.numel_per_class[label_idx] += 1


    def __iter__(self):
        '''
        yield a batch of indexes
        '''
        spc = self.sample_per_class
        cpi = self.classes_per_it
        # torch.manual_seed(1000)
        # torch.cuda.manual_seed(1000)
        for it in range(self.iterations):
            batch_size = spc * cpi
            batch = torch.LongTensor(batch_size) #batch_size就是一个标量，torch.LongTensor生成batch_size大小的tensor，
            c_idxs = torch.randperm(len(self.classes))[:cpi]
            for i, c in enumerate(self.classes[c_idxs]):
                s = slice(i * spc, (i + 1) * spc) #slice切片返回的是索引
                # FIXME when torch.argwhere will exists
                label_idx = torch.arange(len(self.classes)).long()[self.classes == c].item() #返回了一个类别数都是c的索引
                #import torch

                # # 创建一个示例张量
                # x = torch.tensor([1, 2, 3, 4, 5])
                #
                # # 筛选等于2的元素
                # condition = x == 2
                # filtered_tensor = x[condition]
                #
                # print(filtered_tensor)
                #output : tensor([2])

                sample_idxs = torch.randperm(self.numel_per_class[label_idx])[:spc]
                batch[s] = self.indexes[label_idx][sample_idxs]
            batch = batch[torch.randperm(len(batch))] #将生成的随机排列的整数序列作为索引操作符的参数传递给batch张量。
            yield batch

    def __len__(self):
        '''
        returns the number of iterations (episodes) per epoch
        '''
        return self.iterations
