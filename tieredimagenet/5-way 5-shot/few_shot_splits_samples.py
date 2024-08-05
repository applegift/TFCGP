import torch


def split_samples(input, target, n_support):
    target_cpu = target.to('cuda')
    input_cpu = input.to('cuda')

    def supp_idxs(c): #用于返回目标张量中满足条件 target_cpu.eq(c) 的元素索引
        # FIXME when torch will support where as np
        return target_cpu.eq(c).nonzero()[:n_support].squeeze(1)  #true = 1 false = 0 这个张量的形状为 (num_nonzero, num_dimensions)，其中 num_nonzero 是非零元素的个数，num_dimensions 是张量的维度数。

    # FIXME when torch.unique will be available on cuda too
    classes = torch.unique(target_cpu)

    n_query = target_cpu.eq(classes[0].item()).sum().item() - n_support  #为了求查询集样本个数

    # FIXME when torch will support where as np

    support_idxs = list(map(supp_idxs, classes)) #support_idxs: list:60 每个类有5个sample， 一共300

    prototypes = torch.stack([input_cpu[idx_list] for idx_list in support_idxs]).reshape(-1,input.shape[1],input.shape[2],input.shape[3]) #shape: 60 x 64  经过了mean的处理
    # FIXME when torch will support where as np
    query_idxs = torch.stack(list(map(lambda c: target_cpu.eq(c).nonzero()[n_support:], classes))).view(-1) #shape : 300
    query_samples = input_cpu[query_idxs] #300x64
    return prototypes,query_samples,len(classes),n_query