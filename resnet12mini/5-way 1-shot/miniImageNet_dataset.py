# coding=utf-8
from __future__ import print_function
import os
import os.path as osp

import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from typing import Sequence
import torchvision.transforms.functional as TF
import random

class MyRotateTransform:
    def __init__(self, angles: Sequence[int]):
        self.angles = angles

    def __call__(self, x):
        angle = random.choice(self.angles)
        return TF.rotate(x, angle)
class MiniDataset(Dataset):

    def __init__(self, mode='train', root='..'):
        IMAGE_PATH = os.path.join(root, 'miniimageNet/images')
        SPLIT_PATH = os.path.join(root, 'miniimageNet')

        csv_path = osp.join(SPLIT_PATH, mode + '.csv')
        lines = [x.strip() for x in open(csv_path, 'r').readlines()][1:]

        data = []
        label = []
        lb = -1

        self.wnids = []

        for l in lines:
            name, wnid = l.split(',')
            path = osp.join(IMAGE_PATH, name)
            if wnid not in self.wnids:
                self.wnids.append(wnid)
                lb += 1
            data.append(path)
            label.append(lb)

        self.data = data
        self.label = label
        print("== Dataset: Found %d items " % len(self.label))

        self.num_class = len(set(label))
        print("== Dataset: Found %d classes" % self.num_class)

        # Transformation
        if mode == 'val' or mode == 'test':
            image_size = 112
            self.transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
        elif mode == 'train':
            image_size = 112
            self.transform = transforms.Compose([
                transforms.Resize((120, 120)),
                transforms.RandomCrop(112),
                transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        path, label = self.data[i], self.label[i]
        image = self.transform(Image.open(path).convert('RGB'))
        return image, label














# coding=utf-8
# from __future__ import print_function
# import os
# import os.path as osp
#
# import numpy as np
# from PIL import Image
# from torch.utils.data import Dataset
# from torchvision import transforms
# from typing import Sequence
# import torchvision.transforms.functional as TF
# import random

# class MyRotateTransform:
#     def __init__(self, angles: Sequence[int]):
#         self.angles = angles
#
#     def __call__(self, x):
#         angle = random.choice(self.angles)
#         return TF.rotate(x, angle)
# class MiniDataset(Dataset):
#
#     def __init__(self, mode='train', root='..' + os.sep + 'dataset2', transform=None, target_transform=None,return_path=False):
#         TRAIN_PATH = osp.join(root, 'data/train')
#         VAL_PATH = osp.join(root, 'data/val')
#         TEST_PATH = osp.join(root, 'data/test')
#         if mode == 'train':
#             THE_PATH = TRAIN_PATH
#         elif mode == 'test':
#             THE_PATH = TEST_PATH
#         elif mode == 'val':
#             THE_PATH = VAL_PATH
#         else:
#             raise ValueError('Wrong mode.')
#         data = []
#         label = []
#         folders = [osp.join(THE_PATH, label) for label in os.listdir(THE_PATH) if
#                    os.path.isdir(osp.join(THE_PATH, label))]
#         folders.sort()
#
#         for idx in range(len(folders)):
#             this_folder = folders[idx]
#             this_folder_images = os.listdir(this_folder)
#             this_folder_images.sort()
#             for image_path in this_folder_images:
#                 data.append(osp.join(this_folder, image_path))
#                 label.append(idx)
#
#         self.data = data
#         self.label = label
#         print("== Dataset: Found %d items " % len(self.label))
#
#         self.num_class = len(set(label))
#         print("== Dataset: Found %d classes" % self.num_class)
#         self.return_path = return_path
#
#         # Transformation
#         if mode == 'val' or mode == 'test':
#             image_size = 84
#             self.transform = transforms.Compose([
#                 transforms.Resize((image_size, image_size)),
#                 transforms.ToTensor(),
#                 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
#             ])
#         elif mode == 'train':
#             image_size = 84
#             self.transform = transforms.Compose([
#                 transforms.Resize((84, 84)),
#                 transforms.RandomCrop(84),
#                 transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
#                 transforms.RandomHorizontalFlip(),
#                 transforms.ToTensor(),
#                 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
#             ])
#
#     def __len__(self):
#         return len(self.data)
#
#     def __getitem__(self, i):
#         path, label = self.data[i], self.label[i]
#         image = self.transform(Image.open(path).convert('RGB'))
#         if self.return_path:
#             return image, label, path
#         else:
#             return image, label





# class MyRotateTransform:
#     def __init__(self, angles: Sequence[int]):
#         self.angles = angles
#
#     def __call__(self, x):
#         angle = random.choice(self.angles)
#         return TF.rotate(x, angle)
# class MiniDataset(Dataset):
#
#     def __init__(self, mode='train', root='..'):
#         IMAGE_PATH = os.path.join(root, 'miniimageNet/images')
#         SPLIT_PATH = os.path.join(root, 'miniimageNet')
#
#         csv_path = osp.join(SPLIT_PATH, mode + '.csv')
#         lines = [x.strip() for x in open(csv_path, 'r').readlines()][1:]
#
#         data = []
#         label = []
#         lb = -1
#
#         self.wnids = []
#
#         for l in lines:
#             name, wnid = l.split(',')
#             path = osp.join(IMAGE_PATH, name)
#             if wnid not in self.wnids:
#                 self.wnids.append(wnid)
#                 lb += 1
#             data.append(path)
#             label.append(lb)
#
#         self.data = data
#         self.label = label
#         print("== Dataset: Found %d items " % len(self.label))
#
#         self.num_class = len(set(label))
#         print("== Dataset: Found %d classes" % self.num_class)
#
#         # Transformation
#         if mode == 'val' or mode == 'test':
#             image_size = 84
#             self.transform = transforms.Compose([
#                 transforms.Resize((image_size, image_size)),
#                 transforms.ToTensor(),
#                 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
#         elif mode == 'train':
#             image_size = 84
#             self.transform = transforms.Compose([
#                 transforms.Resize((100, 100)),
#                 transforms.RandomCrop(84),
#                 transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
#                 transforms.RandomHorizontalFlip(),
#                 transforms.ToTensor(),
#                 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
#             ])
#
#     def __len__(self):
#         return len(self.data)
#
#     def __getitem__(self, i):
#         path, label = self.data[i], self.label[i]
#         image = self.transform(Image.open(path).convert('RGB'))
#         return image, label


# # coding=utf-8
# from __future__ import print_function
# import torch.utils.data as data
# from PIL import Image
# import numpy as np
# import shutil
# import errno
# import torch
# import os
#
# '''
# Inspired by https://github.com/pytorch/vision/pull/46
# '''
#
# IMG_CACHE = {}
#
#
# class MiniDataset(data.Dataset):
#     vinalys_baseurl = 'https://raw.githubusercontent.com/jakesnell/prototypical-networks/master/data/omniglot/splits/vinyals/'
#
#     vinyals_split_sizes = {
#         'test': vinalys_baseurl + 'test.txt',
#         'train': vinalys_baseurl + 'train.txt',
#         'trainval': vinalys_baseurl + 'trainval.txt',
#         'val': vinalys_baseurl + 'val.txt',
#     }
#
#     urls = [
#         'https://github.com/brendenlake/omniglot/raw/master/python/images_background.zip',
#         'https://github.com/brendenlake/omniglot/raw/master/python/images_evaluation.zip'
#     ]
#
#     splits_folder = os.path.join('splits', 'vinyals')
#     raw_folder = 'raw'
#     processed_folder = 'data'
#
#     def __init__(self, mode='train', root='..' + os.sep + 'dataset2', transform=None, target_transform=None, download=False):
#         '''
#         The items are (filename,category). The index of all the categories can be found in self.idx_classes
#         Args:
#         - root: the directory where the dataset will be stored
#         - transform: how to transform the input
#         - target_transform: how to transform the target
#         - download: need to download the dataset
#         '''
#
#         super(MiniDataset, self).__init__()
#         self.root = root
#         self.transform = transform
#         self.target_transform = target_transform
#
#
#         if not self._check_exists():
#             raise RuntimeError(
#                 'Dataset not found. You can use download=True to download it')
#
#         self.classes = get_current_classes(os.path.join(
#             self.root, self.splits_folder, mode + '.csv'))
#
#         self.all_items = find_items(os.path.join(
#             self.root, self.processed_folder), self.classes)
#
#         self.idx_classes = index_classes(self.all_items) #统计所有的类个数（加上旋转的）
#
#         paths, self.y = zip(*[self.get_path_label(pl)
#                               for pl in range(len(self))])
#         # print(paths)
#         #'D:\\code1\\Prototypical-Networks-for-Few-shot-Learning-PyTorch-master\\Prototypical-Networks-for-Few-shot-Learning-PyTorch-master\\dataset2\\data\\train\\n13133613\\n1313361300001286.jpg\\rot180',
#         # print(paths)
#         #'..\\dataset\\data\\images_evaluation\\Gurmukhi\\character19\\1178_11.png\\rot090'
#
#         self.x = map(load_img, paths, range(len(paths)))
#         self.x = list(self.x)
#         #查看self.x的形状
#
#
#
#     def __getitem__(self, idx):
#         x = self.x[idx]
#         if self.transform:
#             x = self.transform(x)
#         return x, self.y[idx]
#
#     def __len__(self):
#         return len(self.all_items)
#
#     def get_path_label(self, index):
#         filename = self.all_items[index][0]
#         rot = self.all_items[index][-1]
#         img = str.join(os.sep, [self.all_items[index][2], filename]) + rot #使用os.sep去连结后面的列表（str.join的用法）
#         # img = str.join(os.sep, [self.all_items[index][2], filename])#使用os.sep去连结后面的列表（str.join的用法）
#         # 例 ..\dataset\data\images_evaluation\Glagolitic\character35\1149_10.png\rot180
#
#         target = self.idx_classes[self.all_items[index]
#                                   [1] + self.all_items[index][-1]]
#         if self.target_transform is not None:
#             target = self.target_transform(target)
#         #target： 4098。4099。4100...
#         return img, target
#
#     def _check_exists(self):
#         return os.path.exists(os.path.join(self.root, self.processed_folder))
#
# def find_items(root_dir, classes):
#     #对样本进行划分,return格式为：('0716_11.png', 'Alphabet_of_the_Magi\\character08', '..\\dataset\\data\\images_background\\Alphabet_of_the_Magi\\character08', '\\rot180')
#     retour = []
#     rots = [os.sep + 'rot000', os.sep + 'rot090', os.sep + 'rot180', os.sep + 'rot270']
#     for (root, dirs, files) in os.walk(root_dir):
#         for f in files:
#             # print(root) #D:\code1\Prototypical-Networks-for-Few-shot-Learning-PyTorch-master\Prototypical-Networks-for-Few-shot-Learning-PyTorch-master\dataset2\data\test\n01981276
#             r = root.split(os.sep)
#             # print(r) #['D:', 'code1', 'Prototypical-Networks-for-Few-shot-Learning-PyTorch-master', 'Prototypical-Networks-for-Few-shot-Learning-PyTorch-master', 'dataset2', 'data', 'test', 'n01981276']
#             #按照os.sep进行切分
#             lr = len(r)
#             label = r[lr - 1]
#             # print(label) #test\n01981276
#
#
#             #如果f,label在classes中，且f以jpg结尾
#             # if ((f +',' + label) in classes) and f.endswith("jpg"):
#             #     retour.extend([(f, label, root)])
#
#             for rot in rots:
#                 #如果f,label在classes中，且f以jpg结尾
#                 if ((f +',' + label) in classes) and f.endswith("jpg"):
#                     # print([(f, label, root, rot)]) #[('n0211127700000412.jpg', 'n02111277', 'D:\\code1\\Prototypical-Networks-for-Few-shot-Learning-PyTorch-master\\Prototypical-Networks-for-Few-shot-Learning-PyTorch-master\\dataset2\\data\\train\\n02111277', '\\rot180')]
#                     retour.extend([(f, label, root, rot)])
#
#
#     print("== Dataset: Found %d items " % len(retour))
#     return retour
#
#
# def index_classes(items):
#     # 对类进行统计个数：如
#     # 'Aurek-Besh\\character11\\rot270': 3499,
#     # 'Aurek-Besh\\character12\\rot000': 3500,
#     # 'Aurek-Besh\\character12\\rot090': 3501,
#     # 'Aurek-Besh\\character12\\rot180': 3502,
#     # 'Aurek-Besh\\character12\\rot270': 3503,
#     idx = {}
#     for i in items:
#
#         if (not i[1] + i[-1] in idx):
#             idx[i[1] + i[-1]] = len(idx)
#
#     print("== Dataset: Found %d classes" % len(idx))
#     return idx
#
#
# def get_current_classes(fname):
#     with open(fname) as f:
#         classes = f.read().replace('/', os.sep).splitlines()
#     return classes
#
#
# def load_img(path, idx):
#     path, rot = path.split(os.sep + 'rot')
#
#     if path in IMG_CACHE:
#         x = IMG_CACHE[path]
#     else:
#         x = Image.open(path)
#         IMG_CACHE[path] = x
#
#     x = x.rotate(float(rot))
#     # 查看x的维度
#
#     shape =3, x.size[0], x.size[1]
#     x = np.array(x, np.float32, copy=False)
#     #彩色图像的RGB通道值分别为0-255，为了便于计算，需要将其归一化到0-1之间
#     # x = x / 255.
#     # x = 1.0 - torch.from_numpy(x)
#     x = torch.from_numpy(x)
#
#
#     x = x.transpose(0, 1).contiguous().view(shape)
#
#     return x
