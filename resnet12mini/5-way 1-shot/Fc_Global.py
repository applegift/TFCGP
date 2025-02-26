import torch
import torch.nn as nn


class fc_block(nn.Module):
    def __init__(self):
        super(fc_block, self).__init__()
        channel = 100
        self.encoder = nn.Sequential(
            nn.Linear(channel, channel // 4, bias=False),
            nn.Linear(channel // 4, channel, bias=False)
        )

    def forward(self, x):
        x = self.encoder(x)
        return x


# class ChannelAttention(nn.Module):
#     def __init__(self, in_planes, ratio=5):
#         super(ChannelAttention, self).__init__()
#         self.avg_pool = nn.AdaptiveAvgPool2d(1)
#         self.max_pool = nn.AdaptiveMaxPool2d(1)
#
#         # 利用1x1卷积代替全连接
#         self.fc1   = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
#         self.relu1 = nn.ReLU()
#         self.fc2   = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
#
#         self.sigmoid = nn.Sigmoid()
#
#     def forward(self, x):
#         avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
#         max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
#         out = avg_out + max_out
#         return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        # assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        if kernel_size == 1:
            padding = 0
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class cbam_block(nn.Module):
    def __init__(self, channel, ratio=5, kernel_size=3):
        super(cbam_block, self).__init__()
        self.atts = nn.Parameter(torch.tensor(1.0))
        #
        # self.gplength = nn.Parameter(torch.tensor(20.0))
        # self.channelattention = ChannelAttention(channel, ratio=ratio)
        self.spatialattention = SpatialAttention(kernel_size=kernel_size)

    def forward(self, x):
        # x = x * self.channelattention(x)
        # x = x * self.spatialattention(x)
        x = self.spatialattention(x)
        return x
