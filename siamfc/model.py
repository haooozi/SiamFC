from __future__ import absolute_import
import torch.nn as nn
import torch

#特征提取网络使用带批量归一化(BatchNorm2d)的AlexNet
class AlexNet(nn.Module):
    output_stride = 8
    def __init__(self,init_weight=True):
        super(AlexNet, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 96, 11, 2),    #[1,3,127,127]->[1,96,59,59]
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2))         #[1,96,59,59]->[1,96,29,29]
        self.conv2 = nn.Sequential(
            nn.Conv2d(96, 256, 5, 1),   #[1,96,29,29]->[1,256,25,25]
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2))         #[1,256,25,25]->[1,256,12,12]
        self.conv3 = nn.Sequential(
            nn.Conv2d(256, 384, 3, 1),  #[1,256,12,12]->[1,384,10,10]
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True))
        self.conv4 = nn.Sequential(
            nn.Conv2d(384, 384, 3, 1),  #[1,384,10,10]->[1,384,8,8]
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True))
        self.conv5 = nn.Sequential(
            nn.Conv2d(384, 128, 3, 1))   #[1,384,8,8]->[1,128,6,6]
        #初始化网络参数
        if init_weight:
            self._init_weights()

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        return x

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight, 1) #xavier是参数初始化，它的初始化思想是保持输入和输出方差一致，这样就避免了所有输出值都趋向于0
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)     #偏置初始化为0
            elif isinstance(m, nn.BatchNorm2d):      #在激活函数之前，希望输出值由较好的分布，以便于计算梯度和更新参数，这时用到BatchNorm2d函数
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

if __name__ == '__main__':
    model = AlexNet()
    z = torch.randn([1, 3, 255, 255])
    feature_map_z = model(z)

