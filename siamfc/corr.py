from __future__ import absolute_import
import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np

class _corr(nn.Module):
    def __init__(self):
        super(_corr, self).__init__()
    #互相关运算，设batch_size=8
    def forward(self, z, x):
        kernel = z #[8,128,6,6]
        group = z.size(0)  #8
        input = x.view(-1, group*x.size(1), x.size(2), x.size(3))
        #输出为[8,1,17,17], 那么反推input[8,128,22,22]，kernel[1,1024,6,6] group=128/1024？错误
        #所以先输出[1,8,17,17],再view变换维度成[8,1,17,17],那么input[1,1024,22,22],kernel[8,128,6,6],group=1024/128=8=batch_size
        response_maps = F.conv2d(input, kernel,groups=group)
        response_maps = response_maps.view(x.size(0),-1,response_maps.size(2), response_maps.size(3))
        return response_maps
if __name__ == '__main__':
    # z = torch.randn([8, 128, 6, 6])
    z = torch.randn([1,128,6,6])
    x = torch.randn([8, 128, 22, 22])
    f = _corr()
    response_maps = f(z, x)


