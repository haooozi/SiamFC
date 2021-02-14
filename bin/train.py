from __future__ import absolute_import, division, print_function

import os
from got10k.datasets import *
from siamfc.datasets import GOT10kDataset
from siamfc.datasets import SiamFCTransforms
from siamfc.config import cfg
from torch.utils.data import DataLoader
from siamfc.model import AlexNet
from siamfc.corr import _corr
import torch
from torch import optim
import numpy as np
from torch.optim.lr_scheduler import ExponentialLR
from tqdm import tqdm
from siamfc.utils import _create_labels
from siamfc.utils import BalancedLoss
os.environ["CUDA_VISIBLE_DEVICES"] = "0" #多卡情况下默认多卡训练,如果想单卡训练,设置为"0"

def train(data_dir, net_path=None,save_dir='pretrained'):
    #从文件中读取图像数据集
    seq_dataset = GOT10k(data_dir,subset='train',return_meta=False)
    #定义图像预处理方法
    transforms = SiamFCTransforms(
        exemplar_sz=cfg.exemplar_sz, #127
        instance_sz=cfg.instance_sz, #255
        context=cfg.context) #0.5
    #从读取的数据集每个视频序列配对训练图像并进行预处理，裁剪等
    train_dataset = GOT10kDataset(seq_dataset,transforms)
    #加载训练数据集
    loader_dataset = DataLoader( dataset = train_dataset,
                                 batch_size=cfg.batch_size,
                                 shuffle=True,
                                 num_workers=cfg.num_workers,
                                 pin_memory=True,
                                 drop_last=True, )
    #初始化训练网络
    cuda = torch.cuda.is_available()  #支持GPU为True
    device = torch.device('cuda:0' if cuda else 'cpu')  #cuda设备号为0
    model = AlexNet(init_weight=True)
    corr = _corr()
    model = model.to(device)
    corr = corr.to(device)
    # 设置损失函数和标签
    logist_loss = BalancedLoss()
    labels = _create_labels(size=[cfg.batch_size, 1, cfg.response_sz - 2, cfg.response_sz - 2])
    labels = torch.from_numpy(labels).to(device).float()
    #建立优化器，设置指数变化的学习率
    optimizer = optim.SGD(
        model.parameters(),
        lr=cfg.initial_lr,              #初始化的学习率，后续会不断更新
        weight_decay=cfg.weight_decay,  #λ=5e-4，正则化
        momentum=cfg.momentum)          #v(now)=−dx∗lr+v(last)∗momemtum
    gamma = np.power(                   #np.power(a,b) 返回a^b
        cfg.ultimate_lr / cfg.initial_lr,
        1.0 / cfg.epoch_num)
    lr_scheduler = ExponentialLR(optimizer, gamma)  #指数形式衰减，lr=initial_lr*(gamma^epoch)
    """————————————————————————开始训练——————————————————————————"""
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    start_epoch = 1
    #接着上一次训练，提取训练结束保存的net、optimizer、epoch参数
    if net_path is not None:
        checkpoint = torch.load(net_path)
        if 'epoch' in checkpoint:
            start_epoch = checkpoint['epoch'] + 1
            model.load_state_dict(checkpoint['model'])
            optimizer.load_state_dict(checkpoint['optimizer'])
        else:
            model.load_state_dict(checkpoint)
        del checkpoint
        torch.cuda.empty_cache()  #缓存清零
        print("loaded checkpoint!!!")
    for epoch in range(start_epoch, cfg.epoch_num+1):
        model.train()
        #遍历训练集
        for it, batch in enumerate(tqdm(loader_dataset)):
            z = batch[0].to(device,non_blocking=cuda)   # z.shape=([8,3,127,127])
            x = batch[1].to(device, non_blocking=cuda)  # x.shape=([8,3,239,239])
            #输入网络后通过损失函数
            z, x = model(z), model(x)
            responses = corr(z, x) * cfg.out_reduce  # 返回的是heatmap的响应表15x15  因为x是239x239 [8,1,15,15]
            loss = logist_loss(responses, labels)
            #back propagation反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # if (it+1) % 20 ==0:
            print('Epoch: {}[{}/{}]    Loss: {:.5f}    lr: {:.2e}'.format(
                epoch , it + 1, len(loader_dataset), loss.item(),optimizer.param_groups[0]['lr']))
        #更新学习率 (每个epoch)
        lr_scheduler.step()
        #save checkpoint 做玩1个epoch后保存1个输出模型
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        save_path = os.path.join(
            save_dir, 'siamfc_alexnet_e%d.pth' % (epoch))

        torch.save({'epoch':epoch,
                    'model':model.state_dict(),
                    'optimizer':optimizer.state_dict()},save_path)

if __name__ == '__main__':
    train('D:\Dataset\GOT-10k')
