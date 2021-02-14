from __future__ import absolute_import, division

import torch.nn as nn
import cv2
import numpy as np
import torch
from siamfc.config import cfg
import torch.nn.functional as F

def read_image(img_file, cvt_code=cv2.COLOR_BGR2RGB):  #将BGR格式转换成RGB格式，cv.imread都进来直接就是BGR，[w,h,c]
    img = cv2.imread(img_file, cv2.IMREAD_COLOR)  #cv2.imread函数读取图片，后面参数代表加载彩色图片，还有灰度图片等 返回的img为[weight,height,channel]
    if cvt_code is not None:  #这个判断可以省略，上面给出了cvt_code的具体值
        img = cv2.cvtColor(img, cvt_code)
    return img


def show_image(img, boxes=None, box_fmt='ltwh', colors=None,
               thickness=3, fig_n=1, delay=1, visualize=True,
               cvt_code=cv2.COLOR_RGB2BGR):
    if cvt_code is not None:
        img = cv2.cvtColor(img, cvt_code)   #要用cv2显示，要把RGB转化为BGR！！！
    
    # resize img if necessary 有必要的话resize 图片
    max_size = 960        #最大为960
    if max(img.shape[:2]) > max_size:
        scale = max_size / max(img.shape[:2])    #960/max（w，h）
        out_size = (
            int(img.shape[1] * scale),     #960/max（w，h）*h
            int(img.shape[0] * scale))     #960/max（w，h）*w
        img = cv2.resize(img, out_size)
        if boxes is not None:
            boxes = np.array(boxes, dtype=np.float32) * scale
    
    if boxes is not None:
        assert box_fmt in ['ltwh', 'ltrb']
        boxes = np.array(boxes, dtype=np.int32)    #boxes.shape(4,)
        if boxes.ndim == 1:  #boxes的维度是否为1
            boxes = np.expand_dims(boxes, axis=0)   #boxes.shape(1,4) #增加维度 axis=0,比如[2 2 3]变成[1 2 2 3] axis=1,[2 2 3]变成[2 1 2 3] 还有axis=2/3
        if box_fmt == 'ltrb':
            boxes[:, 2:] -= boxes[:, :2]
        
        # clip bounding boxes
        bound = np.array(img.shape[1::-1])[None, :] #img.shape[1::-1]表示[w,h,3]->[h,w,3] ,[None,:]表示[h,w]->[1,h,w,3]
        boxes[:, :2] = np.clip(boxes[:, :2], 0, bound) #boxes前两列
        boxes[:, 2:] = np.clip(boxes[:, 2:], 0, bound - boxes[:, :2]) #boxes后两列
        
        if colors is None:
            colors = [
                (0, 0, 255),
                (0, 255, 0),
                (255, 0, 0),
                (0, 255, 255),
                (255, 0, 255),
                (255, 255, 0),
                (0, 0, 128),
                (0, 128, 0),
                (128, 0, 0),
                (0, 128, 128),
                (128, 0, 128),
                (128, 128, 0)]     #len(colors)=12
        colors = np.array(colors, dtype=np.int32)  #colors.shape=[12 3]
        if colors.ndim == 1:
            colors = np.expand_dims(colors, axis=0)
        
        for i, box in enumerate(boxes):   #我的理解是这里的boxes只有1个
            color = colors[i % len(colors)]   #len(colors)=3
            pt1 = (box[0], box[1])
            pt2 = (box[0] + box[2], box[1] + box[3])
            img = cv2.rectangle(img, pt1, pt2, color.tolist(), thickness)
    
    if visualize:
        winname = 'window_{}'.format(fig_n)  #window_1   {}被格式化为1
        cv2.imshow(winname, img)
        # cv2.imshow('window_1',img)效果与上相同
        cv2.waitKey(delay)   #1秒更新一次

    return img

#裁剪一块以目标为中心，边长为size大小的patch,然后将其resize成out_size的大小
def crop_and_resize(img, center, size, out_size,
                    border_type=cv2.BORDER_CONSTANT,
                    border_value=(0, 0, 0), #border_value使用的是图像均值(averageR,aveG,aveB)
                    interp=cv2.INTER_LINEAR):
    # convert box to corners (0-indexed)
    size = round(size)  #对size取整
    corners = np.concatenate((         #np.concatenate:数组的凭借 np.concatenate((a,b),axis)  axis=0是列拼接，axis=1是行拼接 省略axis为0
        np.round(center - (size - 1) / 2),
        np.round(center - (size - 1) / 2) + size))   #得到corners=[ymin,xmin,ymax,xmax]
    corners = np.round(corners).astype(int)          #转化为int型
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # cv2.imshow('original', img)
    # cv2.imwrite('original.png', img)
    #填充
    pads = np.concatenate((
        -corners[:2], corners[2:] - img.shape[:2]))
    npad = max(0, int(pads.max()))     #得到4个值中最大的与0对比
    if npad > 0:
        img = cv2.copyMakeBorder(
            img, npad, npad, npad, npad,
            border_type, value=border_value)

    # crop image patch
    corners = (corners + npad).astype(int)    #如果经行了填充，那么中心坐标也要变
    patch = img[corners[0]:corners[2], corners[1]:corners[3]]     #得到patch的大小
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # cv2.imshow('padding_img',img)
    # cv2.imwrite('padding_img.png', img)
    # patch = cv2.cvtColor(patch, cv2.COLOR_BGR2RGB)
    # cv2.imshow('contest_img',patch)
    # cv2.imwrite('contest_img.png', patch)
    # resize to out_size
    patch = cv2.resize(patch, (out_size, out_size),
                       interpolation=interp)

    # cv2.imshow('resize255_img',patch)
    # cv2.imwrite('resize255_img.png', patch)
    # cv2.waitKey(0)
    return patch



# 建立标签 1 0
def _create_labels(size):

    def logistic_labels(x, y, r_pos):
        # x^2+y^2<4 的位置设为为1，其他为0
        dist = np.sqrt(x ** 2 + y ** 2)
        labels = np.where(dist <= r_pos,    #r_os=2
                          np.ones_like(x),  #np.ones_like(x),用1填充x
                          np.zeros_like(x)) #np.zeros_like(x),用0填充x
        return labels
    # distances along x- and y-axis
    n, c, h, w = size  # [8,1,15,15]
    x = np.arange(w) - (w - 1) / 2  #x=[-7 -6 ....0....6 7]
    y = np.arange(h) - (h - 1) / 2  #y=[-7 -6 ....0....6 7]
    x, y = np.meshgrid(x, y)
    #建立标签
    r_pos = cfg.r_pos / cfg.total_stride  # 16/8
    labels = logistic_labels(x, y, r_pos)
    #重复batch_size个label，因为网络输出是batch_size张response map
    labels = labels.reshape((1, 1, h, w))   #[1,1,15,15]
    labels = np.tile(labels, (n, c, 1, 1))  #将labels扩展[8,1,15,15]
    return labels



class BalancedLoss(nn.Module):
    def __init__(self, neg_weight=1.0):
        super(BalancedLoss, self).__init__()
        self.neg_weight = neg_weight

    def forward(self, input, target):  #属于BalancedLoss的forward
        pos_mask = (target == 1)    #相应位置标注为True（1）
        neg_mask = (target == 0)    #相应位置标注为True（0）
        pos_num = pos_mask.sum().float()    #计算True的个数（1）
        neg_num = neg_mask.sum().float()    #计算True的个数（0）
        weight = target.new_zeros(target.size())  #创建一个大小与target相同的weight
        weight[pos_mask] = 1 / pos_num
        weight[neg_mask] = 1 / neg_num * self.neg_weight
        weight /= weight.sum()
        #binary_cross_entropy_with_logits等价于sigmod+F.binary_cross_entropy！！！
        return F.binary_cross_entropy_with_logits(  #torch.nn.functional.binary_cross_entropy_with_logits
            input, target, weight, reduction='sum')

if __name__ == '__main__':
    labels = _create_labels([8,1,15,15])
