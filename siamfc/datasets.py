from __future__ import absolute_import, division

from got10k.datasets import *
import numpy as np
import cv2
from torch.utils.data import Dataset
from siamfc.config import cfg
import numbers
import torch
from . import utils


__all__ = ['GOT10kDataset','SiamFCTransforms']


# 就是把一系列的transforms串起来
class Compose(object):  # 继承了object类，就拥有了object类里面好多可以操作的对象

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img):  # 为了将类的实例对象变为可调用对象（相当于重载()运算符）  a=Compose() a.__call__()   和a()的使用是一样的
        for t in self.transforms:
            img = t(img)
        return img


# 主要是随机的resize图片的大小，变化再[1 1.05之内]其中要注意cv2.resize()的一点用法
class RandomStretch(object):

    def __init__(self, max_stretch=0.05):
        self.max_stretch = max_stretch

    def __call__(self, img):
        interp = np.random.choice([  # 调用interp时候随机选择一个
            cv2.INTER_LINEAR,  # 双线性插值（默认设置）
            cv2.INTER_CUBIC,  # 4x4像素领域的双三次插值
            cv2.INTER_AREA,  # 像素区域关系重采样，类似与NEAREST
            cv2.INTER_NEAREST,  # 最近领插值
            cv2.INTER_LANCZOS4])  # 8x8像素的Lanczosc插值
        scale = 1.0 + np.random.uniform(
            -self.max_stretch, self.max_stretch)
        out_size = (
            round(img.shape[1] * scale),  # 这里是width
            round(img.shape[0] * scale))  # 这里是heigth  cv2的用法导致
        return cv2.resize(img, out_size, interpolation=interp)  # 将img的大小resize成out_size


# 从img中心抠一块(size, size)大小的patch，如果不够大，以图片均值进行pad之后再crop
class CenterCrop(object):

    def __init__(self, size):
        if isinstance(size, numbers.Number):  # isinstance(object, classinfo) 判断实例是否是这个类或者object是变量
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, img):
        h, w = img.shape[:2]  # img.shape为[height,width,channel]
        tw, th = self.size
        i = round((h - th) / 2.)  # round(x,n) 对x四舍五入，保留n位小数 省略n 0位小数
        j = round((w - tw) / 2.)

        npad = max(0, -i, -j)
        if npad > 0:
            avg_color = np.mean(img, axis=(0, 1))  # 取整个图片的像素均值
            img = cv2.copyMakeBorder(  # 添加边框函数，上下左右要扩展的像素数都是npad,BORDER_CONSTANT固定值填充，值为avg_color）
                img, npad, npad, npad, npad,
                cv2.BORDER_CONSTANT, value=avg_color)
            i += npad
            j += npad

        return img[i:i + th, j:j + tw]


# 用法类似CenterCrop，只不过从随机的位置抠，没有pad的考虑
class RandomCrop(object):

    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, img):
        h, w = img.shape[:2]
        tw, th = self.size
        i = np.random.randint(0, h - th + 1)
        j = np.random.randint(0, w - tw + 1)
        return img[i:i + th, j:j + tw]


# 就是字面意思，把np.ndarray转化成torch tensor类型
class ToTensor(object):
    def __call__(self, img):
        return torch.from_numpy(img).float().permute((2, 0,
                                                      1))  # permute是维度变换，比如[1 24 24]--->[24 1 24]按照(2,0,1)变换,[height,weight,channel]变为[channel height weight]


class SiamFCTransforms(object):
    def __init__(self, exemplar_sz=127, instance_sz=255, context=0.5):
        self.exemplar_sz = exemplar_sz
        self.instance_sz = instance_sz
        self.context = context
        # transforms_z/x是数据增强方法
        self.transforms_z = Compose([
            RandomStretch(),  # 随机resize图片大小,变化再[1 1.05]之内
            CenterCrop(instance_sz - 8),  # 中心裁剪 裁剪为255-8
            RandomCrop(instance_sz - 2 * 8),  # 随机裁剪  255-8->255-8-8
            CenterCrop(exemplar_sz),  # 中心裁剪 255-8-8->127
            ToTensor()])  # 图片的数据格式从numpy转换成torch张量形式
        self.transforms_x = Compose([
            RandomStretch(),  # s随机resize图片
            CenterCrop(instance_sz - 8),  # 中心裁剪 裁剪为255-8
            RandomCrop(instance_sz - 2 * 8),  # 随机裁剪 255-8->255-8-8
            ToTensor()])  # 图片数据格式转化为torch张量

    def __call__(self, z, x, box_z, box_x):  # z，x表示传进来的图像
        z = self._crop(z, box_z,self.instance_sz)  # 对z(x类似)图像 1、box转换(l,t,w,h)->(y,x,h,w)，并且数据格式转为float32,得到center[y,x],和target_sz[h,w]
        x = self._crop(x, box_x, self.instance_sz)  # 2、得到size=((h+(h+w)/2)*(w+(h+2)/2))^0.5*255(instance_sz)/127
        z = self.transforms_z(z)  # 3、进入crop_and_resize:传入z作为图片img，center，size，outsize=255(instance_sz),随机选方式填充，均值填充
        x = self.transforms_x(x)  # 以center为中心裁剪一块边长为size大小的正方形框(注意裁剪时的padd边框填充问题)，再resize成out_size=255(instance_sz)
        return z, x

    def _crop(self, img, box, out_size):
        # convert box to 0-indexed and center based [y, x, h, w]  因为GOT-10k里面目标的bbox是以（left，top，weight，height）的形式给出
        #  我们要把它变为[y x height weight]  [y,x]为中心点坐标
        box = np.array([
            box[1] - 1 + (box[3] - 1) / 2,
            box[0] - 1 + (box[2] - 1) / 2,
            box[3], box[2]], dtype=np.float32)  # 数据格式转为numpy的float32
        center, target_sz = box[:2], box[2:]

        context = self.context * np.sum(target_sz)  # 0.5*(h+w)
        size = np.sqrt(np.prod(target_sz + context))  # sqrt开根号   prod用来计算所有元素的乘积size=(（h+(h+w)/2）*(w+(h+2)/2))^0.5
        size *= out_size / self.exemplar_sz

        avg_color = np.mean(img, axis=(0, 1), dtype=float)  # 对整个图片取均值，数据转为float型
        interp = np.random.choice([
            cv2.INTER_LINEAR,
            cv2.INTER_CUBIC,
            cv2.INTER_AREA,
            cv2.INTER_NEAREST,
            cv2.INTER_LANCZOS4])
        patch = utils.crop_and_resize(
            img, center, size, out_size,
            border_value=avg_color, interp=interp)

        return patch






class GOT10kDataset(Dataset): #继承了torch.utils.data的Dataset类
    def __init__(self, seqs, transforms=None,
                 pairs_per_seq=1):
        super(GOT10kDataset, self).__init__()
        self.seqs = seqs
        self.transforms = transforms
        self.pairs_per_seq = pairs_per_seq
        self.indices = np.random.permutation(len(seqs))  #len(seqs)=9335，随机打乱0-9335 如果没有permutation 那么indices=array([0 1 2 3... 9335])
        self.return_meta = getattr(seqs, 'return_meta')  #判断return_meta是否在segs中，如果不在，返回False，在的话返回1
#通过index索引返回item=（z,x,box_z,box_x）,然后经过transforms返回一对pair(z,x)
    def __getitem__(self, index):               #__getitem__的作用就是根据索引index遍历数据，并且可以在该函数下面对数据进行处理
        # print(self.indices)
        index = self.indices[index % len(self.indices)]
        # print(index)
        # index = self.indices[index] 与上相同
        # get filename lists and annotations
        if self.return_meta:  #如果为True的话
            img_files, anno, meta = self.seqs[index]
            vis_ratios = meta.get('cover', None)
        else:
            img_files, anno = self.seqs[index][:2]
            vis_ratios = None
        
        # filter out noisy frames
        val_indices = self._filter(
            cv2.imread(img_files[0], cv2.IMREAD_COLOR),
            anno, vis_ratios)
        if len(val_indices) < 2:
            index = np.random.choice(len(self))
            return self.__getitem__(index)

        #得到有效的两帧图片
        rand_z, rand_x = self._sample_pair(val_indices)

        z = cv2.imread(img_files[rand_z], cv2.IMREAD_COLOR)
        x = cv2.imread(img_files[rand_x], cv2.IMREAD_COLOR)
        z = cv2.cvtColor(z, cv2.COLOR_BGR2RGB)
        x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
        
        box_z = anno[rand_z]
        box_x = anno[rand_x]

        item = (z, x, box_z, box_x)   #box就是ground_truth
        if self.transforms is not None:
            item = self.transforms(*item)
        
        return item
#这里定义的长度就是被索引到的视频序列数x每个序列提供的对数（1对）
    def __len__(self):    #__len__的作用就是返回数据集的长度
        return len(self.indices) * self.pairs_per_seq   #len(self.indices)=9335  返回9335*1对
#随机挑选两个索引，这里取的间隔不超过T=100
    def _sample_pair(self, indices):
        n = len(indices)
        assert n > 0
        if n == 1:
            return indices[0], indices[0]
        elif n == 2:
            return indices[0], indices[1]
        else:
            for i in range(100):
                rand_z, rand_x = np.sort(
                    np.random.choice(indices, 2, replace=False))  #False代表抽出的不放回去
                if rand_x - rand_z < 100:
                    break
            else:
                rand_z = np.random.choice(indices)
                rand_x = rand_z

            return rand_z, rand_x
# 通过该函数筛选符合条件的有效索引val_indices
    def _filter(self, img0, anno, vis_ratios=None):
        size = np.array(img0.shape[1::-1])[np.newaxis, :]
        areas = anno[:, 2] * anno[:, 3]

        # acceptance conditions
        c1 = areas >= 20
        c2 = np.all(anno[:, 2:] >= 20, axis=1)
        c3 = np.all(anno[:, 2:] <= 500, axis=1)
        c4 = np.all((anno[:, 2:] / size) >= 0.01, axis=1)
        c5 = np.all((anno[:, 2:] / size) <= 0.5, axis=1)
        c6 = (anno[:, 2] / np.maximum(1, anno[:, 3])) >= 0.25
        c7 = (anno[:, 2] / np.maximum(1, anno[:, 3])) <= 4
        if vis_ratios is not None:
            c8 = (vis_ratios > max(1, vis_ratios.max() * 0.3))
        else:
            c8 = np.ones_like(c1)

        mask = np.logical_and.reduce(
            (c1, c2, c3, c4, c5, c6, c7, c8))
        val_indices = np.where(mask)[0]

        return val_indices

if __name__ == "__main__":
    root_dir = 'D:/Dataset/GOT-10k'
    seq_dataset = GOT10k(root_dir, subset='train')
    transforms = SiamFCTransforms(
        exemplar_sz=cfg.exemplar_sz,  # 127
        instance_sz=cfg.instance_sz,  # 255
        context=cfg.context)  # 0.5
    train_dataset = GOT10kDataset(seq_dataset, transforms)
    train_dataset.__getitem__(1)

