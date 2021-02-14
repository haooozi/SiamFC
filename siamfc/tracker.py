from __future__ import absolute_import, division, print_function

import torch
import numpy as np
import time
import cv2
from got10k.trackers import Tracker
from . import utils
from .corr import _corr
from .model import AlexNet
from siamfc.config import cfg

__all__ = ['TrackerSiamFC']

class TrackerSiamFC(Tracker):

    def __init__(self, net_path=None, ):
        super(TrackerSiamFC, self).__init__('SiamFC', True)  #这个SiamFC为tracker的name
        self.cuda = torch.cuda.is_available()                        #是否支持GPU
        self.device = torch.device('cuda:0' if self.cuda else 'cpu') #cuda设备号为0
        #加载训练好的模型来测试
        self.model = AlexNet(init_weight=True)
        self.corr = _corr()
        if net_path is not None:
            checkpoint = torch.load(net_path,map_location=lambda storage, loc: storage)
            self.model.load_state_dict(checkpoint['model'])
        self.model = self.model.to(self.device)
        self.corr = self.corr.to(self.device)

    #传入第一帧的gt和图片，初始化一些参数，计算一些之后搜索区域的中心等等
    def init(self, img, box):
        #设置成评估模式，测试模型时一开始要加这个，属于Pytorch，训练模型前，要加self.net.train()
        self.model.eval()
        #将原始的目标位置表示[l,t,w,h]->[center_y,center_x,h,w]
        yxhw = ltwh_to_yxhw(ltwh=box)
        self.center, self.target_sz = yxhw[:2], yxhw[2:]
        #创建汉宁窗口update使用
        self.response_upsz = cfg.response_up * cfg.response_sz  # 16*17=272
        self.hann_window = creat_hanning_window(size=self.response_upsz)
        #三种尺度1.0375**(-1,0,1) 三种尺度
        self.scale_factors = three_scales()
        # patch边长,两种边长：目标模板图像z_sz和搜索图像x_zs
        context = cfg.context * np.sum(self.target_sz)  # 上下文信息(h+w)/2
        self.z_sz = np.sqrt(np.prod(self.target_sz + context))  # (h+(h+w)/2)*(w+(h+2)/2))^0.5
        self.x_sz = self.z_sz * cfg.instance_sz / cfg.exemplar_sz  # (h+(h+w)/2)*(w+(h+2)/2))^0.5*255/127
        #图像的RGB均值，返回(aveR,aveG,aveB)
        self.avg_color = np.mean(img, axis=(0, 1))
        #裁剪一块以目标为中心，边长为z_sz大小的patch,然后将其resize成exemplar_sz的大小
        z = z_to127(img, self.center, self.z_sz, cfg.exemplar_sz, self.avg_color)
        z = torch.from_numpy(z).to(    #torch.size=([1,3,127,127])
            self.device).permute(2, 0, 1).unsqueeze(0).float()
        self.kernel = self.model(z)    #torch.size=([1,128,6,6])

    @torch.no_grad() #全局过程中不求导，禁止使用上下文管理器
    #传入后续帧，然后根据SiamFC跟踪流程返回目标的box坐标
    def update(self, img):
        self.model.eval()
        """----------------正向推导得到response map最大值位置---------------------"""
        #三种patch边长 patch*3scales
        x = x_to3s255(img,self.center,self.x_sz,self.scale_factors,cfg.instance_sz,self.avg_color)
        #numpy转为float的torch型张量
        x = torch.from_numpy(x).to(self.device).permute(0, 3, 1, 2).float()
        #[3,255,22,22]
        x = self.model(x)
        #得到三种尺度下的response map
        responses = self.corr(self.kernel, x) * cfg.out_reduce #[3，1，17，17]
        responses = responses.squeeze(1).cpu().numpy()   #压缩为[3，17，17]并转为numpy作后续计算处理
        #将17x17大小的response map->[3,272,272]
        responses = map_to272(responses,out_size=self.response_upsz)
        #对尺度变化做出相应惩罚
        responses[:cfg.scale_num // 2] *= cfg.scale_penalty      #response[0]*（0.9745惩罚项）
        responses[cfg.scale_num // 2 + 1:] *= cfg.scale_penalty  #response[2]*（0.9745惩罚项）
        #找到最大值属于哪个response map，并把该response map赋给response
        scale_id = np.argmax(np.amax(responses, axis=(1, 2))) #里面求得三个map的最大值 再对三个值求最大值 得到索引
        response = responses[scale_id]   #[272，272]
        #一系列数据处理，重点在汉宁窗惩罚
        response = map_process(response,self.hann_window)
        loc = np.unravel_index(response.argmax(),response.shape)   #unravel_index该函数可返回索引response.argmax()的元素的坐标，逐行拉伸，返回第几行第几个
        """---------------由response map最大值位置反推目标在原图像的位置------------"""
        disp_in_response = np.array(loc) - (self.response_upsz - 1) / 2  #峰值点相对于response中心的位移
        disp = disp_in_response / 16
        disp = disp * 8
        disp = disp * self.x_sz * self.scale_factors[scale_id] / cfg.instance_sz
        self.center += disp
        """---------------参数更新------------"""
        scale =  (1 - cfg.scale_lr) * 1.0 + \
            cfg.scale_lr * self.scale_factors[scale_id]   #scale=0.41*1+0.59*某种尺度
        self.target_sz *= scale   #得到目标的长宽
        self.z_sz *= scale        #h+(h+w)/2)*(w+(h+2)/2))^0.5*scale
        self.x_sz *= scale        #h+(h+w)/2)*(w+(h+2)/2))^0.5*255(instance_sz)/127*scale
        #[y,x,h,w]->[l,t,w,h]
        box = np.array([
            self.center[1] + 1 - (self.target_sz[1] - 1) / 2,
            self.center[0] + 1 - (self.target_sz[0] - 1) / 2,
            self.target_sz[1], self.target_sz[0]])    #倒换成[l，t，w，h]
        return box

    def track(self, img_files, box, visualize=False): #这里的box就是第一个groundtruth
        frame_num = len(img_files)   #frame_num=120
        boxes = np.zeros((frame_num, 4))  #120x4大小的零矩阵 120行4列
        boxes[0] = box             #boxes[0]是boxes的第一行
        times = np.zeros(frame_num)   #1x120大小的零矩阵   1行120列

        for f, img_file in enumerate(img_files):  #f=0,1,2,3.....img_file为1.jpg，2.jp.....
            img = utils.read_image(img_file)    #读图片，返回RGB格式[w,h,c]，（0~255）
            begin = time.time()  #记录当前时间
            if f == 0:
                self.init(img, box)
            else:
                boxes[f, :] = self.update(img)
            times[f] = time.time() - begin

            if visualize:
                utils.show_image(img, boxes[f, :])
            # if f==1:
            #     img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            #     boxes[f,:] = np.array(boxes[f,:],dtype = np.int32)
            #     aa = boxes[f,:].astype(np.int32)
            #     cv2.rectangle(img, (aa[0], aa[1]), (aa[0] + aa[2], aa[1] + aa[3]),
            #                   (255, 0, 0), 3)
            #     cv2.imshow('#1',img)
            # if f==10:
            #     img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            #     boxes[f, :] = np.array(boxes[f, :], dtype=np.int32)
            #     aa = boxes[f, :].astype(np.int32)
            #     cv2.rectangle(img, (aa[0], aa[1]), (aa[0] + aa[2], aa[1] + aa[3]),
            #                   (255, 0, 0), 3)
            #     cv2.imshow('#10',img)
            # if f==40:
            #     img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            #     boxes[f, :] = np.array(boxes[f, :], dtype=np.int32)
            #     aa = boxes[f, :].astype(np.int32)
            #     cv2.rectangle(img, (aa[0], aa[1]), (aa[0] + aa[2], aa[1] + aa[3]),
            #                   (255, 0, 0), 3)
            #     cv2.imshow('#40',img)
        return boxes, times

def ltwh_to_yxhw(ltwh):
    yxhw = np.array([
        ltwh[1] - 1 + (ltwh[3] - 1) / 2,
        ltwh[0] - 1 + (ltwh[2] - 1) / 2,
        ltwh[3], ltwh[2]], dtype=np.float32)
    return yxhw

def yxhw_to_ltwh(yxhw):
    ltwh = np.array([
        yxhw[1] + 1 - (yxhw[1] - 1) / 2,
        yxhw[0] + 1 - (yxhw[0] - 1) / 2,
        yxhw[1], yxhw[0]])
    return ltwh

def creat_hanning_window(size):
    hann_window = np.outer(  #a，b都是行向量，则
        np.hanning(size),    #np.outer(a,b)=a^(T)*b 组成一个矩阵
        np.hanning(size))
    hann_window /= hann_window.sum()
    return hann_window

def z_to127(img,center,patch_size,out_size,border_value):
    z = utils.crop_and_resize(
        img, center, patch_size,
        out_size=out_size,
        border_value=border_value)
    return z

def x_to3s255(img,center,patch_size,three_scales,out_size,border_value):
    x = [utils.crop_and_resize(
        img, center, patch_size * scale,
        out_size=out_size,
        border_value=border_value) for scale in three_scales]
    x = np.stack(x, axis=0)  # [3,255,255,3]第一个三代表三种尺度
    return x

def map_process(response,hanning_window):
    # 一系列数据处理，重点在汉宁窗惩罚
    response -= response.min()
    response /= response.sum() + 1e-16
    # import matplotlib.pyplot as plt
    # plt.figure(0)
    # plt.imshow(response)
    response = (1 - cfg.window_influence) * response + \
               cfg.window_influence * hanning_window  # window_influence=0.176
    # plt.figure(1)
    # plt.imshow(response)
    # plt.show()
    # assert 1==2
    return response

def map_to272(responses,out_size):
    responses = np.stack([cv2.resize(
        u, (out_size, out_size),
        interpolation=cv2.INTER_CUBIC)
        for u in responses])
    return responses

def three_scales():
    scale_factors = cfg.scale_step ** np.linspace(  # 1.0375^(-1,0,1)
        -(cfg.scale_num // 2),
        cfg.scale_num // 2, cfg.scale_num)
    return scale_factors