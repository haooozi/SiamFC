from __future__ import absolute_import

import os
from got10k.experiments import *
import torch
from siamfc import TrackerSiamFC
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
if __name__ == '__main__':

    """使用GOT-10k 的val进行测试，我只用了val的前50个"""
    # net_path = 'pretrained/siamfc_alexnet_e50.pth'
    # tracker = TrackerSiamFC(net_path=net_path)

    # experiment = ExperimentGOT10k('F:\data\GOT-10k', subset='val', result_dir='results', report_dir='reports')
    # experiment.run(tracker,visualize=False)
    # experiment.report([tracker.name])

    net_path = 'pretrained\siamfc_alexnet_e50.pth'
    tracker = TrackerSiamFC(net_path=net_path)
    # experiment = ExperimentOTB('F:\OTB100',version='tb100',result_dir='results', report_dir='reports')
    # experiment.run(tracker,visualize=False)
    # experiment.report([tracker.name])

    experiment = ExperimentVOT('E:\VOT2016',version=2016,result_dir='results',report_dir='reports')
    # experiment.run(tracker, visualize=False)
    experiment.report([tracker.name])

