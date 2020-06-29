#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2020/6/25 12:12
# @Author  : Xin Zhang
# @FileName: model.py
# @Software: PyCharm
import os
import torch
import torch.nn as nn
from torch.autograd import Variable
from nets.FCN import vgg19_Net, FCN


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0)
    elif classname.find('BatchNorm2d') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


class Model:
    def name(self):
        return 'Model'

    def __init__(self, opt):
        self.opt = opt
        self.gpu_ids = opt.gpu_ids
        self.isTrain = opt.isTrain
        self.Tensor = torch.cuda.FloatTensor if self.gpu_ids else torch.Tensor
        self.save_dir = os.path.join(opt.checkpoints_dir)
        self.inputImg = self.Tensor(opt.batchsize, 1, 512, 512)
        self.inputCorrdX = self.Tensor(opt.batchsize, 11)
        self.inputCorrdY = self.Tensor(opt.batchsize, 11)
        self.preCoord = []
        self.net = vgg19_Net().cuda()
        self.loss_fn = nn.L1Loss(size_average=True)
        if self.isTrain:
            self.optimizer = torch.optim.Adam(self.net.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))

    def set_input(self, input):
        Img = input['dcm']
        coordX = input['coordX']
        coordY = input['coordY']
        self.inputImg.resize_(Img.size()).copy_(Img)
        self.inputCorrdX.resize_(coordX.size()).copy_(coordX)
        self.inputCorrdY.resize_(coordY.size()).copy_(coordY)

    def forward(self):
        self.img = Variable(self.inputImg)
        self.preCoord = self.net(self.img)
        self.corrdX = Variable(self.inputCorrdX)
        self.corrdY = Variable(self.inputCorrdY)
        self.coord = Variable(torch.cat([self.inputCorrdX, self.inputCorrdY], dim=1))

    def backward(self):
        self.loss = self.loss_fn(self.preCoord, self.coord)
        self.loss.backward()

    def optimize_parameters(self):
        self.forward()
        self.optimizer.zero_grad()
        self.backward()
        self.optimizer.step()

    def getCorrd(self):
        precoord_list = self.preCoord.cpu().detach().numpy()
        preCoord = []
        for i in range(11):
            preCoord.append([precoord_list[0][i], precoord_list[0][i + 11]])
        return preCoord

    # used in test time, no backprop
    def test(self):
        pass

    def get_image_paths(self):
        pass

    def get_current_visuals(self):
        return self.input

    def get_current_errors(self):
        return {}

    def save(self, label):
        pass

    # helper saving function that can be used by subclasses
    def save_network(self, network, epoch_label, gpu_ids):
        save_filename = '%s.pth' % epoch_label
        save_path = os.path.join(self.save_dir, save_filename)
        torch.save(network.cpu().state_dict(), save_path)
        if len(gpu_ids) and torch.cuda.is_available():
            network.cuda(device=0)

    # helper loading function that can be used by subclasses
    def load_network(self, network, network_label, epoch_label):
        save_filename = '%s_net_%s.pth' % (epoch_label, network_label)
        save_path = os.path.join(self.save_dir, save_filename)
        network.load_state_dict(torch.load(save_path))

    def update_learning_rate(self):
        pass
