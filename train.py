#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2020/6/24 15:21
# @Author  : Xin Zhang
# @FileName: train.py
# @Software: PyCharm
import torch
import argparse
import torch.utils.data
from model.model import Model
from dataset.lumbar_dataset import Lumbar_dataset
from torch.utils.tensorboard import SummaryWriter


def train(opt, dataloader, model):
    writer = SummaryWriter('./logs')
    iter = 0
    for epoch in range(opt.epoch):
        loss = 0
        count = 0
        rights = 0
        right = False
        for i, data in enumerate(dataloader):
            count += 1
            model.set_input(data)
            model.optimize_parameters()
            loss += model.loss
            preCorrd = model.getCorrd()
            for j in range(11):
                if abs(int(data['coordX'].numpy()[0][j]) - int(preCorrd[j][0])) < 10 \
                        & abs(int(data['coordY'].numpy()[0][j]) - int(preCorrd[j][1])) < 10:
                    right = True
                else:
                    right = False
            if i % 10 == 0:
                print("epoch:{}, iter:{}, loss:{}".format(epoch, i, model.loss))
            if right:
                rights += 1
        acc = rights/count
        mean_loss = loss / count
        writer.add_scalar('epoch', mean_loss, epoch)
        model.save_network(model.net, epoch, opt.gpu_ids)
        print("epoch:{}, mean_loss:{}, acc:{}".format(epoch, mean_loss, acc))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv_path', type=str, default='./train150.csv')
    parser.add_argument('--checkpoints_dir', type=str, default='./checkpoint')
    parser.add_argument('--isTrain', type=bool, default=True)
    parser.add_argument('--batchsize', type=int, default=1, help='input batch size')
    parser.add_argument('--epoch', type=int, default=250, help='epoch')
    parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
    parser.add_argument('--beta1', type=float, default=0.5, help='momentum term of adam')
    parser.add_argument('--lr', type=float, default=0.00005, help='initial learning rate for adam')
    opt = parser.parse_args()
    dataset = Lumbar_dataset(opt)
    dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=opt.batchsize, shuffle=True)
    model = Model(opt)
    train(opt, dataloader, model)
