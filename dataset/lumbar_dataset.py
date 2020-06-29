#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2020/6/24 15:22
# @Author  : Xin Zhang
# @FileName: lumbar_dataset.py
# @Software: PyCharm
import torch.utils.data as data
import torch
import argparse
import pandas as pd
import demjson
import cv2
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from utils.readDCM import dicom2array

ide_dict = {'T12-L1': 0, 'L1': 1, 'L1-L2': 2,
            'L2': 3, 'L2-L3': 4, 'L3': 5,
            'L3-L4': 6, 'L4': 7, 'L4-L5': 8,
            'L5': 9, 'L5-S1': 10}


class Lumbar_dataset(data.Dataset):
    def __init__(self, opt):
        self.opt = opt
        self.df = pd.read_csv(opt.csv_path, names=['path', 'json'])
        self.dcm_paths = self.df['path'].tolist()
        self.jsons = self.df['json'].tolist()
        transform_list1 = [transforms.ToTensor(),
                           transforms.Normalize(0.5, 0.5)]
        transform_list2 = [transforms.ToPILImage(),
                           transforms.Resize((256, 256)),
                           transforms.ToTensor()]
        self.transform1 = transforms.Compose(transform_list1)
        self.transform2 = transforms.Compose(transform_list2)

    def __getitem__(self, index):
        dcm = dicom2array(self.dcm_paths[index])
        # cv2.imshow('img', dcm)
        # cv2.waitKey(0)
        dcm = self.transform1(dcm)
        jsonstr = self.jsons[index].replace('\'', '\"')
        json = demjson.decode(jsonstr)[0]
        coordX = []
        coordY = []
        coord = []
        for point in json['data']['point']:
            coord.append([point['coord'], ide_dict[point['tag']['identification']]])
        coord.sort(key=lambda x: x[1])
        _, h, w = dcm.size()
        if h > w:
            dcm = dcm[:, h-w:, :]
            coordX = [int(x[0][0]*256/w) for x in coord[:]]
            coordY = [int((y[0][1]-h+w)*256/w) for y in coord[:]]
        elif w > h:
            dcm = dcm[:, :, w-h:]
            coordX = [int((x[0][0]-w+h)*256/h) for x in coord[:]]
            coordY = [int(y[0][1]*256/h) for y in coord[:]]
        else:
            coordX = [int(x[0][0]*256/w) for x in coord[:]]
            coordY = [int(y[0][1]*256/h) for y in coord[:]]
        dcm = self.transform2(dcm)
        # for iter in range(11):
        #     cv2.circle(dcm.numpy()[0], (coordX[iter], coordY[iter]), 1, (0, 0, 255), 4)
        # cv2.imshow('', dcm.numpy()[0])
        # cv2.waitKey(0)
        coordX = torch.Tensor(coordX)
        coordY = torch.Tensor(coordY)
        return {'dcm': dcm, 'coordX': coordX, 'coordY': coordY}

    def __len__(self):
        return len(self.dcm_paths)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv_path', type=str, default='../train150.csv')
    opt = parser.parse_args()
    dataset = Lumbar_dataset(opt)
    for i in range(len(dataset)):
        print(dataset.__getitem__(i))
