import torch
import torch.nn as nn
from nets.unet import Unet
import torch.nn.functional as F
import torchvision.models as models


class vgg19_Net(nn.Module):
    def __init__(self, in_img_rgb=1, in_img_size=256, out_class=22, in_fc_size=147968):
        super(vgg19_Net, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=in_img_rgb, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1)
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1)
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1)
        )

        self.fc5 = nn.Sequential(
            nn.Linear(in_features=in_fc_size, out_features=512, bias=True),
            nn.ReLU()
        )

        self.fc6 = nn.Sequential(
            nn.Linear(in_features=512, out_features=22, bias=True),
            nn.ReLU()
        )

        self.conv_list = [self.conv1, self.conv2, self.conv3, self.conv4]

        self.fc_list = [self.fc5, self.fc6]

    def forward(self, x):

        for conv in self.conv_list:
            x = conv(x)

        fc = x.view(x.size(0), -1)

        # 查看全连接层的参数：in_fc_size  的值
        # print("vgg19_model_fc:",fc.size(1))

        for fc_item in self.fc_list:
            fc = fc_item(fc)

        return fc


class FCN(nn.Module):
    def __init__(self, input_channel=1, output_features=22):
        super(FCN, self).__init__()
        self.feature = Unet(1, 3)
        self.layer1 = nn.Sequential(nn.Linear(256 * 256 * 3, 512), nn.ReLU(True))
        self.layer2 = nn.Sequential(nn.Linear(512, output_features), nn.ReLU(True))

    def forward(self, x):
        x = self.feature(x)
        x = x.view(x.size(0), -1)
        x = self.layer1(x)
        x = self.layer2(x)
        return x
