import torch
import torch.nn as nn
import os


__all__ = ['darknet53']


class Conv_BN_LeakyReLU(nn.Module):
    def __init__(self, in_channels, out_channels, ksize, padding=0, stride=1, dilation=1):
        super(Conv_BN_LeakyReLU, self).__init__()
        self.in_channels = in_channels
        self.convs = nn.Sequential(
            nn.Conv2d(self.in_channels, out_channels, ksize, padding=padding, stride=stride, dilation=dilation),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.1, inplace=True)
        )

    def forward(self, x):
        return self.convs(x)


class resblock(nn.Module):
    def __init__(self, ch, nblocks=1):
        super().__init__()
        self.module_list = nn.ModuleList()
        for _ in range(nblocks):
            resblock_one = nn.Sequential(
                Conv_BN_LeakyReLU(ch, ch//2, 1),
                Conv_BN_LeakyReLU(ch//2, ch, 3, padding=1)
            )
            self.module_list.append(resblock_one)

    def forward(self, x):
        for module in self.module_list:
            x = module(x) + x
        return x


class DarkNet_53(nn.Module):
    """
    DarkNet-53.
    """
    def __init__(self, in_channel=3, num_classes=1000):
        super(DarkNet_53, self).__init__()
        # stride = 2
        self.layer_1 = nn.Sequential(
            Conv_BN_LeakyReLU(in_channel, 32, 3, padding=1),
            Conv_BN_LeakyReLU(32, 64, 3, padding=1, stride=2),
            resblock(64, nblocks=1)
        )
        # stride = 4
        self.layer_2 = nn.Sequential(
            Conv_BN_LeakyReLU(64, 128, 3, padding=1, stride=2),
            resblock(128, nblocks=2)
        )
        # stride = 8
        self.layer_3 = nn.Sequential(
            Conv_BN_LeakyReLU(128, 256, 3, padding=1, stride=2),
            resblock(256, nblocks=8)
        )
        # stride = 16
        self.layer_4 = nn.Sequential(
            Conv_BN_LeakyReLU(256, 512, 3, padding=1, stride=2),
            resblock(512, nblocks=8)
        )
        # stride = 32
        self.layer_5 = nn.Sequential(
            Conv_BN_LeakyReLU(512, 1024, 3, padding=1, stride=2),
            resblock(1024, nblocks=4)
        )

        # self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # self.fc = nn.Linear(1024, num_classes)

    def forward(self, x, targets=None):
        c1 = self.layer_1(x)
        c2 = self.layer_2(c1)
        c3 = self.layer_3(c2)
        c4 = self.layer_4(c3)
        c5 = self.layer_5(c4)

        return c3, c4, c5


def darknet53(pretrained=False, in_channel=3, **kwargs):
    """Constructs a darknet-53 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = DarkNet_53(in_channel=in_channel)
    if pretrained:
        try:
            print('Loading the pretrained model ...')
            # path_to_dir = os.path.dirname(os.path.abspath(__file__))
            # checkpoint = torch.load(path_to_dir + '/weights/darknet53/darknet53.pth', map_location='cpu')
            checkpoint = torch.load(os.path.join('/home/lsf_storage/homes/jiahang/jiahang/Event-YOLO/PyTorch_YOLO-Family/weights/darknet53/darknet53.pth'), map_location='cpu')
            # checkpoint = os.path.join('/home/lsf_storage/darknet53.pth')
            model.load_state_dict(checkpoint, strict=False)
            print('The pretrained weight of darknet53is found successfully ...')
        except:
            print('The pretrained weight of darknet53 can not be found ...')
            pass
    else:
        print('Training the darknet53 from scratch ...')
    return model
