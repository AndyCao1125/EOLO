from .resnet import resnet18, resnet50, resnet101, event_resnet18
from .darknet import darknet53
from .cspdarknet_tiny import cspdarknet_tiny
from .cspdarknet53 import cspdarknet53
from .yolox_backbone import yolox_cspdarknet_s, yolox_cspdarknet_m, yolox_cspdarknet_l, \
                            yolox_cspdarknet_x, yolox_cspdarknet_tiny, yolox_cspdarknet_nano
from .shufflenetv2 import shufflenetv2
from .vit import vit_base_patch16_224

from .spk_resnet import spiking_resnet18
import torch
import torch.nn as nn
def build_backbone(model_name='r18', pretrained=False, freeze=None, img_size=224, in_channel=3, time_step=4):
    if model_name == 'r18':
        print('Backbone: ResNet-18 ...')
        model = resnet18(pretrained=True)
        conv1_new = nn.Conv2d(2, 64, kernel_size=7, stride=2, padding=3, bias=False)
        model.conv1 = conv1_new
        feature_channels = [128, 256, 512]
        strides = [8, 16, 32]
    elif model_name == 'r50':
        print('Backbone: ResNet-50 ...')
        model = resnet50(pretrained=pretrained)
        feature_channels = [512, 1024, 2048]
        strides = [8, 16, 32]
    elif model_name == 'r101':
        print('Backbone: ResNet-101 ...')
        model = resnet101(pretrained=pretrained)
        feature_channels = [512, 1024, 2048]
        strides = [8, 16, 32]
    elif model_name == 'd53':
        print('Backbone: DarkNet-53 ...')
        model = darknet53(pretrained=pretrained, in_channel=in_channel)
        feature_channels = [256, 512, 1024]
        strides = [8, 16, 32]
    elif model_name == 'cspd53':
        print('Backbone: CSPDarkNet-53 ...')
        model = cspdarknet53(pretrained=pretrained)
        feature_channels = [256, 512, 1024]
        strides = [8, 16, 32]
    elif model_name == 'cspd_tiny':
        print('[Backbone: CSPDarkNet-Tiny] ...')
        model = cspdarknet_tiny(pretrained=pretrained, in_channel=in_channel)
        feature_channels = [128, 256, 512]
        strides = [8, 16, 32]
    elif model_name == 'sfnet_v2':
        print('Backbone: ShuffleNet-V2 ...')
        model = shufflenetv2(pretrained=pretrained)
        feature_channels = [116, 232, 464]
        strides = [8, 16, 32]
    elif model_name == 'vit_base_16':
        print('Backbone: ViT_Base_16 ...')
        model = vit_base_patch16_224(img_size=img_size, pretrained=pretrained)
        feature_channels = [None, None, 768]
        strides = [None, None, 16]

    elif model_name == 'spike_r18_jelly':

        from spikingjelly.activation_based import surrogate, neuron, functional, layer
        print('[Backbone: Spk-ResNet-18-jelly] ...')
        model = spiking_resnet18(pretrained=True, spiking_neuron=neuron.IFNode, surrogate_function=surrogate.ATan(), detach_reset=True, T=time_step)
        functional.set_step_mode(model, step_mode='m')
        # functional.set_backend(model, 'cupy', neuron.IFNode)
        conv1_new = layer.Conv2d(2, 64, kernel_size=7, stride=2, padding=3, bias=False)
        functional.set_step_mode(conv1_new, step_mode='m')
        model.conv1 = conv1_new
        feature_channels = [128, 256, 512]
        strides = None

    # YOLOX backbone
    elif model_name == 'csp_s':
        print('Backbone: YOLOX-CSPDarkNet-S ...')
        model = yolox_cspdarknet_s(pretrained=pretrained, freeze=freeze)
        feature_channels = [128, 256, 512]
        strides = [8, 16, 32]
    elif model_name == 'csp_m':
        print('Backbone: YOLOX-CSPDarkNet-M ...')
        model = yolox_cspdarknet_m(pretrained=pretrained, freeze=freeze)
        feature_channels = [192, 384, 768]
        strides = [8, 16, 32]
    elif model_name == 'csp_l':
        print('Backbone: YOLOX-CSPDarkNet-L ...')
        model = yolox_cspdarknet_l(pretrained=pretrained, freeze=freeze)
        feature_channels = [256, 512, 1024]
        strides = [8, 16, 32]
    elif model_name == 'csp_x':
        print('Backbone: YOLOX-CSPDarkNet-X ...')
        model = yolox_cspdarknet_x(pretrained=pretrained, freeze=freeze)
        feature_channels = [320, 640, 1280]
        strides = [8, 16, 32]
    elif model_name == 'csp_t':
        print('Backbone: YOLOX-CSPDarkNet-Tiny ...')
        model = yolox_cspdarknet_tiny(pretrained=pretrained, freeze=freeze)
        feature_channels = [96, 192, 384]
        strides = [8, 16, 32]
    elif model_name == 'csp_n':
        print('Backbone: YOLOX-CSPDarkNet-Nano ...')
        model = yolox_cspdarknet_nano(pretrained=pretrained, freeze=freeze)
        feature_channels = [64, 128, 256]
        strides = [8, 16, 32]
    
    return model, feature_channels, strides
