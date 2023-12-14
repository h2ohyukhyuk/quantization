# https://pytorch.org/tutorials/prototype/fx_graph_mode_ptq_static.html

import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
from torch.ao.quantization import get_default_qconfig, get_default_qat_qconfig, QConfigMapping
from torch.ao.quantization.quantize_fx import prepare_fx, convert_fx, fuse_fx

from example.utils import _make_divisible, ConvBNReLU, InvertedResidual2
from example.utils import AverageMeter, print_size_of_model
from example.utils import accuracy, evaluate


import warnings
warnings.filterwarnings(
    action='ignore',
    category=DeprecationWarning,
    module=r'.*'
)
warnings.filterwarnings(
    action='default',
    module=r'torch.ao.quantization'
)

# Specify random seed for repeatable results
_ = torch.manual_seed(191009)

class MobileNetV2(nn.Module):
    def __init__(self, num_classes=1000, width_mult=1.0, inverted_residual_setting=None, round_nearest=8):
        """
        MobileNet V2 main class
        Args:
            num_classes (int): Number of classes
            width_mult (float): Width multiplier - adjusts number of channels in each layer by this amount
            inverted_residual_setting: Network structure
            round_nearest (int): Round the number of channels in each layer to be a multiple of this number
            Set to 1 to turn off rounding
        """
        super(MobileNetV2, self).__init__()
        block = InvertedResidual2
        input_channel = 32
        last_channel = 1280

        if inverted_residual_setting is None:
            inverted_residual_setting = [
                # t, c, n, s
                [1, 16, 1, 1],
                [6, 24, 2, 2],
                [6, 32, 3, 2],
                [6, 64, 4, 2],
                [6, 96, 3, 1],
                [6, 160, 3, 2],
                [6, 320, 1, 1],
            ]

        # only check the first element, assuming user knows t,c,n,s are required
        if len(inverted_residual_setting) == 0 or len(inverted_residual_setting[0]) != 4:
            raise ValueError("inverted_residual_setting should be non-empty "
                             "or a 4-element list, got {}".format(inverted_residual_setting))

        # building first layer
        input_channel = _make_divisible(input_channel * width_mult, round_nearest)
        self.last_channel = _make_divisible(last_channel * max(1.0, width_mult), round_nearest)
        features = [ConvBNReLU(3, input_channel, stride=2)]
        # building inverted residual blocks
        for t, c, n, s in inverted_residual_setting:
            output_channel = _make_divisible(c * width_mult, round_nearest)
            for i in range(n):
                stride = s if i == 0 else 1
                features.append(block(input_channel, output_channel, stride, expand_ratio=t))
                input_channel = output_channel
        # building last several layers
        features.append(ConvBNReLU(input_channel, self.last_channel, kernel_size=1))
        # make it nn.Sequential
        self.features = nn.Sequential(*features)

        # building classifier
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.last_channel, num_classes),
        )

        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.features(x)
        x = x.mean([2, 3])
        x = self.classifier(x)
        return x

def load_model(model_file):
    model = MobileNetV2()
    state_dict = torch.load(model_file)
    model.load_state_dict(state_dict)
    model.to('cpu')
    return model

def prepare_data_loader_train(data_path_train = 'D:/ImageNet1K/train'):
    # ILSVRC2012_devkit_t12.tar.gz
    train_batch_size = 30
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    transform_train = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
    dataset = torchvision.datasets.ImageFolder(root=data_path_train,transform=transform_train)


    train_sampler = torch.utils.data.RandomSampler(dataset)

    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=train_batch_size,
        sampler=train_sampler)

    return data_loader

def prepare_data_loader_test(eval_batch_size = 50, data_path_test = 'D:/ImageNet1K/val'):
    # ILSVRC2012_devkit_t12.tar.gz
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    transform_test = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])
    dataset_test = torchvision.datasets.ImageFolder(data_path_test, transform=transform_test)

    test_sampler = torch.utils.data.SequentialSampler(dataset_test)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=eval_batch_size,
        sampler=test_sampler)

    return data_loader_test

def calibrate(model, data_loader):
    model.eval()
    with torch.no_grad():
        for image, target in data_loader:
            model(image)


def fx_quantization():
    path_float_model = '../data/mobilenet_v2-b0353104.pth'
    float_model = load_model(path_float_model)
    float_model.eval()

    model_to_quant = load_model(path_float_model)
    model_to_quant.eval()

    default_qconfig = get_default_qconfig("x86")
    qconfig_mapping = QConfigMapping().set_global(default_qconfig)

    data_loader_test = prepare_data_loader_test()
    example_inputs = (next(iter(data_loader_test))[0])
    prepared_model = prepare_fx(model_to_quant, qconfig_mapping, example_inputs)
    print('prepared model graph -------------------------------')
    print(prepared_model.graph)

    # run calibration on sample data
    calibrate(prepared_model, data_loader_test)

    # convert model
    quantized_model = convert_fx(prepared_model)
    print('quantized model -------------------------------')
    print(quantized_model)

    quantized_model_scripted = torch.jit.script(quantized_model)
    quantized_model_scripted.save('../data/mobilenet_v2_fx_quant_per_ch_scripted.pth')
    print('fx quantization model saved at ../data/mobilenet_v2_fx_quant_per_ch_scripted.pth')

def fx_quantization_acc():
    quantized_model = torch.jit.load('../data/mobilenet_v2_fx_quant_per_ch_scripted.pth')
    quantized_model.to('cpu')
    quantized_model.eval()

    data_loader_test = prepare_data_loader_test()

    top1, top5 = evaluate(quantized_model, data_loader_test, 100, 'cpu')
    print("[after serialization/deserialization] Evaluation accuracy on test dataset: %2.2f, %2.2f" % (
    top1.avg, top5.avg))
    # .[after serialization/deserialization] Evaluation accuracy on test dataset: 74.98, 91.62

if __name__ == '__main__':
    #fx_quantization()
    fx_quantization_acc()




