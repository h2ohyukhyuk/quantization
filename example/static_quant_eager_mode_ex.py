'''
(BETA) STATIC QUANTIZATION WITH EAGER MODE IN PYTORCH
https://pytorch.org/tutorials/advanced/static_quantization_tutorial.html
'''
import os
import sys
import time
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import torchvision
from torchvision import datasets
import torchvision.transforms as transforms

from example.utils import _make_divisible, ConvBNReLU, InvertedResidual, AverageMeter, accuracy, print_size_of_model

# Set up warnings
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
torch.manual_seed(191009)

from torch.ao.quantization import QuantStub, DeQuantStub

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
        block = InvertedResidual
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
        self.quant = QuantStub()
        self.dequant = DeQuantStub()
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
        x = self.quant(x)
        x = self.features(x)
        x = x.mean([2, 3])
        x = self.classifier(x)
        x = self.dequant(x)
        return x

    # Fuse Conv+BN and Conv+BN+Relu modules prior to quantization
    # This operation does not change the numerics
    def fuse_model(self, is_qat=False):
        fuse_modules = torch.ao.quantization.fuse_modules_qat if is_qat else torch.ao.quantization.fuse_modules
        for m in self.modules():
            if type(m) == ConvBNReLU:
                fuse_modules(m, ['0', '1', '2'], inplace=True)
            if type(m) == InvertedResidual:
                for idx in range(len(m.conv)):
                    if type(m.conv[idx]) == nn.Conv2d:
                        fuse_modules(m.conv, [str(idx), str(idx + 1)], inplace=True)


def evaluate(model, criterion, data_loader, neval_batches, dev):
    model.eval()
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    cnt = 0
    with torch.no_grad():
        for i, (image, target) in enumerate(data_loader):
            image, target = image.to(dev), target.to(dev)
            output = model(image)
            loss = criterion(output, target)
            cnt += 1
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            print(f'{i}.', end = '')
            if i % 41 == 0:
                print('\n')
            top1.update(acc1[0], image.size(0))
            top5.update(acc5[0], image.size(0))
            if cnt >= neval_batches:
                 return top1, top5

    print('\n')
    return top1, top5

def load_model(model_file):
    model = MobileNetV2()
    state_dict = torch.load(model_file)
    model.load_state_dict(state_dict)
    model.to('cpu')
    return model

def prepare_data_loaders(data_path):
    # ILSVRC2012_devkit_t12.tar.gz
    train_batch_size = 30
    eval_batch_size = 50
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    dataset = torchvision.datasets.ImageNet(
        data_path, split="train", transform=transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))
    dataset_test = torchvision.datasets.ImageNet(
        data_path, split="val", transform=transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ]))

    train_sampler = torch.utils.data.RandomSampler(dataset)
    test_sampler = torch.utils.data.SequentialSampler(dataset_test)

    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=train_batch_size,
        sampler=train_sampler)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=eval_batch_size,
        sampler=test_sampler)

    return data_loader, data_loader_test

def prepare_data_loader_train():
    # ILSVRC2012_devkit_t12.tar.gz
    train_batch_size = 30
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    data_path_train = 'D:/ImageNet1K/train'
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

def prepare_data_loader_test():
    # ILSVRC2012_devkit_t12.tar.gz
    eval_batch_size = 50
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    data_path_test = 'D:/ImageNet1K/val'
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

def train_one_epoch(model, criterion, optimizer, data_loader, device, ntrain_batches):
    model.train()
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    avgloss = AverageMeter('Loss', '1.5f')

    cnt = 0
    for image, target in data_loader:
        start_time = time.time()
        print('.', end = '')
        cnt += 1
        image, target = image.to(device), target.to(device)
        output = model(image)
        loss = criterion(output, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        top1.update(acc1[0], image.size(0))
        top5.update(acc5[0], image.size(0))
        avgloss.update(loss, image.size(0))
        if cnt >= ntrain_batches:
            print('Loss', avgloss.avg)

            print('Training: * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
                  .format(top1=top1, top5=top5))
            return

    print('Full imagenet train set:  * Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f}'
          .format(top1=top1, top5=top5))
    return


def baseline():

    saved_model_dir = '../data/'
    float_model_file = 'mobilenet_v2-b0353104.pth'
    scripted_float_model_file = 'mobilenet_float_scripted.pth'

    train_batch_size = 30
    eval_batch_size = 50

    data_loader_test = prepare_data_loader_test()
    criterion = nn.CrossEntropyLoss()
    float_model = load_model(saved_model_dir + float_model_file).to('cpu')

    # Next, we'll "fuse modules"; this can both make the model faster by saving on memory access
    # while also improving numerical accuracy. While this can be used with any model, this is
    # especially common with quantized models.

    print('\n Inverted Residual Block: Before fusion \n\n', float_model.features[1].conv)
    float_model.eval()

    # Fuses modules
    float_model.fuse_model()

    # Note fusion of Conv+BN+Relu and Conv+Relu
    print('\n Inverted Residual Block: After fusion\n\n', float_model.features[1].conv)

    num_eval_batches = 1000

    print("Size of baseline model")
    print_size_of_model(float_model)

    dev = device = 'cuda' if torch.cuda.is_available() else 'cpu'
    float_model.to(dev)
    top1, top5 = evaluate(float_model, criterion, data_loader_test, neval_batches=num_eval_batches, dev=dev)
    print('Evaluation accuracy on %d images, %2.2f' % (num_eval_batches * eval_batch_size, top1.avg))
    # Evaluation accuracy on 50000 images, 71.86


    torch.jit.save(torch.jit.script(float_model), saved_model_dir + scripted_float_model_file)


def post_train_static_quant():
    saved_model_dir = '../data/'
    float_model_file = 'mobilenet_v2-b0353104.pth'
    scripted_quantized_model_file = 'mobilenet_quantization_scripted_quantized.pth'

    num_calibration_batches = 32

    data_loader_train = prepare_data_loader_train()
    data_loader_test = prepare_data_loader_test()
    criterion = nn.CrossEntropyLoss()

    myModel = load_model(saved_model_dir + float_model_file).to('cpu')
    myModel.eval()

    # Fuse Conv, bn and relu
    myModel.fuse_model()

    # Specify quantization configuration
    # Start with simple min/max range estimation and per-tensor quantization of weights
    myModel.qconfig = torch.ao.quantization.default_qconfig
    print('myModel.qconfig')
    print(myModel.qconfig)
    torch.ao.quantization.prepare(myModel, inplace=True)

    # Calibrate first
    print('Post Training Quantization Prepare: Inserting Observers')
    print('\n Inverted Residual Block:After observer insertion \n\n', myModel.features[1].conv)

    # Calibrate with the training set
    evaluate(myModel, criterion, data_loader_train, neval_batches=num_calibration_batches, dev='cpu')
    print('Post Training Quantization: Calibration done')

    # Convert to quantized model
    torch.ao.quantization.convert(myModel, inplace=True)
    print('Post Training Quantization: Convert done')
    print('\n Inverted Residual Block: After fusion and quantization, note fused modules: \n\n',
          myModel.features[1].conv)

    print("Size of model after quantization")
    print_size_of_model(myModel)

    num_eval_batches = 1000
    eval_batch_size = 50

    top1, top5 = evaluate(myModel, criterion, data_loader_test, neval_batches=num_eval_batches, dev='cpu')
    print('Evaluation accuracy on %d images, %2.2f' % (num_eval_batches * eval_batch_size, top1.avg))

    print('Per Channel Quantization')
    per_channel_quantized_model = load_model(saved_model_dir + float_model_file)
    per_channel_quantized_model.eval()
    per_channel_quantized_model.fuse_model()
    # The old 'fbgemm' is still available but 'x86' is the recommended default.
    per_channel_quantized_model.qconfig = torch.ao.quantization.get_default_qconfig('x86')
    print(per_channel_quantized_model.qconfig)

    torch.ao.quantization.prepare(per_channel_quantized_model, inplace=True)
    evaluate(per_channel_quantized_model, criterion, data_loader_train, num_calibration_batches, dev='cpu')
    torch.ao.quantization.convert(per_channel_quantized_model, inplace=True)
    top1, top5 = evaluate(per_channel_quantized_model, criterion, data_loader_test, neval_batches=num_eval_batches, dev='cpu')
    print('Evaluation accuracy on %d images, %2.2f' % (num_eval_batches * eval_batch_size, top1.avg))
    torch.jit.save(torch.jit.script(per_channel_quantized_model), saved_model_dir + scripted_quantized_model_file)

def quant_aware_train():
    saved_model_dir = '../data/'
    float_model_file = 'mobilenet_v2-b0353104.pth'
    qat_model = load_model(saved_model_dir + float_model_file)
    qat_model.fuse_model(is_qat=True)

    optimizer = torch.optim.SGD(qat_model.parameters(), lr=0.0001)
    # The old 'fbgemm' is still available but 'x86' is the recommended default.
    qat_model.qconfig = torch.ao.quantization.get_default_qat_qconfig('x86')

    torch.ao.quantization.prepare_qat(qat_model, inplace=True)
    print('Inverted Residual Block: After preparation for QAT, note fake-quantization modules \n',
          qat_model.features[1].conv)

    device = 'cpu'
    qat_model = qat_model.to(device)

    data_loader = prepare_data_loader_train()
    data_loader_test = prepare_data_loader_test()

    num_train_batches = 20
    num_eval_batches = 1000
    eval_batch_size = 50

    criterion = nn.CrossEntropyLoss()
    # QAT takes time and one needs to train over a few epochs.
    # Train and check accuracy after each epoch
    for nepoch in range(8):
        train_one_epoch(qat_model, criterion, optimizer, data_loader, device, num_train_batches)
        if nepoch > 3:
            # Freeze quantizer parameters
            qat_model.apply(torch.ao.quantization.disable_observer)
        if nepoch > 2:
            # Freeze batch norm mean and variance estimates
            qat_model.apply(torch.nn.intrinsic.qat.freeze_bn_stats)

        # Check the accuracy after each epoch
        quantized_model = torch.ao.quantization.convert(qat_model.eval(), inplace=False)
        quantized_model.eval()
        top1, top5 = evaluate(quantized_model, criterion, data_loader_test, neval_batches=num_eval_batches, dev=device)
        print('Epoch %d :Evaluation accuracy on %d images, %2.2f' % (
        nepoch, num_eval_batches * eval_batch_size, top1.avg))

    torch.jit.save(torch.jit.script(quantized_model), saved_model_dir + 'mobilenet_qat_scripted.pth')

def check_scripted_quantized_model():
    path_model = '../data/mobilenet_quantization_scripted_quantized.pth'
    model_scripted = torch.jit.load(path_model)
    model_scripted = model_scripted.to('cpu')
    model_scripted = model_scripted.eval()

    data_loader = prepare_data_loader_test()
    dev = 'cpu'
    cnt = 0

    with torch.no_grad():
        for i, (image, target) in enumerate(data_loader):
            image, target = image.to(dev), target.to(dev)
            output = model_scripted(image)
            _, pred = output.topk(k=1, dim=1, largest=True, sorted=True)

            correct = pred.eq(target.view(-1, 1))
            print('correct: ', torch.sum(correct).item(), ' / ', target.size(0))
            cnt += 1
            if cnt >= 10:
                break

if __name__ == '__main__':
    #baseline()
    #post_train_static_quant()
    #check_scripted_quantized_model()
    quant_aware_train()