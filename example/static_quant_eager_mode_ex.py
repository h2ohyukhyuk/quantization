'''
(BETA) STATIC QUANTIZATION WITH EAGER MODE IN PYTORCH
https://pytorch.org/tutorials/advanced/static_quantization_tutorial.html
'''
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import torchvision
import torchvision.transforms as transforms

from example.utils import _make_divisible, ConvBNReLU, InvertedResidual
from example.utils import print_size_of_model, evaluate, train_one_epoch
from torch.utils.mobile_optimizer import optimize_for_mobile
from onnxruntime.quantization import QuantFormat, QuantType, quantize_static

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


def baseline():

    saved_model_dir = '../data/'
    float_model_file = 'mobilenet_v2-b0353104.pth'
    scripted_float_model_file = 'mobilenet_v2_float_scripted.pth'

    dev = 'cpu'
    eval_batch_size = 50
    num_eval_batches = 1000

    data_loader_test = prepare_data_loader_test(eval_batch_size)
    float_model = load_model(saved_model_dir + float_model_file).to(dev)

    # Next, we'll "fuse modules"; this can both make the model faster by saving on memory access
    # while also improving numerical accuracy. While this can be used with any model, this is
    # especially common with quantized models.

    print('\n Inverted Residual Block: Before fusion \n\n', float_model.features[1].conv)
    float_model.eval()

    # Fuses modules
    float_model.fuse_model()

    # Note fusion of Conv+BN+Relu and Conv+Relu
    print('\n Inverted Residual Block: After fusion\n\n', float_model.features[1].conv)

    print("Size of baseline model")
    print_size_of_model(float_model)

    float_model.to(dev)
    top1, top5 = evaluate(float_model, data_loader_test, neval_batches=num_eval_batches, dev=dev)

    print('Evaluation accuracy on %d images, top-1: %2.2f top-5: %2.2f' % (num_eval_batches * eval_batch_size, top1.avg, top5.avg))
    #Evaluation accuracy on 50000 images, 71.86 90.24

    script_model = torch.jit.script(float_model)
    script_model.save(saved_model_dir + scripted_float_model_file)

def optimized_for_mobile(trace=False):
    # https://pytorch.org/tutorials/recipes/script_optimized.html
    saved_model_dir = '../data/'
    float_model_file = 'mobilenet_v2-b0353104.pth'
    if trace:
        scripted_float_model_file = 'mobilenet_v2_opt_trace_float_scripted.pth'
    else:
        scripted_float_model_file = 'mobilenet_v2_opt_float_scripted.pth'

    dev = 'cpu'

    float_model = load_model(saved_model_dir + float_model_file).to(dev)
    float_model.eval()
    #float_model.fuse_model()

    if trace:
        script_model = torch.jit.trace(float_model, example_inputs=torch.randn((1,3,224,224)))
    else:
        script_model = torch.jit.script(float_model)

    opt_script_model = optimize_for_mobile(script_model)

    opt_script_model.save(saved_model_dir + scripted_float_model_file)

def cvt_optimized():
    path_load = '../data/mobilenet_v2_quant_per_ch_scripted.pth'
    path_save = '../data/mobilenet_v2_opt_quant_per_ch_scripted.pth'
    model = torch.jit.load(path_load)
    from torch.utils.mobile_optimizer import optimize_for_mobile
    model_opt = optimize_for_mobile(model)
    model_opt.save(path_save)

def exp_jit_onnx():
    # ----------------------------------------
    path_load = '../data/mobilenet_v2_opt_quant_per_ch_scripted.pth'
    path_save = '../data/mobilenet_v2_opt_quant_per_ch_scripted.onnx'
    model = torch.jit.load(path_load)
    torch_input = torch.randn(1, 1, 224, 224)

    # onnx_program = torch.onnx.dynamo_export(model, torch_input)
    # onnx_program.save(path_save)

    torch.onnx.export(model, torch_input, path_save, export_params=True, opset_version=10,
                      do_constant_folding=True, input_names=['input'], output_names=['output'])
    # dynamic_axes = {'input': {0: 'batch_size'},  # 가변적인 길이를 가진 차원
    #                 'output': {0: 'batch_size'}})

def exp_onnx():

    # ----------------------------------------
    saved_model_dir = '../data/'
    float_model_file = 'mobilenet_v2-b0353104.pth'
    onnx_fp32_model_file = 'mobilenet_v2_fp32.onnx'
    onnx_fp16_model_file = 'mobilenet_v2_fp16.onnx'
    onnx_quantized_model_file = 'mobilenet_v2_quant_per_ch.onnx'

    num_calibration_batches = 32
    data_loader_train = prepare_data_loader_train()

    path_float_model = saved_model_dir + float_model_file
    float_model = load_model(path_float_model)
    float_model.eval()
    float_model.fuse_model()

    torch_input = torch.randn(1, 3, 224, 224)
    path_save = saved_model_dir + onnx_fp32_model_file
    torch.onnx.export(float_model, torch_input, path_save, export_params=True, opset_version=13,
                      do_constant_folding=True, input_names=['input'], output_names=['output'])
    print('onnx fp32 model saved at ', path_save)

    # fp16_model = float_model.half()
    # fp16_torch_input = torch_input.half()

    # path_save = saved_model_dir + onnx_fp16_model_file
    # torch.onnx.export(fp16_model, fp16_torch_input, path_save, export_params=True, opset_version=13,
    #                   do_constant_folding=True, input_names=['input'], output_names=['output'])
    # print('onnx fp16 model saved at ', path_save)


    print('Per Channel Quantization')
    # The old 'fbgemm' is still available but 'x86' is the recommended default.
    float_model.qconfig = torch.ao.quantization.get_default_qconfig('x86')
    print('qconfig -------')
    print(float_model.qconfig)
    torch.ao.quantization.prepare(float_model, inplace=True)

    print('\n Inverted Residual Block:After observer insertion \n\n', float_model.features[1].conv)

    # Calibrate with the training set
    print('start calibration')
    evaluate(float_model, data_loader_train, num_calibration_batches, dev='cpu')
    print('Per Channel Post Training Quantization: Calibration done')

    # Convert to quantized model
    per_channel_quanti_model = torch.ao.quantization.convert(float_model)
    print('Per Channel Post Training Quantization: Convert done')
    print('\n Inverted Residual Block: After fusion and quantization, note fused modules: \n\n',
          per_channel_quanti_model.features[1].conv)

    print("Size of model after quantization")
    print_size_of_model(per_channel_quanti_model)

    path_save = saved_model_dir + onnx_quantized_model_file

    # onnx_program = torch.onnx.dynamo_export(per_channel_quanti_model, torch_input)
    # onnx_program.save(path_save)

    torch.onnx.export(per_channel_quanti_model, torch_input, path_save, export_params=True, opset_version=13,
                      do_constant_folding=True, input_names=['input'], output_names=['output'])

    print('onnx quantized model saved at ', path_save)


def post_train_static_quant():
    saved_model_dir = '../data/'
    float_model_file = 'mobilenet_v2-b0353104.pth'
    scripted_quantized_model_file = 'mobilenet_v2_quant_scripted.pth'

    num_calibration_batches = 32
    data_loader_train = prepare_data_loader_train()

    print('Post trining Quantization')
    path_float_model = saved_model_dir + float_model_file
    float_model = load_model(path_float_model).to('cpu')
    float_model.eval()
    float_model.fuse_model()

    # Specify quantization configuration
    # Start with simple min/max range estimation and per-tensor quantization of weights
    float_model.qconfig = torch.ao.quantization.default_qconfig
    print('qconfig -------')
    print(float_model.qconfig)
    torch.ao.quantization.prepare(float_model, inplace=True)

    print('\n Inverted Residual Block:After observer insertion \n\n', float_model.features[1].conv)

    # Calibrate with the training set
    print('start calibration')
    evaluate(float_model, data_loader_train, neval_batches=num_calibration_batches, dev='cpu')
    print('Post Training Quantization: Calibration done')

    # Convert to quantized model
    quant_model = torch.ao.quantization.convert(float_model)
    print('Post Training Quantization: Convert done')
    print('\n Inverted Residual Block: After fusion and quantization, note fused modules: \n\n',
          quant_model.features[1].conv)

    print("Size of model after quantization")
    print_size_of_model(quant_model)

    quant_model_scripted = torch.jit.script(quant_model)
    path_save = saved_model_dir + scripted_quantized_model_file
    quant_model_scripted.save(path_save)
    print('saved at ', path_save)

def post_train_static_per_ch_quant():
    saved_model_dir = '../data/'
    float_model_file = 'mobilenet_v2-b0353104.pth'
    scripted_quantized_model_file = 'mobilenet_v2_quant_per_ch_scripted.pth'

    num_calibration_batches = 32
    data_loader_train = prepare_data_loader_train()

    print('Per Channel Quantization')
    path_float_model = saved_model_dir + float_model_file
    float_model = load_model(path_float_model)
    float_model.eval()
    float_model.fuse_model()

    # The old 'fbgemm' is still available but 'x86' is the recommended default.
    float_model.qconfig = torch.ao.quantization.get_default_qconfig('x86')
    print('qconfig -------')
    print(float_model.qconfig)
    torch.ao.quantization.prepare(float_model, inplace=True)

    print('\n Inverted Residual Block:After observer insertion \n\n', float_model.features[1].conv)

    # Calibrate with the training set
    print('start calibration')
    evaluate(float_model, data_loader_train, num_calibration_batches, dev='cpu')
    print('Per Channel Post Training Quantization: Calibration done')

    # Convert to quantized model
    per_channel_quanti_model = torch.ao.quantization.convert(float_model)
    print('Per Channel Post Training Quantization: Convert done')
    print('\n Inverted Residual Block: After fusion and quantization, note fused modules: \n\n',
          per_channel_quanti_model.features[1].conv)

    print("Size of model after quantization")
    print_size_of_model(per_channel_quanti_model)

    per_channel_quanti_model_scripted = torch.jit.script(per_channel_quanti_model)
    path_save = saved_model_dir + scripted_quantized_model_file
    per_channel_quanti_model_scripted.save(saved_model_dir + scripted_quantized_model_file)
    print('saved at ', path_save)

def quant_aware_train():
    saved_model_dir = '../data/'
    float_model_file = 'mobilenet_v2-b0353104.pth'
    device = 'cpu'
    num_train_batches = 20
    num_eval_batches = 1000
    eval_batch_size = 50

    print('QAT')
    path_float_model = saved_model_dir + float_model_file
    qat_model = load_model(path_float_model)
    qat_model.fuse_model(is_qat=True)

    optimizer = torch.optim.SGD(qat_model.parameters(), lr=0.0001)
    # The old 'fbgemm' is still available but 'x86' is the recommended default.
    qat_model.qconfig = torch.ao.quantization.get_default_qat_qconfig('x86')

    torch.ao.quantization.prepare_qat(qat_model, inplace=True)
    print('Inverted Residual Block: After preparation for QAT, note fake-quantization modules \n',
          qat_model.features[1].conv)

    data_loader = prepare_data_loader_train()
    data_loader_test = prepare_data_loader_test()

    criterion = nn.CrossEntropyLoss()
    # QAT takes time and one needs to train over a few epochs.
    # Train and check accuracy after each epoch
    # batch norm 과 quantization param을 초기에 프리즈하여
    # weight가 더 잘 학습되도록 한다.

    for nepoch in range(8):
        train_one_epoch(qat_model, criterion, optimizer, data_loader, device, num_train_batches)

        if nepoch > 2:
            # Freeze batch norm mean and variance estimates
            qat_model.apply(torch.nn.intrinsic.qat.freeze_bn_stats)

        if nepoch > 3:
            # Freeze quantizer parameters(scale and zero-point)
            qat_model.apply(torch.ao.quantization.disable_observer)

        # Check the accuracy after each epoch
        quantized_model = torch.ao.quantization.convert(qat_model.eval(), inplace=False)
        quantized_model.eval()
        top1, top5 = evaluate(quantized_model, criterion, data_loader_test, neval_batches=num_eval_batches, dev=device)
        print('\nEpoch %d :Evaluation accuracy on %d images, %2.2f %2.2f' % (
        nepoch, num_eval_batches * eval_batch_size, top1.avg, top5.avg))

    quantized_model_scripted = torch.jit.script(quantized_model)
    path_save = saved_model_dir + 'mobilenet_v2_qat_scripted.pth'
    quantized_model_scripted.save(path_save)

def check_scripted_model_acc():
    #path_model = '../data/mobilenet_v2_float_scripted.pth'
    #path_model = '../data/mobilenet_v2_quant_scripted.pth'
    #path_model = '../data/mobilenet_v2_quant_per_ch_scripted.pth'
    path_model = '../data/mobilenet_v2_qat_scripted.pth'

    if 'quant' in path_model or 'qat' in path_model:
        device = 'cpu'
    else:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model_scripted = torch.jit.load(path_model)
    model_scripted = model_scripted.to(device)
    model_scripted = model_scripted.eval()

    num_eval_batches = 100 # 1000
    eval_batch_size = 50
    data_loader = prepare_data_loader_test(eval_batch_size)

    top1, top5 = evaluate(model_scripted, data_loader, num_eval_batches, dev=device)
    print('Evaluation accuracy on %d images, top-1: %2.2f top-5: %2.2f' % (num_eval_batches * eval_batch_size, top1.avg, top5.avg))
    # mobilenet_v2_float_scripted.pth
    # Evaluation accuracy on 5000 images, top-1: 78.46 top-5: 93.44

    # mobilenet_v2_quant_scripted.pth
    # Evaluation accuracy on 5000 images, top-1: 65.00 top-5: 86.72

    # mobilenet_v2_quant_per_ch_scripted.pth
    # Evaluation accuracy on 5000 images, top-1: 75.54 top-5: 92.10

    # mobilenet_v2_qat_scripted.pth
    # Evaluation accuracy on 5000 images, top-1: 74.38 top-5: 91.78

def print_model():
    #path_model = '../data/mobilenet_v2_quant_scripted.pth'
    #path_model = '../data/mobilenet_v2_quant_per_ch_scripted.pth'
    path_model = '../data/mobilenet_v2_qat_scripted.pth'

    if 'quant' in path_model or 'qat' in path_model:
        device = 'cpu'
    else:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model_scripted = torch.jit.load(path_model)
    model_scripted = model_scripted.to(device)
    model_scripted = model_scripted.eval()

    print(model_scripted)

def check_w():
    saved_model_dir = '../data/'
    float_model_file = 'mobilenet_v2-b0353104.pth'

    print('QAT')
    path_float_model = saved_model_dir + float_model_file
    qat_model = load_model(path_float_model)
    qat_model.fuse_model(is_qat=True)

    # The old 'fbgemm' is still available but 'x86' is the recommended default.
    qat_model.qconfig = torch.ao.quantization.get_default_qat_qconfig('x86')

    torch.ao.quantization.prepare_qat(qat_model, inplace=True)

    for i in range(2):
        with torch.no_grad():
            image = torch.randn((50,3,224,224))
            qat_model(image)

    # Check the accuracy after each epoch
    quantized_model = torch.ao.quantization.convert(qat_model.eval(), inplace=False)
    quantized_model.eval()

    print(quantized_model)

    for n, m in quantized_model.named_modules():
        print(n, type(m))

        if isinstance(m, torch.ao.nn.quantized.modules.conv.Conv2d):
             print(m.weight)

        if isinstance(m, torch.ao.nn.intrinsic.quantized.modules.conv_relu.ConvReLU2d):
             print(m.weight)

    for n, p in quantized_model.named_parameters():
        print(n, p.shape)

if __name__ == '__main__':
    #baseline()
    #optimized_for_mobile(True)
    #post_train_static_quant()
    #post_train_static_per_ch_quant()
    #quant_aware_train()

    #check_scripted_model_acc()
    #print_model()
    #check_w()
    #cvt_optimized()
    #exp_jit_onnx()
    exp_onnx()

