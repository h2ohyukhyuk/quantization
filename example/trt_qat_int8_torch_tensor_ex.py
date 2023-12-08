#https://pytorch.org/TensorRT/_notebooks/vgg-qat.html

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
#import torch_tensorrt

from torch.utils.tensorboard import SummaryWriter

import pytorch_quantization
from pytorch_quantization import nn as quant_nn
from pytorch_quantization import quant_modules
from pytorch_quantization.tensor_quant import QuantDescriptor
from pytorch_quantization import calib
from tqdm import tqdm

print(pytorch_quantization.__version__)

import os
import sys
sys.path.insert(0, "../examples/int8/training/vgg16")
from vgg16 import vgg16