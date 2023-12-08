'''
https://pytorch.org/tutorials/intermediate/realtime_rpi.html
라즈베리 파이 예제
windows에서 qnnpack  RuntimeError: quantized engine QNNPACK is not supported
'''
import torch
from torchvision import models
from torchvision import transforms
import cv2
from PIL import Image
import time


classes = []
with open('data/imagenet1000_clsidx_to_label.txt') as f:
    for line in f.readlines():
        cls = line.replace('{', '').replace('}', '').rstrip()
        classes.append(cls)

# for cls in classes:
#     print(cls)

torch.backends.quantized.engine = 'fbgemm' # all_engines = {0: "none", 1: "fbgemm", 2: "qnnpack", 3: "onednn", 4: "x86"}
#net = models.quantization.mobilenet_v2(pretrained=True, quantize=True)

net = models.mobilenet_v2(pretrained=True)
#net = models.mobilenet_v3_small(pretrained=True)
net.eval()
net = torch.jit.script(net)

preprocess = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])



#path_img = 'images/chimpanzee.jpg'
path_img = 'images/goldfish.png'

with torch.no_grad():
    image = cv2.imread(path_img)
    image = cv2.resize(image, dsize=(224, 224))
    image = image[:,:, [2,1,0]]
    # cv2.imshow('', image)
    # cv2.waitKey()
    permuted = image

    started = time.time()
    # preprocess
    input_tensor = preprocess(image)

    # create a mini-batch as expected by the model
    input_batch = input_tensor.unsqueeze(0)
    # torch.min(input_tensor)
    # torch.max(input_tensor)
    # torch.mean(input_tensor)
    # input_tensor.shape

    # run model
    output = net(input_batch)

    now = time.time()
    print(f"{now - started}")

    top = list(enumerate(output[0].softmax(dim=0)))
    top.sort(key=lambda x: x[1], reverse=True)
    for idx, val in top[:10]:
        print(f"{val.item() * 100:.2f}% {classes[idx]}")