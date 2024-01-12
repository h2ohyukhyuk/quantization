# https://pytorch.org/tutorials/prototype/pt2e_quant_ptq_x86_inductor.html

import torch
import torchvision.models as models
import copy
from torch.ao.quantization.quantize_pt2e import prepare_pt2e, convert_pt2e
import torch.ao.quantization.quantizer.x86_inductor_quantizer as xiq
from torch.ao.quantization.quantizer.x86_inductor_quantizer import X86InductorQuantizer
from torch._export import capture_pre_autograd_graph

# Create the Eager Model
model_name = "resnet18"
model = models.__dict__[model_name](pretrained=True)

# Set the model to eval mode
model = model.eval()

# Create the data, using the dummy data here as an example
traced_bs = 50
x = torch.randn(traced_bs, 3, 224, 224).contiguous(memory_format=torch.channels_last)
example_inputs = (x,)

# Capture the FX Graph to be quantized
with torch.no_grad():
     # if you are using the PyTorch nightlies or building from source with the pytorch master,
    # use the API of `capture_pre_autograd_graph`
    # Note 1: `capture_pre_autograd_graph` is also a short-term API, it will be updated to use the official `torch.export` API when that is ready.
    # exported_model = capture_pre_autograd_graph(
    #     model,
    #     example_inputs
    # )
    # Note 2: if you are using the PyTorch 2.1 release binary or building from source with the PyTorch 2.1 release branch,
    # please use the API of `torch._dynamo.export` to capture the FX Graph.
    exported_model, guards = torch._dynamo.export(
        model,
        *copy.deepcopy(example_inputs),
        aten_graph=True,
    )

quantizer = X86InductorQuantizer()
quantizer.set_global(xiq.get_default_x86_inductor_quantization_config())

prepared_model = prepare_pt2e(exported_model, quantizer)

# We use the dummy data as an example here
prepared_model(*example_inputs)

# Alternatively: user can define the dataset to calibrate
# def calibrate(model, data_loader):
#     model.eval()
#     with torch.no_grad():
#         for image, target in data_loader:
#             model(image)
# calibrate(prepared_model, data_loader_test)  # run calibration on sample data

converted_model = convert_pt2e(prepared_model)

with torch.no_grad():
    optimized_model = torch.compile(converted_model)

    # Running some benchmark
    optimized_model(*example_inputs)

with torch.autocast(device_type="cpu", dtype=torch.bfloat16, enabled=True), torch.no_grad():
    # Turn on Autocast to use int8-mixed-bf16 quantization. After lowering into Inductor CPP Backend,
    # For operators such as QConvolution and QLinear:
    # * The input data type is consistently defined as int8, attributable to the presence of a pair
    #   of quantization and dequantization nodes inserted at the input.
    # * The computation precision remains at int8.
    # * The output data type may vary, being either int8 or BFloat16, contingent on the presence
    #   of a pair of quantization and dequantization nodes at the output.
    # For non-quantizable pointwise operators, the data type will be inherited from the previous node,
    # potentially resulting in a data type of BFloat16 in this scenario.
    # For quantizable pointwise operators such as QMaxpool2D, it continues to operate with the int8
    # data type for both input and output.
    optimized_model = torch.compile(converted_model)

    # Running some benchmark
    optimized_model(*example_inputs)