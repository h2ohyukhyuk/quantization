# https://pytorch.org/tutorials/prototype/pt2e_quant_ptq.html


def simple_ex():
    import torch
    from torch._export import capture_pre_autograd_graph

    class M(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(5, 10)

        def forward(self, x):
            return self.linear(x)

    example_inputs = (torch.randn(1,5),)
    m = M().eval()

    # Step 1. program capture
    # NOTE: this API will be updated to torch.export API in the future, but the captured
    # result should mostly stay the same
    m = capture_pre_autograd_graph(m, *example_inputs)
    # we get a model with aten ops

    # Step 2. quantization
    from torch.ao.quantization.quantize_pt2e import (prepare_pt2e, convert_pt2e)

    from torch.ao.quantization.quantizer import (XNNPACKQuantizer, get_symmetric_quantization_config)

    # backend developer will write their own Quantizer and expose methods to allow
    # users to express how they
    # want the model to be quantized
    quantizer = XNNPACKQuantizer().set_global(get_symmetric_quantization_config())
    m = prepare_pt2e(m, quantizer)

    # calibration omitted

    m = convert_pt2e(m)
    # we have a model with aten ops doing integer computations when possible

simple_ex()