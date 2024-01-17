from qtorch.quant import Quantizer, quantizer
from qtorch import FloatingPoint
from typing import Optional
import torch.nn as nn

class Id(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()

    def forward(self, x):
        return x

def make_quant_func(args):
    def _make_float(mantissa_bit_width: Optional[int] = None, exponent_bit_width: Optional[int] = None):
        return FloatingPoint(exp=exponent_bit_width, man=mantissa_bit_width)
    make_ae_quant = lambda : Quantizer(forward_number=_make_float(args.activation_bw, args.activation_ew), 
                                       backward_number=_make_float(args.error_bw, args.error_ew),
                                       forward_rounding=args.activation_round, 
                                       backward_rounding=args.error_round)
    weight_quant = quantizer(forward_number=_make_float(args.weight_bw, args.weight_ew),
                                  forward_rounding=args.weight_round)
    grad_quant = quantizer(forward_number=_make_float(args.gradient_bw, args.gradient_ew),
                                forward_rounding=args.gradient_round)
    return {
        "make_ae_quant": make_ae_quant,
        "weight_quant": weight_quant,
        "grad_quant": grad_quant,
    }