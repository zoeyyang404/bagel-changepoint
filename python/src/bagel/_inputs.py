"""Convert tensor inputs without loading PyTorch for NumPy users."""
from __future__ import annotations

import sys
from numpy import ndarray


def tensor_to_numpy(value, name: str):
    """Convert a dense, real tensor to CPU NumPy data; return other inputs as-is.

    Code that creates a tensor has already imported torch. Looking up that
    module lets us recognise tensors without importing torch ourselves.
    """
    if isinstance(value, ndarray):
        return value
    torch = sys.modules.get("torch")
    if torch is None or not isinstance(value, torch.Tensor):
        return value
    if value.is_nested or value.layout != torch.strided or value.is_quantized:
        raise TypeError(f"{name} must be a dense, non-quantized PyTorch tensor")
    if value.is_complex():
        raise TypeError(f"{name} must be real-valued")
    if value.device.type == "meta":
        raise ValueError(f"{name} must contain data; meta tensors are not supported")
    # Transfer before casting: some devices cannot represent float64. Casting
    # on the CPU also supports bfloat16, which NumPy cannot directly consume.
    return value.detach().cpu().to(dtype=torch.float64).resolve_neg().numpy()
