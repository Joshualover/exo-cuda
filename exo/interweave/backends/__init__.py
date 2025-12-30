"""
Interweave Backend Implementations

Available backends:
- llamacpp: llama.cpp CPU backend (supports quantized models)
- tinygrad_cuda: TinyGrad CUDA backend (GPU inference)
- tinygrad_cpu: TinyGrad CPU backend (fallback)
"""

# Import backends to trigger registration
from . import llamacpp
from . import tinygrad_cuda

# Re-export backend classes for convenience
from .llamacpp import LlamaCppBackend
from .tinygrad_cuda import TinygradCudaBackend, TinygradCpuBackend, TinygradInterweaveAdapter

__all__ = [
    'llamacpp',
    'tinygrad_cuda',
    'LlamaCppBackend',
    'TinygradCudaBackend',
    'TinygradCpuBackend',
    'TinygradInterweaveAdapter',
]
