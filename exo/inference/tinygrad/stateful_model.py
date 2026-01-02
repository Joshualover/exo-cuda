from tinygrad import Tensor, Variable, dtypes
from tinygrad.helpers import getenv
from collections import OrderedDict
from typing import List, Optional

def create_kv_cache(x: Tensor, layer):
  # USE_FP32=1 forces f32 for devices without fp16 support (e.g., FirePro D500)
  cache_dtype = dtypes.float32 if getenv("USE_FP32", 0) else x.dtype
  cache_kv = Tensor.zeros(2, x.shape[0], layer.max_context, layer.n_kv_heads, layer.head_dim, dtype=cache_dtype).contiguous().realize()
  if isinstance(x.device, tuple):
    # TODO: instead of specifying how to shard, it can follow how xk and xv are being sharded
    cache_kv.shard_((x.device), axis=3 if getenv("SHARD_KVCACHE") else None).realize()
  return cache_kv.realize()

class ModelState:
  cache: List[Tensor]
  start: int 
  def __init__(self, cache: List[Tensor], start: int = 0):
    self.cache = cache
    self.start = start

def make_prompt_state(x: Tensor, model):
  cache = [create_kv_cache(x, l.attention) for l in model.layers]

  return ModelState(cache)
