# https://huggingface.co/blog/kv-cache-quantization
# https://github.com/hkproj/pytorch-paligemma/blob/main/modeling_gemma.py

# More concretely, key-value cache acts as a memory bank for autoregressive generative models,
# where the model stores key-value pairs derived from self-attention layers for previously processed tokens

from dataclasses import dataclass
import einops
import torch
import torch.nn as nn
from torchvision import datasets

class KVCache():
  def __init__(self) -> None:
    # at init the cache would be empty
    self.key_cache = []
    self.value_cache = []

  def num_items(self) -> int:
    # return T of B h T d_k
    if len(self.key_cache) == 0:
      return 0
    return self.key_cache[0].shape[-2]

  def update(self,
    key_states:torch.Tensor,
    value_states:torch.Tensor,
    layer_idx:int):
      if len(self.key_cache) <= layer_idx:
        self.key_cache.append(key_states)
        self.value_cache.append(value_states)
      else:
        # update the cache
        self.key_cache[layer_idx] = torch.cat([self.key_cache[layer_idx], key_states],dim=-2)
        self.value_cache[layer_idx] = torch.cat([self.value_cache[layer_idx],value_states],dim=-2)

      return self.key_cache[layer_idx] , self.value_cache[layer_idx]
