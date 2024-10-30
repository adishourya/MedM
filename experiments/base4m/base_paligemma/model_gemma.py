from dataclasses import dataclass
import einops
import matplotlib.pyplot as plt
import matplotx
plt.style.use(matplotx.styles.pacoty)

import torch
import torch.nn as nn

from model_siglip import SiglipConfig , SiglipVIT



@dataclass
class GemmaConfig():
  vocab_size:int
  d_model:int
  d_ff:int
