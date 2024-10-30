# %%
import einops
import numpy as np
import matplotlib.pyplot as plt
import matplotx
plt.style.use(matplotx.styles.pacoty)

from torchvision.io import read_image
from torchvision.io.image import ImageReadMode
from torchvision import transforms

import torch
import torch.nn as nn
import torch.functional as F
from dataclasses import dataclass

# %%
@dataclass
class SiglipConfig:
  d_model:int = 768
  d_ff:int = d_model * 4
  num_hidden_layers:int = 12
  num_heads : int = 12
  assert d_model % num_heads == 0 , "Produced Invalid head size"
  num_channel: int = 1 # Medpix
  image_size: int = 224
  patch_size: int = 16 # decrease for higher resolution
  eps:float = 1e-6
  attn_drop_prob:float = 0.0 # important to not have dropout
  assert image_size % patch_size == 0, "Image size not divisible by patch size"
  num_patches :int = int(image_size/patch_size)**2
  causal:bool = False # images are not causal
  num_image_tokens = None

sample_config = SiglipConfig()


# %%
# we will use this image as sample image
sample_img = read_image("sample_img.png",ImageReadMode.GRAY)
def prepare_sample(config,sample_img):
  print("input size",sample_img.shape)
  # we will keep this as the image size in the config
  transformations = transforms.Compose(
    [
    transforms.Resize((config.image_size)),
    transforms.ConvertImageDtype(torch.float),
    ]
  )
  sample_img = transformations(sample_img)
  print("After Resizing:" ,sample_img.shape, sample_img.dtype)

  # ALL images are greyscaled
  plt.imshow(einops.rearrange(sample_img,"1 h w -> h w 1"),cmap="grey")
  plt.show()

  # add batch dimension
  sample_img = einops.rearrange(sample_img,"c h w -> 1 c h w")
  print("Final Image shape",sample_img.shape)
  return sample_img
sample_img = prepare_sample(sample_config,sample_img)


# %%
class SiglipEmbedding(nn.Module):
  def __init__(self, config:SiglipConfig) -> None:
    super().__init__()
    self.config = config

    # b,c,h,w -> b,d_model,T,T (irrespective of the input channel size)
    # so here 1,1,224,224 -> 1, d_model,224/16,224/16
    # so each patch gets an embedding like wte for image
    self.patch_embedding = nn.Conv2d(in_channels=config.num_channel,
      out_channels=config.d_model,
      kernel_size=config.patch_size,
      stride=config.patch_size,
      padding="valid") # valid does not pad with 0's

    # pe
    self.num_positions = config.num_patches
    self.position_embeding = nn.Embedding(
      num_embeddings=config.num_patches,
      embedding_dim=config.d_model)

    # make this a row vector
    self.register_buffer("position_id",
      torch.arange(self.num_positions).reshape(1,-1),
      persistent=False
    )

  def forward(self,x:torch.Tensor):
    # x is just an image
    patch_emb = self.patch_embedding(x) # B,d,T,T
    # like translation we B T d
    patch_emb = einops.rearrange(patch_emb,"B d T1 T2 -> B (T1 T2) d")
    emb = patch_emb + self.position_embeding(self.position_id)
    return emb

# %%
# exemplar run
def exemplar(config):
  with torch.no_grad():
    model = SiglipEmbedding(config)
    forward_out = model(sample_img)

    print("------FWD PASS--------")
    print("input ",sample_img.shape)
    print("-------")
    print("wte")
    patch_emb = model.patch_embedding(sample_img)
    print("patch emb out :",patch_emb.shape)
    patch_emb = einops.rearrange(patch_emb , "1 d t1 t2 -> 1 (t1 t2) d")
    print("patch emb out after transpose :",patch_emb.shape)
    print(patch_emb.shape)
    plt.imshow(einops.rearrange(patch_emb, "1 T d -> T d"))
    plt.show()
    print("-------")
    pe = model.position_embeding(model.position_id)
    print("pe",pe.shape)
    plt.imshow(einops.rearrange(pe,"1 t d -> t d"))
    plt.show()
    print("-------")
    out = patch_emb + pe
    print("out",out.shape)
    plt.imshow(einops.rearrange(out , "1 t d -> t d"))
    plt.show()

    assert torch.allclose(out,forward_out) , "Incorrect implementation"
    print("All Tests Passed!")
exemplar(config=sample_config)


# %%
class SiglipAttention(nn.Module):
  def __init__(self,config:SiglipConfig) -> None:
    super().__init__()
    self.config = config

    self.d_model = config.d_model
    self.num_heads = config.num_heads
    self.d_k = self.d_model // self.num_heads

    # weights for all the heads
    # split vertically to get Wk , Wv
    self.Wkv = nn.Linear(self.d_model , self.d_model*2)
    self.Wq = nn.Linear(self.d_model , self.d_model)
    self.Wo = nn.Linear(self.d_model , self.d_model)

  def forward(self,source_q:torch.Tensor , source_kv:torch.Tensor):
    # source shape : B,T,d
    kv = self.Wkv(source_kv)
    q = self.Wq(source_q)
    k,v = torch.chunk(kv,chunks=2,dim=2)
    # q,k,v still : shape B,t,d

    # give h to infer splitting
    q = einops.rearrange(q,"B T (h d_k) -> B h T d_k",h=self.num_heads)
    k = einops.rearrange(k,"B T (h d_k) -> B h T d_k",h=self.num_heads)
    v = einops.rearrange(v,"B T (h d_k) -> B h T d_k",h=self.num_heads)

    # (B h T d_k @ B h d_k T ) @ B h T d_k
    # B h T T @ B h T d_k -> B h T d_k
    out = nn.functional.scaled_dot_product_attention(q,k,v,
      attn_mask=None,dropout_p=0,is_causal=False)

    # back to keep token in second position : B,T,d
    out = einops.rearrange(out,"B h T d_k -> B T (h d_k)")

    # BTd @ B d,d -> B T d
    out = self.Wo(out)

    return out

# %%
def exemplar(config,img):
  with torch.no_grad():
    emb_model = SiglipEmbedding(config)
    attn_model = SiglipAttention(config)
    emb_out = emb_model(img)
    plt.imshow(einops.rearrange(emb_out, "1 T d -> T d"))
    plt.show()
    print("emb out is of the shape B T d",emb_out.shape)
    # this is how we would do self attention
    out = attn_model(source_kv=emb_out , source_q = emb_out)
    plt.imshow(einops.rearrange(out, "1 T d -> T d"))
    print(out.shape)
exemplar(sample_config,sample_img)

# %%
class SiglipMlp(nn.Module):
  def __init__(self, config:SiglipConfig) -> None:
    super().__init__()
    self.d_model = config.d_model
    self.d_ff = config.d_ff
    self.W1 = nn.Linear(self.d_model , self.d_ff)
    self.W2 = nn.Linear(self.d_ff,self.d_model)

  def forward(self,x:torch.Tensor):
    out = self.W1(x)
    out = nn.functional.gelu(input=out,approximate="tanh")
    out = self.W2(out)
    return out


# %%
class SiglipSingleEncoder(nn.Module):
  def __init__(self,config:SiglipConfig) -> None:
    super().__init__()
    self.config = config
    self.d_model = config.d_model
    self.selfattn = SiglipAttention(config=self.config)
    self.ln1 = nn.LayerNorm(normalized_shape=self.d_model)
    self.mlp = SiglipMlp(config=self.config)
    self.ln2 = nn.LayerNorm(normalized_shape=self.d_model)

    # ln->attn->mlp(c)->ln->mlp(c)
  def forward(self,x):
    # save it to connect later (skip connection)
    residual = x
    out = self.ln1(x)
    out = self.selfattn(source_kv = out, source_q = out)
    out = out + residual

    residual = out
    out = self.ln2(out)
    out = self.mlp(out)
    out = out + residual
    return out
    # assert shape to be B , T ,d

def exemplar(config,img):
  emb_model = SiglipEmbedding(config)
  x = emb_model(img)
  single_encoder = SiglipSingleEncoder(config)
  out = single_encoder(x)
  print(out.shape)

exemplar(config=sample_config,img=sample_img)

# %%
class SiglipEncoder(nn.Module):
  def __init__(self,config:SiglipConfig):
    super().__init__()
    self.config = config
    self.repeat = config.num_hidden_layers
    self.allEncoders = nn.ModuleList([
      SiglipSingleEncoder(config) for _ in range(self.repeat)
    ])

  def forward(self,x:torch.Tensor):
    # we expect this after the embedding is done
    # so x is after wte + pe
    out = x
    for encoder in self.allEncoders():
      out = encoder(out)
    return out


# %%
# embedding -> encoders -> ln
class SiglipVIT(nn.Module):
  def __init__(self, config:SiglipConfig) -> None:
    super().__init__()
    self.config = config
    self.emb = SiglipEmbedding(self.config)
    self.Encoders = SiglipEncoder(self.config)
    self.postEncLn = nn.LayerNorm(normalized_shape=config.d_model)

  def forward(self,x):
    emb_out = self.emb(x)
    out = self.Encoders(emb_out)
    out  =self.postEncLn(out)
    return out
