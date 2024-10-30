# %%
from typing import Dict , List , Tuple
import einops
import matplotlib.pyplot as plt
import matplotx
from torchvision.io.image import ImageReadMode
from torchvision.transforms.functional import Optional
plt.style.use(matplotx.styles.pacoty)

import torch
import torch.nn as nn

from torchvision.io import read_image
from torchvision.transforms import transforms

# %%

IMAGE_MEAN = 0.5
IMAGE_STD  = 0.5

def concat_tokens(prefix_prompt , img_seq_len:int, img_identifier , bos_token):
  # as 1d row vector. Fig 1 page 3
  # concats the image token from vit and prompts from sentencepiece
  # care for \n :: The BOS token then
  # marks the start of text tokens. We use \n as SEP
  # token, it does not appear in any of our prefixes.
  return f"{img_identifier * img_seq_len}{bos_token}{prefix_prompt}\n"


def img_transform(img,size=448):
  # from page 3
  # We always resize the image to a fixed square
  # size (224, 448, or 896 pixels). This leads to a
  # fixed number of image tokens per model vari-
  # ant (respectively 256, 1024, or 4096 tokens)
  # normalize expects the input to be be b/n 0,1
  # normalization of img token : page 6
  composed_transformations = transforms.Compose([
    transforms.ConvertImageDtype(torch.float),
    transforms.Lambda(lambda x : x/255.0),
    transforms.Resize(size=size),
    transforms.Normalize(mean=IMAGE_MEAN,std=IMAGE_STD),
  ])
  out = composed_transformations(img)
  # add batch dimension
  out = einops.rearrange(out,"c h w -> 1 c h w")
  return out

# %%
def transformations_exemplar():
  sample_img = read_image("sample_img.png",mode=ImageReadMode.GRAY)
  plt.imshow(einops.rearrange(sample_img,"c h w -> h w c"))
  plt.show()
  out = img_transform(sample_img)
  print(out.mean() , out.std())
  plt.imshow(einops.rearrange(out,"1 c h w -> h w (c 1)"))
  plt.show()
transformations_exemplar()

# %%
class PaligemmaProcessor:
  img_identifier = "<img>"

  def __init__(self,tokenizer,img_seq_len , img_size) -> None:
    """
    # page 3
    tokens = [image tokens...,
    BOS, prefix tokens..., SEP,
    suffix tokens..., EOS, PAD...]
    """
    super().__init__()
    self.img_seq_len = img_seq_len
    self.img_size = img_size

    # see line 38 hf processing paligemma [these were hardcoded]
    # sentencepiece would need the special tokens we are going to add
    # 1. img identifier 2. bounding box 3. segementation
    img_special_token = {"additional_special_tokens":[self.img_identifier]}
    # the ">" would pad 0 to the left of num . so this goes 0000 , 0001 -> 1024
    bbox_special_token = [f"<loc{i:0>4}>" for i in range(1024)]
    seg_special_token = [f"<seg{i:0>3}>" for i in range(128)]
    tokenizer.add_special_tokens(img_special_token)
    tokenizer.add_tokens(bbox_special_token + seg_special_token)
    self.img_token_id = tokenizer.convert_tokens_to_ids(self.img_identifier)

    # beginning and end of sentence token [we would add this ourselves]
    tokenizer.add_bos_token = False
    tokenizer.add_eos_token = False

    # init the tokenizer
    self.tokenizer = tokenizer

  def __call__(self,
    text:List[str],
    images,
    padding:str = "longest",
    truncation:bool = True
  ) -> dict:

    assert len(images) == len(text) == 1 , f"Recieved {len(images)} and {len(text)}"
    transformed_img = img_transform(images,size=self.img_size)

    input_compiled = [
      concat_tokens(prefix_prompt=prompt,
        bos_token=self.tokenizer.bos_token,
        img_seq_len=self.img_seq_len,
        img_identifier=self.img_identifier)
      for prompt in text
    ]

    input_tokenized = self.tokenizer(
      input_compiled,
      return_tensors = "pt",
      padding = padding,
      truncation = truncation
    )

    dict_data = {"image":transformed_img, **input_tokenized }
    return dict_data


# %%
def exemplar():
  pass
exemplar()
