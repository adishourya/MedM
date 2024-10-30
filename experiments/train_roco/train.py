import torch
from torch.utils.data import Dataset
from typing import Any , List , Dict
from datasets import load_dataset
from transformers import (
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
    AutoProcessor,
    PaliGemmaForConditionalGeneration,
)
from peft import get_peft_model, LoraConfig
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import einops
# for image handling
import PIL.Image as PILImage
import io
import datetime

model_id = "google/paligemma-3b-mix-224"
huggingface_card = ""
# start with default
input_max_length :int

wand_b_project = ""
wand_b_name = ""


processor = AutoProcessor.from_pretrained(model_id)
tokenizer = processor.tokenizer

from datasets import load_dataset
train_dataset = load_dataset("adishourya/ROCO-QA",split="Vaild")

# In [4]: dataset
# Out[4]:
# Dataset({
#     features: ['image_id', 'image', 'question', 'answer'],
#     num_rows: 8175
# })

class Example():
    def __init__(self,dataset):
        self.dataset = dataset

    # checkout an example
    @staticmethod
    def see_example(dataset):
        random_index = np.random.randint(low=0,high=len(dataset),size=1)[0]
        example = dataset[int(random_index)]
        print(f"Image Id : {example["image_id"]}")
        example_img = example["image"]["bytes"]
        # force rgb only works with b,3,h,w
        pil_img = PILImage.open(io.BytesIO(example_img)).convert("RGB")
        pil_img.show()
        print(example["question"], example["answer"])
        return {"question":example["question"] , 
                "image":pil_img ,
                "answer":example["answer"]}

    # tokenization_example
    @staticmethod
    def tokenization_example(example:dict):
        question = example["question"]
        image = example["image"]
        answer = example["answer"]

        tokens_out = processor(
            text=question,
            images=image,
            suffix=answer,
            return_tensors="pt",  # this was originally trained with jax
            padding="longest",  # pad to the longest answer which is about 80 words
            # padding = True,
            tokenize_newline_separately = False,
            # max_length = input_max_length
        )
        # returns -> [input_ids , token_type_ids, attention_mask, pixel_values , labels]
        # of shape -> torch.Size([1, 299]) torch.Size([1, 299]) torch.Size([1, 299]) torch.Size([1, 3, 224, 224]) torch.Size([1, 299])
        tokens_input_id = tokens_out["input_ids"]
        token_type_ids = tokens_out["token_type_ids"]
        attention_mask = tokens_out["attention_mask"]
        vision_out = tokens_out["pixel_values"]
        labels = tokens_out["labels"]

        print(tokens_input_id.shape ,
              token_type_ids.shape,
              attention_mask.shape,
              vision_out.shape,
              labels.shape)

        return dict(input_ids = tokens_input_id,
                    token_type_ids = token_type_ids,
                    attention_mask = attention_mask,
                    pixel_values = vision_out,
                    labels = labels)

    @staticmethod
    def decode_example(tokens_out:dict):
        print("decoding pixel values")
        img_decoded = einops.rearrange(tokens_out["pixel_values"],"1 c h w -> h w c") 
        plt.imshow(img_decoded)
        plt.show()
        print("input id")
        print(processor.batch_decode(tokens_out["input_ids"]))
        print("attention mask is all 1 ?")
        print(tokens_out["attention_mask"].all())

    def one_example(self):
        eg = self.see_example(self.dataset)
        # print(eg)
        tokens_out = self.tokenization_example(eg)
        self.decode_example(tokens_out)






# create dataset
class ROCODataset(Dataset):
    def __init__(self,dataset):
        super().__init__()
        self.dataset = dataset
        self.dataset_length = len(dataset) # dataset["num_rows"]
        # we use wordpiece tokenizer
        self.examples = []


# create collate function

# see processor decode with label

# 
