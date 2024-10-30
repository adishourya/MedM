from transformers import AutoProcessor, PaliGemmaForConditionalGeneration , BitsAndBytesConfig
from PIL import Image
import torch
import os

import matplotlib.pyplot as plt
import matplotx
plt.style.use(matplotx.styles.pacoty)

# 4bit quantization
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

# model_id = "google/paligemma-3b-mix-224"
model_id = "google/paligemma-3b-mix-448"
device = "cuda:0"
dtype = torch.bfloat16 ## this could even be int8.

image_dir = '../../datasets/MEDPIX-ClinQA/'
image_path = os.path.join(image_dir, 'sample/MPX1007_synpic46722.png')
image = plt.imread(image_path)


model = PaliGemmaForConditionalGeneration.from_pretrained(
    model_id,
    torch_dtype=dtype,
    device_map=device,
    quantization_config=bnb_config,
    # revision="bfloat16",
).eval()
processor = AutoProcessor.from_pretrained(model_id)

prompt = "answer en what abnormalities does the patient have ?"
model_inputs = processor(text=prompt, images=image, return_tensors="pt").to(model.device)
input_len = model_inputs["input_ids"].shape[-1]

with torch.inference_mode():
    generation = model.generate(**model_inputs, max_new_tokens=100, do_sample=False)
    generation = generation[0][input_len:]
    decoded = processor.decode(generation, skip_special_tokens=True)
    # print(decoded)

plt.imshow(image)
plt.suptitle("Q: " + prompt)
plt.title("A: "+decoded)
plt.show()
