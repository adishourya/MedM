# %%
import einops
from model_siglip import SiglipVIT, SiglipConfig , sample_img
from transformers import AutoProcessor, PaliGemmaForConditionalGeneration
from PIL import Image
import torch
import gc
from torchvision import transforms

# %% Load SiglipVIT model
siglip_config = SiglipConfig()  # Assuming you have the config for your SiglipVIT
siglip_vit = SiglipVIT(siglip_config).to("cuda")  # Move SiglipVIT to GPU

# %% Setup PaliGemma and processor
model_id = "google/paligemma-3b-mix-224"
device = "cuda:0"
dtype = torch.bfloat16 # faster inference

# %% Load Pretrained
model = PaliGemmaForConditionalGeneration.from_pretrained(
    model_id,
    torch_dtype=dtype,
    device_map=device,
    revision="bfloat16",
).eval()
processor = AutoProcessor.from_pretrained(model_id)

# %% concat img tokens and prefix prompt
image = Image.open("sample_img.png")
processed_inputs = processor(text="caption en Is there something wrong with this person?", images=image, return_tensors="pt").to(device)

pixel_values = processed_inputs["pixel_values"]
input_ids = processed_inputs["input_ids"]
attention_mask = processed_inputs["attention_mask"]

# Pass SiglipVIT features
with torch.no_grad():
    image_features = siglip_vit(sample_img.to(device))

# %% Modify inputs
model_inputs = {
    "input_ids": input_ids,
    "attention_mask": attention_mask,
    "pixel_values": pixel_values,  # Use processor-generated pixel_values for compatibility
}

input_len = input_ids.shape[-1]

# %% inference ::
with torch.inference_mode():
    generation = model.generate(**model_inputs, max_new_tokens=100, do_sample=False)
    generation = generation[0][input_len:]
    decoded = processor.decode(generation, skip_special_tokens=True)
    print(decoded)

# %% Clean up memory
del model
del processor
gc.collect()
torch.cuda.synchronize()
torch.cuda.empty_cache()
torch.cuda.ipc_collect()
torch.cuda.reset_max_memory_allocated()
