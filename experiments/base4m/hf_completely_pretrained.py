# %%
from transformers import AutoProcessor, PaliGemmaForConditionalGeneration
from PIL import Image
import requests
import torch
import matplotlib.pyplot as plt
import gc

# %%
model_id = "google/paligemma-3b-mix-224"
device = "cuda:0"
dtype = torch.bfloat16

# %%
# url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/car.jpg?download=true"
# image = Image.open(requests.get(url, stream=True).raw)
image = plt.imread("base_paligemma/sample_img.png")
plt.imshow(image)
plt.show()

# %%

model = PaliGemmaForConditionalGeneration.from_pretrained(
    model_id,
    torch_dtype=dtype,
    device_map=device,
    revision="bfloat16",
).eval()
processor = AutoProcessor.from_pretrained(model_id)

prompt = "caption en Is there something wrong with this person ?"
model_inputs = processor(text=prompt, images=image, return_tensors="pt").to(model.device)
input_len = model_inputs["input_ids"].shape[-1]

with torch.inference_mode():
    generation = model.generate(**model_inputs, max_new_tokens=100, do_sample=False)
    generation = generation[0][input_len:]
    decoded = processor.decode(generation, skip_special_tokens=True)
    print(decoded)

# %%
# might have to clean cache before running
del model
gc.collect()
torch.cuda.empty_cache()
gc.collect()
torch.cuda.synchronize()
torch.cuda.ipc_collect()
torch.cuda.reset_max_memory_allocated()
