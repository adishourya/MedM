import torch
from transformers import AutoProcessor, PaliGemmaForConditionalGeneration
from pprint import pprint

def get_model_size(model):
    param_size = 0
    layer_info = {}
    
    for name, param in model.named_parameters():
        # Calculate parameter size in bytes: number of elements * size of element (in bytes)
        param_bytes = param.numel() * param.element_size()
        param_size += param_bytes
        layer_info[name] = {
            "shape": param.shape,  # Shape of the layer
            "size_gb": param_bytes / (1024 ** 3)  # Convert to gigabytes
        }
    
    total_size_gb = param_size / (1024 ** 3)  # Total size in gigabytes
    return layer_info, total_size_gb

# Huggingface implementation of PaliGemma
model_id = "google/paligemma-3b-mix-224"
dtype = torch.bfloat16

# Initialize processor and model
processor = AutoProcessor.from_pretrained(model_id)
tokenizer = processor.tokenizer  # Use the processor's tokenizer for answers

# Load the model in bfloat16 precision with memory limits
model = PaliGemmaForConditionalGeneration.from_pretrained(
    model_id,
    torch_dtype=dtype,
    device_map="auto",
).train()

layer_info, total_size_gb = get_model_size(model)

# Print layer info (name, shape, size in GB)
print("Layer Information (Shape and Size in GB):")
pprint(layer_info)

# Print total model size
print(f"Total model size: {total_size_gb:.4f} GB")
