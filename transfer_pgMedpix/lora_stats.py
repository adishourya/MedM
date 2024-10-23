import torch
from transformers import AutoProcessor, PaliGemmaForConditionalGeneration
from peft.tuners.lora import LoraConfig
from peft.mapping import get_peft_model
import bitsandbytes as bnb  # for 4-bit quantization

# Load the processor and tokenizer
model_id = "google/paligemma-3b-mix-224"
processor = AutoProcessor.from_pretrained(model_id)
tokenizer = processor.tokenizer

# Load the model in 4-bit quantized precision using bitsandbytes
model = PaliGemmaForConditionalGeneration.from_pretrained(
    model_id,
    load_in_4bit=True,  # Quantization to 4-bit
    device_map="auto",
    torch_dtype=torch.float16,  # Use float16 for further memory savings
)

# Set up LoRA configuration for fine-tuning the self-attention layers
lora_config = LoraConfig(
    r=8,  # Low-rank dimension
    lora_alpha=32,  # Scaling factor for LoRA layers
    target_modules=["self_attn.q_proj", "self_attn.k_proj" , "self_attn.v_proj"],  # Only apply LoRA to the self-attention layers
    lora_dropout=0.05,  # Dropout for LoRA layers
    bias="none"  # Do not add any bias terms
)

# Apply LoRA to the model
model = get_peft_model(model, lora_config)

# Freeze all layers except those with 'language' and 'self_attn' in their name
total_trainable_params = 0
for name, param in model.named_parameters():
    # Only unfreeze floating-point parameters
    if 'language' in name and "self_attn" in name:
        if param.dtype in [torch.float16, torch.float32, torch.bfloat16]:
            param.requires_grad = True
            total_trainable_params += param.numel()
            print(f"{name:90} Frozen = False")
        else:
            print(f"{name:90} (Skipped non-float parameter)")
    else:
        param.requires_grad = False
        print(f"{name:90} Frozen = True")

print(f"Total Trainable Parameters: {total_trainable_params}")


def get_quantized_model_size(model):
    param_size = 0
    layer_sizes = {}
    
    for name, param in model.named_parameters():
        # Calculate parameter size in bytes based on quantized dtype (4 bits or 0.5 bytes for 4-bit quantization)
        param_bytes = param.numel() * param.element_size()  # element_size() gives size in bytes
        param_size += param_bytes
        layer_sizes[name] = param_bytes / (1024 ** 3)  # Convert to gigabytes
    
    total_size_gb = param_size / (1024 ** 3)  # Total size in gigabytes
    return layer_sizes, total_size_gb

# Get the size of the quantized model
layer_sizes, total_size_gb = get_quantized_model_size(model)

# Print layer-wise sizes and total size
print("Layer-wise sizes (in GB):")
for layer_name, size in layer_sizes.items():
    print(f"{layer_name}: {size:.4f} GB")

print(f"\nTotal quantized model size: {total_size_gb:.4f} GB")
