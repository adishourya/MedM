import torch
import pandas as pd
from datasets import load_dataset
from transformers import AutoProcessor, PaliGemmaForConditionalGeneration, BitsAndBytesConfig
from peft import PeftModel, PeftConfig
from accelerate import infer_auto_device_map

# ┌──────────────────┐
# │ Load Quantized Model │
# └──────────────────┘

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
)

adapter_model_id = "adishourya/results__fullrun__0310-134147"
peft_config = PeftConfig.from_pretrained(adapter_model_id)

model_id = peft_config.base_model_name_or_path
base_model = PaliGemmaForConditionalGeneration.from_pretrained(
    model_id,
    quantization_config=quantization_config,
    device_map="auto",
)

device_map = infer_auto_device_map(
    base_model,
    max_memory={0: "2GB", "cpu": "10GB"},
)

model = PeftModel.from_pretrained(
    base_model,
    adapter_model_id,
)

processor = AutoProcessor.from_pretrained(model_id)
model.eval()

# ┌───────┐
# │ Dataset│
# └───────┘

# Load the full test dataset
test_dataset = load_dataset("adishourya/MEDPIX-ShortQA", split="Test")

# ┌─────────────────────┐
# │ Inference Function  │
# └─────────────────────┘

def generate_answer(batch):
    if isinstance(batch, dict):
        batch = [{key: batch[key][i] for key in batch} for i in range(len(batch["image_id"]))]
    
    images = [item["image_id"].convert("RGB") for item in batch]
    questions = ["answer " + item["question"] for item in batch]
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    inputs = processor(
        text=questions,
        images=images,
        return_tensors="pt",
        padding="longest"
    ).to(device)

    with torch.no_grad():
        generated_ids = model.generate(
            inputs["input_ids"], 
            max_new_tokens=100,
        )
    
    generated_answers = processor.batch_decode(generated_ids, skip_special_tokens=True)
    return generated_answers

# ┌────────────────────────┐
# │ Perform Inference      │
# └────────────────────────┘

batch_size = 1
output_data = []

for i in range(0, len(test_dataset), batch_size):
    batch = test_dataset[i:i + batch_size]
    
    if isinstance(batch, dict):
        batch = [{key: batch[key][i] for key in batch} for i in range(len(batch["image_id"]))]
    
    generated_answers = generate_answer(batch)
    
    for image_id, question, generated_answer, label in zip(
        [item["image_id"] for item in batch],
        [item["question"] for item in batch], 
        generated_answers, 
        [item["answer"] for item in batch]
    ):
        output_data.append({
            "Image ID": image_id,  # You can modify this if you want to save image paths or metadata
            "Question": question,
            "Generated Answer": generated_answer,
            "Label": label
        })
        print(generated_answer)
        print("-"*25)
        print(label)
        print("="*50)

# ┌─────────────────────┐
# │ Save to Excel       │
# └─────────────────────┘

# Create a DataFrame from the output data
df = pd.DataFrame(output_data)

# Save DataFrame to an Excel file
output_file = "inference_results.xlsx"
df.to_excel(output_file, index=False)

print(f"Results saved to {output_file}")
