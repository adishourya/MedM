import datetime
import random
import json
import pandas as pd
from textwrap import wrap
import os
import torch
from transformers import AutoProcessor, PaliGemmaForConditionalGeneration, BitsAndBytesConfig
from PIL import Image
import matplotlib.pyplot as plt
import matplotx

plt.style.use(matplotx.styles.pacoty)

# Paths
checkpoint_dir = "../results__max256__2809-072523/checkpoint-4612"
image_dir = '../../../../datasets/MEDPIX-ClinQA/'

# Load the processor and configure int8 quantization for the model
model_id = "google/paligemma-3b-mix-224"
processor = AutoProcessor.from_pretrained(model_id)

# Enable int8 quantization for faster inference
bnb_config = BitsAndBytesConfig(
    load_in_8bit=True,
)

# Load the model with int8 precision
model = PaliGemmaForConditionalGeneration.from_pretrained(
    checkpoint_dir,
    quantization_config=bnb_config,
    device_map="auto"
).eval()

# Function for inference
def predict(image_path, question):
    # Load and preprocess the image
    image = Image.open(image_path).convert('RGB')

    # Preprocess the inputs (image and question) using the processor
    model_inputs = processor(
        text="answer en " + question,
        images=image,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )

    model_inputs = {k: v.to(model.device) for k, v in model_inputs.items()}

    # Use top_p sampling for generating diverse and longer outputs
    generated_ids = model.generate(
        input_ids=model_inputs["input_ids"],
        pixel_values=model_inputs["pixel_values"],
        attention_mask=model_inputs["attention_mask"],
        max_new_tokens=256,  # Allow the model to generate long answers
        do_sample=True,  # Enable sampling instead of greedy decoding
    )

    # Decode the generated predictions back to text
    generated_text = processor.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    return generated_text

# Example usage for inference
questions = ["what abnormalities does the patient have?"]

with open("./image_caption_pairs.json") as f:
    captions = json.load(f)

# captions is 1 big list
sample_df = pd.read_csv("./sample.csv")
sample_images = sample_df["image_id"].unique()
print(sample_images, len(sample_images))

description = dict()
for item in captions:
    img_name = "sample/" + item["image"] + ".png"
    if img_name in sample_images:
        description[img_name] = str(item)

# Ensure the 'figures' directory exists
output_dir = "./figures" + checkpoint_dir
os.makedirs(output_dir, exist_ok=True)

for k, v in description.items():
    question = random.choice(questions)
    predicted_answer = predict(image_dir + k, question)
    
    # Create subplots: 1 row, 3 columns
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 10))
    
    # Display the image on the middle subplot
    img = plt.imread(image_dir + k)
    ax2.imshow(img)
    ax2.axis('off')  # Turn off axis for the image

    # Add caption text in a box on the right subplot
    caption_text = v
    
    # Wrap the text to fit in the box
    wrapped_caption = "\n".join(wrap(caption_text, width=80))  # Adjust width as needed
    
    ax3.text(0.05, 0.5, wrapped_caption, fontsize=10, 
             va='center', ha='left',
             linespacing=1.5,
             bbox=dict(facecolor='#f0f0f0', edgecolor='black', boxstyle='round,pad=0.5'))
    ax3.axis('off')

    # Add the predicted answer on the left subplot
    wrapped_prediction = "\n".join(wrap(predicted_answer, width=40))
    
    ax1.text(0.05, 0.5, wrapped_prediction, fontsize=10, 
             va='center', ha='left',
             linespacing=1.5,
             bbox=dict(facecolor='#e0f7fa', edgecolor='black', boxstyle='round,pad=0.5'))
    ax1.axis('off')  # Hide axis for prediction text box
    
    # Adjust layout
    plt.tight_layout()

    # Save the figure
    plt.savefig(os.path.join(output_dir, k.replace("/", "_")), dpi=300)

    # Show the plot
    # plt.show()
    print(predicted_answer)
    # break
