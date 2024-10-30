import ollama
from tqdm import tqdm
import time
import json
import os

# Function to call ollama and generate Q&A pairs
def generate_qa_with_ollama(caption):
    # Construct the prompt for ollama
    prompt = f"""
    Based on the following medical image captions generate appropriate question
    The medical image caption is in square bracket; treat this as the answer to generate a single question:
    [{caption}]
    Please! Don't use any filler text at the start or at the end of your response.
    """

    # Now with the following radiology caption, improve clarity, remove temporal/historical context, and focus on key findings. This would be used to build a radiology based visual question answer dataset
    response = ollama.chat(model='llama3.1', messages=[
        {
            'role': 'user',
            'content': prompt
        }
    ])

    # Return the generated text from the response
    return response['message']['content'].strip()

# Load the dataset from a text file
file_path = "./train_captions_continue.txt"

# Set up the output JSON file, or create it if it doesn't exist
output_file = file_path.split(".txt")[0]+"_qa_pairs_output.json"

# Function to append data to a JSON file
def append_to_json(new_entry, output_file):
    # Check if the file exists
    if os.path.exists(output_file):
        # Load existing data
        with open(output_file, 'r') as f:
            data = json.load(f)
    else:
        # Start with an empty list if the file doesn't exist
        data = []

    # Append the new entry
    data.append(new_entry)

    # Write the updated data back to the file
    with open(output_file, 'w') as f:
        json.dump(data, f, indent=4)

# Read each line from the text file and process it
with open(file_path, 'r') as file:
    lines = file.readlines()

for line in tqdm(lines, total=len(lines)):
    # Split the line by space, assuming the first part is image_id and the rest is the caption
    parts = line.strip().split(maxsplit=1)
    if len(parts) < 2:
        print(f"Skipping malformed line: {line}")
        continue

    image_id, caption = parts[0], parts[1]

    try:
        # Generate Q&A pairs
        qa_pairs = generate_qa_with_ollama(caption)

        # Create a new entry for the current line
        new_entry = {
            "image_id": image_id,
            "original caption": caption,
            "qa_pair": qa_pairs
        }

        # Append the new entry to the JSON file
        append_to_json(new_entry, output_file)

    except Exception as e:
        print(f"Error processing image {image_id}: {e}")

print(f"Q&A generation completed and saved to {output_file}.")
