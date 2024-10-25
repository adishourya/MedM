import pandas as pd
import ollama
from tqdm import tqdm
import time
import json
import os

# Function to call ollama and generate Q&A pairs
def generate_qa_with_ollama(caption, subject):
    # Construct the prompt for ollama
    prompt = f"""
    Based on the following medical image caption and case information,

    Caption Information:
    {caption}
    Plane and Location Information of the Image:
    {subject}
    Generate question-answer pairs [exhaustive of the information given].
    Assume I am going to use this to train a Visual Question Answering model for a medical dataset.
    Keep the QA pairs such that they can only be answered when the image is in context.
    Do not ask questions about measurements (avoid numericals in question answer pairs) , history of patient , or if information is unavailable.
    Question on 1 line and answer on the new line. Please! Don't use any filler text.

    """
    # prompt from paper

    # Ask 5 questions about the content and generate four options for each question. The questions should be 
    # answerable with the information provided in the caption, and the four options should include one correct
    # and three incorrect options, with the position of the correct option randomized. The output should use
    # the following template: i:‘the question index’ question:‘the generate question’ choice: ‘A:option content
    # B:option content C:option content D:option content’ answer: The correct option(A/B/C/D).

    # Use ollama.chat to generate the response
    response = ollama.chat(model='llama3.1', messages=[
        {
            'role': 'user',
            'content': prompt
        }
    ])

    # Return the generated text from the response
    return response['message']['content'].strip()

# Load the dataset
df = pd.read_excel("./dataset_description.xlsx")

# Set up the output JSON file, or create it if it doesn't exist
output_file = "qa_pairs_output.json"

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

# Loop over each row in the dataframe and generate Q&A pairs
for index, row in tqdm(df.iterrows(), total=len(df)):
    caption = row['caption']
    subject = row['subject']            # Adjust the column name if necessary
    image_id = row["Unnamed: 0"]
    split = row["split"]

    try:
        # Generate Q&A pairs
        qa_pairs = generate_qa_with_ollama(caption,subject)

        # Add a delay to avoid overloading the API
        # time.sleep(2)

        # Create a new entry for the current row
        new_entry = {
            "image_id":image_id,
            "split":split,
            'caption': caption,
            'subject': subject,
            'qa_pairs': qa_pairs
        }

        # Append the new entry to the JSON file
        append_to_json(new_entry, output_file)

    except Exception as e:
        print(f"Error processing row {index}: {e}")

print(f"Q&A generation completed and saved to {output_file}.")
