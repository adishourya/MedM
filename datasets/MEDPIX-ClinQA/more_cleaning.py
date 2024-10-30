import ollama
import pandas as pd
import time
from tqdm import tqdm

def clean_qa(row):
    prompt = f"""
    IF there are answers that talk about the dimensions that modify the answer so that it avoids talking about it, modify answers that talk about demography or geography, remove rows that talk about codes like Acr, icd and so on, remove the rows that talk about history, if the answer is 60 words or more then summarize it to the essential bits. Keep in mind that this dataset will be used for finetuning a visual question answer model; and your output should be csv.
    """

