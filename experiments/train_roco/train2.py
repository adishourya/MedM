from nltk.metrics.scores import shuffle
from transformers.utils import quantization_config
import wandb
import nltk
# from nltk import edit_distance # nltk also has edit_distance
import evaluate
import editdistance
import torch
from torch.nn.modules import padding
from torch.utils.data import Dataset
from typing import Any, List, Dict
from datasets import load_dataset
from transformers import (
    AutoModel,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    BitsAndBytesConfig,
    AutoProcessor,
    PaliGemmaForConditionalGeneration,
)
from peft import get_peft_model, LoraConfig
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import einops

# for image handling
import PIL.Image as PILImage
import io
import datetime

model_id = "google/paligemma-3b-mix-224"
run_name = model_id.replace("/","_") + str(datetime.datetime.now().strftime("%d%m-%H%M%S"))
huggingface_card = "adishourya/" + run_name
# start with default
input_max_length: int

wandb_project = model_id.replace("/","_")
wandb_name = run_name

# dataset
train_dataset = (
    load_dataset("adishourya/ROCO-QA", split="Vaild").shuffle(seed=1).select(range(10))
)  # pyright: ignore
valid_dataset = (
    load_dataset("adishourya/ROCO-QA", split="Test").shuffle(seed=1).select(range(5))
)
# processor for multimodals inputs and tokenizer for text based output
processor = AutoProcessor.from_pretrained(model_id)
tokenizer = AutoTokenizer.from_pretrained(model_id)
max_length: int

#  ______________________________________________________________
# / # In [4]: dataset                                            \
# | # Out[4]:                                                    |
# | # Dataset({                                                  |
# | #     features: ['image_id', 'image', 'question', 'answer'], |
# | #     num_rows: 8175                                         |
# \ # })                                                         /
#  --------------------------------------------------------------

# Prefix for question input
prefix = "answer en "

def preprocess_function(batch):
    """
    Processes a batch of data for input into the model.
    Args:
        batch: Dictionary with keys 'image', 'question', and 'answer'
    Returns:
        Dictionary with tokenized input data
    """
    def show(inputs,labels):
        """
        shows one batch of tokenization every batch
        """
        print("------------------")
        print(processor.batch_decode(inputs["input_ids"][0]))
        print("------")
        print(processor.batch_decode(labels["input_ids"][0]))

    questions = [prefix + q for q in batch["question"]]  # Add prefix to each question
    answers = batch["answer"]
    images = [PILImage.open(io.BytesIO(i["bytes"])).convert("RGB") for i in batch["image"]]
    
    # Process images and questions for model input
    inputs = processor(images=images, text=questions, padding="longest", return_tensors="pt")
    labels = tokenizer(answers,padding="longest",return_tensors="pt")
    
    
    inputs["labels"] = labels["input_ids"]
    show(inputs,labels)
    return inputs

# Tokenize and map dataset
train_dataset = train_dataset.map(preprocess_function, batched=True)

# evaluation
nltk.download("punkt",quiet=True)
metric = evaluate.load("rouge")

# Load evaluation metrics
rouge_metric = evaluate.load("rouge")
# cider_metric = evaluate.load("cider")  
bleu_metric = evaluate.load("bleu") 

def compute_metrics(eval_preds):
    preds, labels = eval_preds

    # Decode predictions and labels
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Prepare text for rouge and CIDEr
    decoded_preds = ["\n".join(nltk.sent_tokenize(pred.strip())) for pred in decoded_preds]
    decoded_labels = ["\n".join(nltk.sent_tokenize(label.strip())) for label in decoded_labels]

    # Compute ROUGE
    rouge_result = rouge_metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
    
    # Compute BLEU
    bleu_result = bleu_metric.compute(predictions=decoded_preds, references=[[label] for label in decoded_labels])

    # Compute Edit Distance
    edit_distances = [
        editdistance.eval(pred, label) for pred, label in zip(decoded_preds, decoded_labels)
    ]
    avg_edit_distance = np.mean(edit_distances)

    # Format results
    results = {
        "rouge1": rouge_result["rouge1"],
        "rouge2": rouge_result["rouge2"],
        "rougeL": rouge_result["rougeL"],
        "bleu": bleu_result["bleu"],
        "avg_edit_distance": avg_edit_distance,
    }
    return results

# get quantized model -> get model with lora adapters
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

model = PaliGemmaForConditionalGeneration.from_pretrained(
    model_id,
    device_map="auto",
    quantization_config = quantization_config
)

lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=[
        "q_proj",
        "o_proj",
        "k_proj",
        "v_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters() # about 0.38% -> 2.93M


# Initialize W&B
wandb.init(project=wandb_project, name=wandb_name)

# Training arguments
training_args = Seq2SeqTrainingArguments(
    output_dir="./results",
    evaluation_strategy="epochs",
    logging_strategy="steps",
    logging_steps=50,
    save_strategy="epochs",
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    learning_rate=1e-5,
    num_train_epochs=5,
    predict_with_generate=True,
    generation_max_length=128,
    fp16=True,
    report_to="wandb",  # Reports to W&B
    run_name=wandb_name,
)

# Data collator for sequence-to-sequence tasks
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, padding="longest")

# Initialize Trainer
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=valid_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

# Train the model
trainer.train()
