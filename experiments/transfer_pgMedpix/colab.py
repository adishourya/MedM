from huggingface_hub import login
import numpy as np
import evaluate
import torch
from datasets import load_dataset
from transformers import (
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
    AutoProcessor,
    PaliGemmaForConditionalGeneration,
)
from peft import get_peft_model, LoraConfig
import datetime

# ┌──────────────────┐
# │run configurations│
# └──────────────────┘

# directory
LOGGING_DIR = "./logs"
RESULTS_DIR = "./RESULTS"
experiment_name = "__quickrun__"
run_name = experiment_name + str(datetime.datetime.now().strftime("%d%m-%H%M%S"))
checkpoint = ""  # for continuing training
model_id = "google/paligemma-3b-mix-224"
LOGGING_DIR = f"./logs/run{run_name}"
RESULTS_DIR = f"./RESULTS/results{run_name}"
# out [didn't use it.. testing defaults]
MAX_LENGTH = 512
GENERATION_MAX_LENGTH = 100  # this can be learned so not necessary rn
# main settings
epochs = 10

# ┌───────┐
# │dataset│
# └───────┘
train_dataset = load_dataset("adishourya/MEDPIX-ShortQA", split="Train")
valid_dataset = load_dataset("adishourya/MEDPIX-ShortQA", split="Valid")

if "quickrun" in run_name:
    epochs = 3
    train_dataset = train_dataset.select(range(10))
    valid_dataset = train_dataset.select(range(10))

print(f"Train shape {train_dataset.shape}, Val shape {valid_dataset.shape}")

# ┌─────┐
# │Model│
# └─────┘
processor = AutoProcessor.from_pretrained(model_id)
tokenizer = processor.tokenizer

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

# get quantized model
model = PaliGemmaForConditionalGeneration.from_pretrained(
    model_id, device_map="auto", quantization_config=quantization_config
)

# low-rank trainable settings
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
# see the trainable params
model.print_trainable_parameters()

# ┌─────────────────┐
# │processing inputs│
# └─────────────────┘
def collate_fn(batch):
    image = [idx["image_id"].convert("RGB") for idx in batch]
    question = ["answer " + idx["question"] for idx in batch]
    label = [idx["answer"] for idx in batch]

    # Pad inputs and labels to the longest sequence in the batch
    tokens_out = processor(
        text=question,
        images=image,
        suffix=label,
        return_tensors="pt",
        padding="longest",  # Ensure longest padding for consistency
        max_length=MAX_LENGTH,  # Enforce max length to avoid overflow
        truncation=True
    )

    # Replace padding in labels with -100 (ignore index for cross-entropy loss)
    if 'labels' in tokens_out and tokens_out['labels'] is not None:
        tokens_out['labels'][tokens_out['labels'] == processor.tokenizer.pad_token_id] = -100

    return tokens_out

# ┌──────────────────┐
# │compute metrics fix│
# └──────────────────┘

def compute_metrics(eval_preds):
    metric = evaluate.load("glue", "mnli")  # Ensure you're using the correct task from GLUE
    logits, labels = eval_preds

    # Ensure logits and labels are both of the same length along the first dimension
    max_len = min(logits.shape[1], labels.shape[1])
    logits = logits[:, :max_len]
    labels = labels[:, :max_len]

    predictions = np.argmax(logits, axis=-1)

    # Mask out the padding tokens (assuming pad token id is -100)
    non_pad_mask = labels != -100
    labels = labels[non_pad_mask]
    predictions = predictions[non_pad_mask]

    return metric.compute(predictions=predictions, references=labels)

# ┌──────────────────┐
# │run configurations│
# └──────────────────┘

training_args = TrainingArguments(
    optim="adamw_hf",
    num_train_epochs=epochs,
    learning_rate=1e-5,
    lr_scheduler_type="constant",
    label_smoothing_factor=0,
    weight_decay=0.0,
    gradient_accumulation_steps=8,
    warmup_steps=2,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    save_steps=600,
    eval_strategy="epoch",
    report_to=["tensorboard"],
    output_dir=RESULTS_DIR,
    logging_dir=LOGGING_DIR,
    logging_steps=5,
    push_to_hub=True,
    fp16=True,
    remove_unused_columns=False,
    dataloader_pin_memory=False,
    do_eval=True,
)

# Trainer initialization
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=valid_dataset,
    data_collator=collate_fn,  # Collator handles padding
    compute_metrics=compute_metrics
)

# Training
trainer.train()

# Push to hub if it's a full run
if "full" in run_name:
    trainer.push_to_hub("adishourya/medpix_pg_" + run_name)
