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
# out
MAX_LENGTH = 256
GENERATION_MAX_LENGTH = 100  # this can be learnt so not necessary rn
# main settings
epochs = 2

# ┌───────┐
# │dataset│
# └───────┘
train_dataset = load_dataset("adishourya/MEDPIX-ShortQA",split="Train")
valid_dataset = load_dataset("adishourya/MEDPIX-ShortQA",split="Valid")

if "quickrun" in run_name:
    epochs = 10
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

# low rank trainable settings
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
    # print(batch)
    image = [
        idx["image_id"].convert("RGB") for idx in batch
    ]  # takes in an rgb image [just to be sure]
    question = ["answer " + idx["question"] for idx in batch]
    label = [idx["answer"] for idx in batch]
    # print("hi")

    tokens_out = processor(
        text=question,
        images=image,
        suffix=label,
        return_tensors="pt",  # this was originally trained with jax
        padding="longest",  # pad to the longest answer which is about 80 words
    )

    # print(tokens_out)
    return tokens_out


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
    gradient_accumulation_steps=4,
    warmup_steps=2,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    save_steps=0.5,
    eval_steps=0.25,
    report_to=["tensorboard"],
    output_dir=RESULTS_DIR,
    logging_dir=LOGGING_DIR,
    logging_steps=1,
    push_to_hub=True,
    fp16=True,
    remove_unused_columns=False,
    dataloader_pin_memory=False,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=valid_dataset,
    data_collator=collate_fn,
)

trainer.train()
trainer.push_to_hub("adishourya/medpix_pg" + run_name)
