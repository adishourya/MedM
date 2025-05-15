[![Typing SVG](https://readme-typing-svg.herokuapp.com?font=Space+Mono&size=50&duration=1500&color=57a773&center=true&vCenter=true&multiline=true&width=1335&height=300&lines=Adapting+Lightweight+Vision+Language+Model;for+Radiological+Visual+Question+Answering)](https://git.io/typing-svg)

# Data Explorer (Sample)
Please Find the prompts to generate QA Pairs in the submitted article.
we present a sample here
<img width="824" alt="image" src="https://github.com/user-attachments/assets/1c4762f1-c058-42d7-9b9f-5aebc8bcd88f" />






## Code Navigation

### First Stage FineTuning
![Stage1 FineTuning](./assets/Stage1.png)


```py
...
model = PaliGemmaForConditionalGeneration.from_pretrained(
    model_id,
    device_map="auto",
    quantization_config=quantization_config,
)

# Freeze all parameters besides the projection layer
for param in model.parameters():
    param.requires_grad = False

for name, param in model.named_parameters():
    if name.startswith("multi_modal_projector.linear"):
        param.requires_grad = True
...
```



### Second Stage FineTuning
![Stage2 FineTuning](./assets/Stage2.png)
Find the code for performing First stage finetuning here : 

```py
...
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

...
```


# Diagnostic Tool 
You can find our submodule at:
https://anonymous.4open.science/r/lvlm-interpret-4A27
