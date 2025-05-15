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
```



### Second Stage FineTuning
![Stage2 FineTuning](./assets/Stage2.png)
Find the code for performing First stage finetuning here : 

```py
def bar():
    pass
```


# Diagnostic Tool 
You can find our submodule at:
https://anonymous.4open.science/r/lvlm-interpret-4A27
