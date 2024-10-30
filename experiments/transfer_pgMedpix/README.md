# Performs paligemma transfer for MEDPIX CLINQA

## settings
* allow multiple images per question ?

### Hyperparams in decreasing order of importance
Recommended Initial attempt in Bold:
* Resolution (i.e. checkpoint): **224**, 448, 896.
* Epochs: **1, 3, 10**, 30, 100.
* Learning-rate: 3e-5, **1e-5**, 3e-6.
* Label-smoothing: **0.0**, 0.1, 0.3.
* Dropout in the LLM: **0.0**, 0.1, 0.3.
* Weight decay: **0.0** or 0.1 × learning-rate.
* Freeze ViT: **false**, true.

FineTuning [Not VQA] : [notebook][https://colab.research.google.com/github/google-research/big_vision/blob/main/big_vision/configs/proj/paligemma/finetune_paligemma.ipynb]

┌───────────────────────────────────────┐
│vision towers are supposed to be frozen│
└───────────────────────────────────────┘
```
vision_tower.vision_model.embeddings.patch_embedding.weight            Frozen  :  True
vision_tower.vision_model.embeddings.patch_embedding.bias              Frozen  :  True
vision_tower.vision_model.embeddings.position_embedding.weight         Frozen  :  True
vision_tower.vision_model.encoder.layers.0.self_attn.k_proj.weight     Frozen  :  True
vision_tower.vision_model.encoder.layers.0.self_attn.k_proj.bias       Frozen  :  True
vision_tower.vision_model.encoder.layers.0.self_attn.v_proj.weight     Frozen  :  True
vision_tower.vision_model.encoder.layers.0.self_attn.v_proj.bias       Frozen  :  True
vision_tower.vision_model.encoder.layers.0.self_attn.q_proj.weight     Frozen  :  True
vision_tower.vision_model.encoder.layers.0.self_attn.q_proj.bias       Frozen  :  True
vision_tower.vision_model.encoder.layers.0.self_attn.out_proj.weight   Frozen  :  True
vision_tower.vision_model.encoder.layers.0.self_attn.out_proj.bias     Frozen  :  True
vision_tower.vision_model.encoder.layers.0.layer_norm1.weight          Frozen  :  True
vision_tower.vision_model.encoder.layers.0.layer_norm1.bias            Frozen  :  True
vision_tower.vision_model.encoder.layers.0.mlp.fc1.weight              Frozen  :  True
vision_tower.vision_model.encoder.layers.0.mlp.fc1.bias                Frozen  :  True
vision_tower.vision_model.encoder.layers.0.mlp.fc2.weight              Frozen  :  True
vision_tower.vision_model.encoder.layers.0.mlp.fc2.bias                Frozen  :  True
vision_tower.vision_model.encoder.layers.0.layer_norm2.weight          Frozen  :  True
vision_tower.vision_model.encoder.layers.0.layer_norm2.bias            Frozen  :  True
vision_tower.vision_model.encoder.layers.1.self_attn.k_proj.weight     Frozen  :  True
vision_tower.vision_model.encoder.layers.1.self_attn.k_proj.bias       Frozen  :  True
vision_tower.vision_model.encoder.layers.1.self_attn.v_proj.weight     Frozen  :  True
vision_tower.vision_model.encoder.layers.1.self_attn.v_proj.bias       Frozen  :  True
vision_tower.vision_model.encoder.layers.1.self_attn.q_proj.weight     Frozen  :  True
vision_tower.vision_model.encoder.layers.1.self_attn.q_proj.bias       Frozen  :  True
vision_tower.vision_model.encoder.layers.1.self_attn.out_proj.weight   Frozen  :  True
vision_tower.vision_model.encoder.layers.1.self_attn.out_proj.bias     Frozen  :  True
vision_tower.vision_model.encoder.layers.1.layer_norm1.weight          Frozen  :  True
vision_tower.vision_model.encoder.layers.1.layer_norm1.bias            Frozen  :  True
vision_tower.vision_model.encoder.layers.1.mlp.fc1.weight              Frozen  :  True
vision_tower.vision_model.encoder.layers.1.mlp.fc1.bias                Frozen  :  True
vision_tower.vision_model.encoder.layers.1.mlp.fc2.weight              Frozen  :  True
vision_tower.vision_model.encoder.layers.1.mlp.fc2.bias                Frozen  :  True
vision_tower.vision_model.encoder.layers.1.layer_norm2.weight          Frozen  :  True
vision_tower.vision_model.encoder.layers.1.layer_norm2.bias            Frozen  :  True
vision_tower.vision_model.encoder.layers.2.self_attn.k_proj.weight     Frozen  :  True
vision_tower.vision_model.encoder.layers.2.self_attn.k_proj.bias       Frozen  :  True
vision_tower.vision_model.encoder.layers.2.self_attn.v_proj.weight     Frozen  :  True
vision_tower.vision_model.encoder.layers.2.self_attn.v_proj.bias       Frozen  :  True
vision_tower.vision_model.encoder.layers.2.self_attn.q_proj.weight     Frozen  :  True
vision_tower.vision_model.encoder.layers.2.self_attn.q_proj.bias       Frozen  :  True
vision_tower.vision_model.encoder.layers.2.self_attn.out_proj.weight   Frozen  :  True
vision_tower.vision_model.encoder.layers.2.self_attn.out_proj.bias     Frozen  :  True
vision_tower.vision_model.encoder.layers.2.layer_norm1.weight          Frozen  :  True
vision_tower.vision_model.encoder.layers.2.layer_norm1.bias            Frozen  :  True
vision_tower.vision_model.encoder.layers.2.mlp.fc1.weight              Frozen  :  True
vision_tower.vision_model.encoder.layers.2.mlp.fc1.bias                Frozen  :  True
vision_tower.vision_model.encoder.layers.2.mlp.fc2.weight              Frozen  :  True
vision_tower.vision_model.encoder.layers.2.mlp.fc2.bias                Frozen  :  True
vision_tower.vision_model.encoder.layers.2.layer_norm2.weight          Frozen  :  True
vision_tower.vision_model.encoder.layers.2.layer_norm2.bias            Frozen  :  True
vision_tower.vision_model.encoder.layers.3.self_attn.k_proj.weight     Frozen  :  True
vision_tower.vision_model.encoder.layers.3.self_attn.k_proj.bias       Frozen  :  True
vision_tower.vision_model.encoder.layers.3.self_attn.v_proj.weight     Frozen  :  True
vision_tower.vision_model.encoder.layers.3.self_attn.v_proj.bias       Frozen  :  True
vision_tower.vision_model.encoder.layers.3.self_attn.q_proj.weight     Frozen  :  True
vision_tower.vision_model.encoder.layers.3.self_attn.q_proj.bias       Frozen  :  True
vision_tower.vision_model.encoder.layers.3.self_attn.out_proj.weight   Frozen  :  True
vision_tower.vision_model.encoder.layers.3.self_attn.out_proj.bias     Frozen  :  True
vision_tower.vision_model.encoder.layers.3.layer_norm1.weight          Frozen  :  True
vision_tower.vision_model.encoder.layers.3.layer_norm1.bias            Frozen  :  True
vision_tower.vision_model.encoder.layers.3.mlp.fc1.weight              Frozen  :  True
vision_tower.vision_model.encoder.layers.3.mlp.fc1.bias                Frozen  :  True
vision_tower.vision_model.encoder.layers.3.mlp.fc2.weight              Frozen  :  True
vision_tower.vision_model.encoder.layers.3.mlp.fc2.bias                Frozen  :  True
vision_tower.vision_model.encoder.layers.3.layer_norm2.weight          Frozen  :  True
vision_tower.vision_model.encoder.layers.3.layer_norm2.bias            Frozen  :  True
vision_tower.vision_model.encoder.layers.4.self_attn.k_proj.weight     Frozen  :  True
vision_tower.vision_model.encoder.layers.4.self_attn.k_proj.bias       Frozen  :  True
vision_tower.vision_model.encoder.layers.4.self_attn.v_proj.weight     Frozen  :  True
vision_tower.vision_model.encoder.layers.4.self_attn.v_proj.bias       Frozen  :  True
vision_tower.vision_model.encoder.layers.4.self_attn.q_proj.weight     Frozen  :  True
vision_tower.vision_model.encoder.layers.4.self_attn.q_proj.bias       Frozen  :  True
vision_tower.vision_model.encoder.layers.4.self_attn.out_proj.weight   Frozen  :  True
vision_tower.vision_model.encoder.layers.4.self_attn.out_proj.bias     Frozen  :  True
vision_tower.vision_model.encoder.layers.4.layer_norm1.weight          Frozen  :  True
vision_tower.vision_model.encoder.layers.4.layer_norm1.bias            Frozen  :  True
vision_tower.vision_model.encoder.layers.4.mlp.fc1.weight              Frozen  :  True
vision_tower.vision_model.encoder.layers.4.mlp.fc1.bias                Frozen  :  True
vision_tower.vision_model.encoder.layers.4.mlp.fc2.weight              Frozen  :  True
vision_tower.vision_model.encoder.layers.4.mlp.fc2.bias                Frozen  :  True
vision_tower.vision_model.encoder.layers.4.layer_norm2.weight          Frozen  :  True
vision_tower.vision_model.encoder.layers.4.layer_norm2.bias            Frozen  :  True
vision_tower.vision_model.encoder.layers.5.self_attn.k_proj.weight     Frozen  :  True
vision_tower.vision_model.encoder.layers.5.self_attn.k_proj.bias       Frozen  :  True
vision_tower.vision_model.encoder.layers.5.self_attn.v_proj.weight     Frozen  :  True
vision_tower.vision_model.encoder.layers.5.self_attn.v_proj.bias       Frozen  :  True
vision_tower.vision_model.encoder.layers.5.self_attn.q_proj.weight     Frozen  :  True
vision_tower.vision_model.encoder.layers.5.self_attn.q_proj.bias       Frozen  :  True
vision_tower.vision_model.encoder.layers.5.self_attn.out_proj.weight   Frozen  :  True
vision_tower.vision_model.encoder.layers.5.self_attn.out_proj.bias     Frozen  :  True
vision_tower.vision_model.encoder.layers.5.layer_norm1.weight          Frozen  :  True
vision_tower.vision_model.encoder.layers.5.layer_norm1.bias            Frozen  :  True
vision_tower.vision_model.encoder.layers.5.mlp.fc1.weight              Frozen  :  True
vision_tower.vision_model.encoder.layers.5.mlp.fc1.bias                Frozen  :  True
vision_tower.vision_model.encoder.layers.5.mlp.fc2.weight              Frozen  :  True
vision_tower.vision_model.encoder.layers.5.mlp.fc2.bias                Frozen  :  True
vision_tower.vision_model.encoder.layers.5.layer_norm2.weight          Frozen  :  True
vision_tower.vision_model.encoder.layers.5.layer_norm2.bias            Frozen  :  True
vision_tower.vision_model.encoder.layers.6.self_attn.k_proj.weight     Frozen  :  True
vision_tower.vision_model.encoder.layers.6.self_attn.k_proj.bias       Frozen  :  True
vision_tower.vision_model.encoder.layers.6.self_attn.v_proj.weight     Frozen  :  True
vision_tower.vision_model.encoder.layers.6.self_attn.v_proj.bias       Frozen  :  True
vision_tower.vision_model.encoder.layers.6.self_attn.q_proj.weight     Frozen  :  True
vision_tower.vision_model.encoder.layers.6.self_attn.q_proj.bias       Frozen  :  True
vision_tower.vision_model.encoder.layers.6.self_attn.out_proj.weight   Frozen  :  True
vision_tower.vision_model.encoder.layers.6.self_attn.out_proj.bias     Frozen  :  True
vision_tower.vision_model.encoder.layers.6.layer_norm1.weight          Frozen  :  True
vision_tower.vision_model.encoder.layers.6.layer_norm1.bias            Frozen  :  True
vision_tower.vision_model.encoder.layers.6.mlp.fc1.weight              Frozen  :  True
vision_tower.vision_model.encoder.layers.6.mlp.fc1.bias                Frozen  :  True
vision_tower.vision_model.encoder.layers.6.mlp.fc2.weight              Frozen  :  True
vision_tower.vision_model.encoder.layers.6.mlp.fc2.bias                Frozen  :  True
vision_tower.vision_model.encoder.layers.6.layer_norm2.weight          Frozen  :  True
vision_tower.vision_model.encoder.layers.6.layer_norm2.bias            Frozen  :  True
vision_tower.vision_model.encoder.layers.7.self_attn.k_proj.weight     Frozen  :  True
vision_tower.vision_model.encoder.layers.7.self_attn.k_proj.bias       Frozen  :  True
vision_tower.vision_model.encoder.layers.7.self_attn.v_proj.weight     Frozen  :  True
vision_tower.vision_model.encoder.layers.7.self_attn.v_proj.bias       Frozen  :  True
vision_tower.vision_model.encoder.layers.7.self_attn.q_proj.weight     Frozen  :  True
vision_tower.vision_model.encoder.layers.7.self_attn.q_proj.bias       Frozen  :  True
vision_tower.vision_model.encoder.layers.7.self_attn.out_proj.weight   Frozen  :  True
vision_tower.vision_model.encoder.layers.7.self_attn.out_proj.bias     Frozen  :  True
vision_tower.vision_model.encoder.layers.7.layer_norm1.weight          Frozen  :  True
vision_tower.vision_model.encoder.layers.7.layer_norm1.bias            Frozen  :  True
vision_tower.vision_model.encoder.layers.7.mlp.fc1.weight              Frozen  :  True
vision_tower.vision_model.encoder.layers.7.mlp.fc1.bias                Frozen  :  True
vision_tower.vision_model.encoder.layers.7.mlp.fc2.weight              Frozen  :  True
vision_tower.vision_model.encoder.layers.7.mlp.fc2.bias                Frozen  :  True
vision_tower.vision_model.encoder.layers.7.layer_norm2.weight          Frozen  :  True
vision_tower.vision_model.encoder.layers.7.layer_norm2.bias            Frozen  :  True
vision_tower.vision_model.encoder.layers.8.self_attn.k_proj.weight     Frozen  :  True
vision_tower.vision_model.encoder.layers.8.self_attn.k_proj.bias       Frozen  :  True
vision_tower.vision_model.encoder.layers.8.self_attn.v_proj.weight     Frozen  :  True
vision_tower.vision_model.encoder.layers.8.self_attn.v_proj.bias       Frozen  :  True
vision_tower.vision_model.encoder.layers.8.self_attn.q_proj.weight     Frozen  :  True
vision_tower.vision_model.encoder.layers.8.self_attn.q_proj.bias       Frozen  :  True
vision_tower.vision_model.encoder.layers.8.self_attn.out_proj.weight   Frozen  :  True
vision_tower.vision_model.encoder.layers.8.self_attn.out_proj.bias     Frozen  :  True
vision_tower.vision_model.encoder.layers.8.layer_norm1.weight          Frozen  :  True
vision_tower.vision_model.encoder.layers.8.layer_norm1.bias            Frozen  :  True
vision_tower.vision_model.encoder.layers.8.mlp.fc1.weight              Frozen  :  True
vision_tower.vision_model.encoder.layers.8.mlp.fc1.bias                Frozen  :  True
vision_tower.vision_model.encoder.layers.8.mlp.fc2.weight              Frozen  :  True
vision_tower.vision_model.encoder.layers.8.mlp.fc2.bias                Frozen  :  True
vision_tower.vision_model.encoder.layers.8.layer_norm2.weight          Frozen  :  True
vision_tower.vision_model.encoder.layers.8.layer_norm2.bias            Frozen  :  True
vision_tower.vision_model.encoder.layers.9.self_attn.k_proj.weight     Frozen  :  True
vision_tower.vision_model.encoder.layers.9.self_attn.k_proj.bias       Frozen  :  True
vision_tower.vision_model.encoder.layers.9.self_attn.v_proj.weight     Frozen  :  True
vision_tower.vision_model.encoder.layers.9.self_attn.v_proj.bias       Frozen  :  True
vision_tower.vision_model.encoder.layers.9.self_attn.q_proj.weight     Frozen  :  True
vision_tower.vision_model.encoder.layers.9.self_attn.q_proj.bias       Frozen  :  True
vision_tower.vision_model.encoder.layers.9.self_attn.out_proj.weight   Frozen  :  True
vision_tower.vision_model.encoder.layers.9.self_attn.out_proj.bias     Frozen  :  True
vision_tower.vision_model.encoder.layers.9.layer_norm1.weight          Frozen  :  True
vision_tower.vision_model.encoder.layers.9.layer_norm1.bias            Frozen  :  True
vision_tower.vision_model.encoder.layers.9.mlp.fc1.weight              Frozen  :  True
vision_tower.vision_model.encoder.layers.9.mlp.fc1.bias                Frozen  :  True
vision_tower.vision_model.encoder.layers.9.mlp.fc2.weight              Frozen  :  True
vision_tower.vision_model.encoder.layers.9.mlp.fc2.bias                Frozen  :  True
vision_tower.vision_model.encoder.layers.9.layer_norm2.weight          Frozen  :  True
vision_tower.vision_model.encoder.layers.9.layer_norm2.bias            Frozen  :  True
vision_tower.vision_model.encoder.layers.10.self_attn.k_proj.weight    Frozen  :  True
vision_tower.vision_model.encoder.layers.10.self_attn.k_proj.bias      Frozen  :  True
vision_tower.vision_model.encoder.layers.10.self_attn.v_proj.weight    Frozen  :  True
vision_tower.vision_model.encoder.layers.10.self_attn.v_proj.bias      Frozen  :  True
vision_tower.vision_model.encoder.layers.10.self_attn.q_proj.weight    Frozen  :  True
vision_tower.vision_model.encoder.layers.10.self_attn.q_proj.bias      Frozen  :  True
vision_tower.vision_model.encoder.layers.10.self_attn.out_proj.weight  Frozen  :  True
vision_tower.vision_model.encoder.layers.10.self_attn.out_proj.bias    Frozen  :  True
vision_tower.vision_model.encoder.layers.10.layer_norm1.weight         Frozen  :  True
vision_tower.vision_model.encoder.layers.10.layer_norm1.bias           Frozen  :  True
vision_tower.vision_model.encoder.layers.10.mlp.fc1.weight             Frozen  :  True
vision_tower.vision_model.encoder.layers.10.mlp.fc1.bias               Frozen  :  True
vision_tower.vision_model.encoder.layers.10.mlp.fc2.weight             Frozen  :  True
vision_tower.vision_model.encoder.layers.10.mlp.fc2.bias               Frozen  :  True
vision_tower.vision_model.encoder.layers.10.layer_norm2.weight         Frozen  :  True
vision_tower.vision_model.encoder.layers.10.layer_norm2.bias           Frozen  :  True
vision_tower.vision_model.encoder.layers.11.self_attn.k_proj.weight    Frozen  :  True
vision_tower.vision_model.encoder.layers.11.self_attn.k_proj.bias      Frozen  :  True
vision_tower.vision_model.encoder.layers.11.self_attn.v_proj.weight    Frozen  :  True
vision_tower.vision_model.encoder.layers.11.self_attn.v_proj.bias      Frozen  :  True
vision_tower.vision_model.encoder.layers.11.self_attn.q_proj.weight    Frozen  :  True
vision_tower.vision_model.encoder.layers.11.self_attn.q_proj.bias      Frozen  :  True
vision_tower.vision_model.encoder.layers.11.self_attn.out_proj.weight  Frozen  :  True
vision_tower.vision_model.encoder.layers.11.self_attn.out_proj.bias    Frozen  :  True
vision_tower.vision_model.encoder.layers.11.layer_norm1.weight         Frozen  :  True
vision_tower.vision_model.encoder.layers.11.layer_norm1.bias           Frozen  :  True
vision_tower.vision_model.encoder.layers.11.mlp.fc1.weight             Frozen  :  True
vision_tower.vision_model.encoder.layers.11.mlp.fc1.bias               Frozen  :  True
vision_tower.vision_model.encoder.layers.11.mlp.fc2.weight             Frozen  :  True
vision_tower.vision_model.encoder.layers.11.mlp.fc2.bias               Frozen  :  True
vision_tower.vision_model.encoder.layers.11.layer_norm2.weight         Frozen  :  True
vision_tower.vision_model.encoder.layers.11.layer_norm2.bias           Frozen  :  True
vision_tower.vision_model.encoder.layers.12.self_attn.k_proj.weight    Frozen  :  True
vision_tower.vision_model.encoder.layers.12.self_attn.k_proj.bias      Frozen  :  True
vision_tower.vision_model.encoder.layers.12.self_attn.v_proj.weight    Frozen  :  True
vision_tower.vision_model.encoder.layers.12.self_attn.v_proj.bias      Frozen  :  True
vision_tower.vision_model.encoder.layers.12.self_attn.q_proj.weight    Frozen  :  True
vision_tower.vision_model.encoder.layers.12.self_attn.q_proj.bias      Frozen  :  True
vision_tower.vision_model.encoder.layers.12.self_attn.out_proj.weight  Frozen  :  True
vision_tower.vision_model.encoder.layers.12.self_attn.out_proj.bias    Frozen  :  True
vision_tower.vision_model.encoder.layers.12.layer_norm1.weight         Frozen  :  True
vision_tower.vision_model.encoder.layers.12.layer_norm1.bias           Frozen  :  True
vision_tower.vision_model.encoder.layers.12.mlp.fc1.weight             Frozen  :  True
vision_tower.vision_model.encoder.layers.12.mlp.fc1.bias               Frozen  :  True
vision_tower.vision_model.encoder.layers.12.mlp.fc2.weight             Frozen  :  True
vision_tower.vision_model.encoder.layers.12.mlp.fc2.bias               Frozen  :  True
vision_tower.vision_model.encoder.layers.12.layer_norm2.weight         Frozen  :  True
vision_tower.vision_model.encoder.layers.12.layer_norm2.bias           Frozen  :  True
vision_tower.vision_model.encoder.layers.13.self_attn.k_proj.weight    Frozen  :  True
vision_tower.vision_model.encoder.layers.13.self_attn.k_proj.bias      Frozen  :  True
vision_tower.vision_model.encoder.layers.13.self_attn.v_proj.weight    Frozen  :  True
vision_tower.vision_model.encoder.layers.13.self_attn.v_proj.bias      Frozen  :  True
vision_tower.vision_model.encoder.layers.13.self_attn.q_proj.weight    Frozen  :  True
vision_tower.vision_model.encoder.layers.13.self_attn.q_proj.bias      Frozen  :  True
vision_tower.vision_model.encoder.layers.13.self_attn.out_proj.weight  Frozen  :  True
vision_tower.vision_model.encoder.layers.13.self_attn.out_proj.bias    Frozen  :  True
vision_tower.vision_model.encoder.layers.13.layer_norm1.weight         Frozen  :  True
vision_tower.vision_model.encoder.layers.13.layer_norm1.bias           Frozen  :  True
vision_tower.vision_model.encoder.layers.13.mlp.fc1.weight             Frozen  :  True
vision_tower.vision_model.encoder.layers.13.mlp.fc1.bias               Frozen  :  True
vision_tower.vision_model.encoder.layers.13.mlp.fc2.weight             Frozen  :  True
vision_tower.vision_model.encoder.layers.13.mlp.fc2.bias               Frozen  :  True
vision_tower.vision_model.encoder.layers.13.layer_norm2.weight         Frozen  :  True
vision_tower.vision_model.encoder.layers.13.layer_norm2.bias           Frozen  :  True
vision_tower.vision_model.encoder.layers.14.self_attn.k_proj.weight    Frozen  :  True
vision_tower.vision_model.encoder.layers.14.self_attn.k_proj.bias      Frozen  :  True
vision_tower.vision_model.encoder.layers.14.self_attn.v_proj.weight    Frozen  :  True
vision_tower.vision_model.encoder.layers.14.self_attn.v_proj.bias      Frozen  :  True
vision_tower.vision_model.encoder.layers.14.self_attn.q_proj.weight    Frozen  :  True
vision_tower.vision_model.encoder.layers.14.self_attn.q_proj.bias      Frozen  :  True
vision_tower.vision_model.encoder.layers.14.self_attn.out_proj.weight  Frozen  :  True
vision_tower.vision_model.encoder.layers.14.self_attn.out_proj.bias    Frozen  :  True
vision_tower.vision_model.encoder.layers.14.layer_norm1.weight         Frozen  :  True
vision_tower.vision_model.encoder.layers.14.layer_norm1.bias           Frozen  :  True
vision_tower.vision_model.encoder.layers.14.mlp.fc1.weight             Frozen  :  True
vision_tower.vision_model.encoder.layers.14.mlp.fc1.bias               Frozen  :  True
vision_tower.vision_model.encoder.layers.14.mlp.fc2.weight             Frozen  :  True
vision_tower.vision_model.encoder.layers.14.mlp.fc2.bias               Frozen  :  True
vision_tower.vision_model.encoder.layers.14.layer_norm2.weight         Frozen  :  True
vision_tower.vision_model.encoder.layers.14.layer_norm2.bias           Frozen  :  True
vision_tower.vision_model.encoder.layers.15.self_attn.k_proj.weight    Frozen  :  True
vision_tower.vision_model.encoder.layers.15.self_attn.k_proj.bias      Frozen  :  True
vision_tower.vision_model.encoder.layers.15.self_attn.v_proj.weight    Frozen  :  True
vision_tower.vision_model.encoder.layers.15.self_attn.v_proj.bias      Frozen  :  True
vision_tower.vision_model.encoder.layers.15.self_attn.q_proj.weight    Frozen  :  True
vision_tower.vision_model.encoder.layers.15.self_attn.q_proj.bias      Frozen  :  True
vision_tower.vision_model.encoder.layers.15.self_attn.out_proj.weight  Frozen  :  True
vision_tower.vision_model.encoder.layers.15.self_attn.out_proj.bias    Frozen  :  True
vision_tower.vision_model.encoder.layers.15.layer_norm1.weight         Frozen  :  True
vision_tower.vision_model.encoder.layers.15.layer_norm1.bias           Frozen  :  True
vision_tower.vision_model.encoder.layers.15.mlp.fc1.weight             Frozen  :  True
vision_tower.vision_model.encoder.layers.15.mlp.fc1.bias               Frozen  :  True
vision_tower.vision_model.encoder.layers.15.mlp.fc2.weight             Frozen  :  True
vision_tower.vision_model.encoder.layers.15.mlp.fc2.bias               Frozen  :  True
vision_tower.vision_model.encoder.layers.15.layer_norm2.weight         Frozen  :  True
vision_tower.vision_model.encoder.layers.15.layer_norm2.bias           Frozen  :  True
vision_tower.vision_model.encoder.layers.16.self_attn.k_proj.weight    Frozen  :  True
vision_tower.vision_model.encoder.layers.16.self_attn.k_proj.bias      Frozen  :  True
vision_tower.vision_model.encoder.layers.16.self_attn.v_proj.weight    Frozen  :  True
vision_tower.vision_model.encoder.layers.16.self_attn.v_proj.bias      Frozen  :  True
vision_tower.vision_model.encoder.layers.16.self_attn.q_proj.weight    Frozen  :  True
vision_tower.vision_model.encoder.layers.16.self_attn.q_proj.bias      Frozen  :  True
vision_tower.vision_model.encoder.layers.16.self_attn.out_proj.weight  Frozen  :  True
vision_tower.vision_model.encoder.layers.16.self_attn.out_proj.bias    Frozen  :  True
vision_tower.vision_model.encoder.layers.16.layer_norm1.weight         Frozen  :  True
vision_tower.vision_model.encoder.layers.16.layer_norm1.bias           Frozen  :  True
vision_tower.vision_model.encoder.layers.16.mlp.fc1.weight             Frozen  :  True
vision_tower.vision_model.encoder.layers.16.mlp.fc1.bias               Frozen  :  True
vision_tower.vision_model.encoder.layers.16.mlp.fc2.weight             Frozen  :  True
vision_tower.vision_model.encoder.layers.16.mlp.fc2.bias               Frozen  :  True
vision_tower.vision_model.encoder.layers.16.layer_norm2.weight         Frozen  :  True
vision_tower.vision_model.encoder.layers.16.layer_norm2.bias           Frozen  :  True
vision_tower.vision_model.encoder.layers.17.self_attn.k_proj.weight    Frozen  :  True
vision_tower.vision_model.encoder.layers.17.self_attn.k_proj.bias      Frozen  :  True
vision_tower.vision_model.encoder.layers.17.self_attn.v_proj.weight    Frozen  :  True
vision_tower.vision_model.encoder.layers.17.self_attn.v_proj.bias      Frozen  :  True
vision_tower.vision_model.encoder.layers.17.self_attn.q_proj.weight    Frozen  :  True
vision_tower.vision_model.encoder.layers.17.self_attn.q_proj.bias      Frozen  :  True
vision_tower.vision_model.encoder.layers.17.self_attn.out_proj.weight  Frozen  :  True
vision_tower.vision_model.encoder.layers.17.self_attn.out_proj.bias    Frozen  :  True
vision_tower.vision_model.encoder.layers.17.layer_norm1.weight         Frozen  :  True
vision_tower.vision_model.encoder.layers.17.layer_norm1.bias           Frozen  :  True
vision_tower.vision_model.encoder.layers.17.mlp.fc1.weight             Frozen  :  True
vision_tower.vision_model.encoder.layers.17.mlp.fc1.bias               Frozen  :  True
vision_tower.vision_model.encoder.layers.17.mlp.fc2.weight             Frozen  :  True
vision_tower.vision_model.encoder.layers.17.mlp.fc2.bias               Frozen  :  True
vision_tower.vision_model.encoder.layers.17.layer_norm2.weight         Frozen  :  True
vision_tower.vision_model.encoder.layers.17.layer_norm2.bias           Frozen  :  True
vision_tower.vision_model.encoder.layers.18.self_attn.k_proj.weight    Frozen  :  True
vision_tower.vision_model.encoder.layers.18.self_attn.k_proj.bias      Frozen  :  True
vision_tower.vision_model.encoder.layers.18.self_attn.v_proj.weight    Frozen  :  True
vision_tower.vision_model.encoder.layers.18.self_attn.v_proj.bias      Frozen  :  True
vision_tower.vision_model.encoder.layers.18.self_attn.q_proj.weight    Frozen  :  True
vision_tower.vision_model.encoder.layers.18.self_attn.q_proj.bias      Frozen  :  True
vision_tower.vision_model.encoder.layers.18.self_attn.out_proj.weight  Frozen  :  True
vision_tower.vision_model.encoder.layers.18.self_attn.out_proj.bias    Frozen  :  True
vision_tower.vision_model.encoder.layers.18.layer_norm1.weight         Frozen  :  True
vision_tower.vision_model.encoder.layers.18.layer_norm1.bias           Frozen  :  True
vision_tower.vision_model.encoder.layers.18.mlp.fc1.weight             Frozen  :  True
vision_tower.vision_model.encoder.layers.18.mlp.fc1.bias               Frozen  :  True
vision_tower.vision_model.encoder.layers.18.mlp.fc2.weight             Frozen  :  True
vision_tower.vision_model.encoder.layers.18.mlp.fc2.bias               Frozen  :  True
vision_tower.vision_model.encoder.layers.18.layer_norm2.weight         Frozen  :  True
vision_tower.vision_model.encoder.layers.18.layer_norm2.bias           Frozen  :  True
vision_tower.vision_model.encoder.layers.19.self_attn.k_proj.weight    Frozen  :  True
vision_tower.vision_model.encoder.layers.19.self_attn.k_proj.bias      Frozen  :  True
vision_tower.vision_model.encoder.layers.19.self_attn.v_proj.weight    Frozen  :  True
vision_tower.vision_model.encoder.layers.19.self_attn.v_proj.bias      Frozen  :  True
vision_tower.vision_model.encoder.layers.19.self_attn.q_proj.weight    Frozen  :  True
vision_tower.vision_model.encoder.layers.19.self_attn.q_proj.bias      Frozen  :  True
vision_tower.vision_model.encoder.layers.19.self_attn.out_proj.weight  Frozen  :  True
vision_tower.vision_model.encoder.layers.19.self_attn.out_proj.bias    Frozen  :  True
vision_tower.vision_model.encoder.layers.19.layer_norm1.weight         Frozen  :  True
vision_tower.vision_model.encoder.layers.19.layer_norm1.bias           Frozen  :  True
vision_tower.vision_model.encoder.layers.19.mlp.fc1.weight             Frozen  :  True
vision_tower.vision_model.encoder.layers.19.mlp.fc1.bias               Frozen  :  True
vision_tower.vision_model.encoder.layers.19.mlp.fc2.weight             Frozen  :  True
vision_tower.vision_model.encoder.layers.19.mlp.fc2.bias               Frozen  :  True
vision_tower.vision_model.encoder.layers.19.layer_norm2.weight         Frozen  :  True
vision_tower.vision_model.encoder.layers.19.layer_norm2.bias           Frozen  :  True
vision_tower.vision_model.encoder.layers.20.self_attn.k_proj.weight    Frozen  :  True
vision_tower.vision_model.encoder.layers.20.self_attn.k_proj.bias      Frozen  :  True
vision_tower.vision_model.encoder.layers.20.self_attn.v_proj.weight    Frozen  :  True
vision_tower.vision_model.encoder.layers.20.self_attn.v_proj.bias      Frozen  :  True
vision_tower.vision_model.encoder.layers.20.self_attn.q_proj.weight    Frozen  :  True
vision_tower.vision_model.encoder.layers.20.self_attn.q_proj.bias      Frozen  :  True
vision_tower.vision_model.encoder.layers.20.self_attn.out_proj.weight  Frozen  :  True
vision_tower.vision_model.encoder.layers.20.self_attn.out_proj.bias    Frozen  :  True
vision_tower.vision_model.encoder.layers.20.layer_norm1.weight         Frozen  :  True
vision_tower.vision_model.encoder.layers.20.layer_norm1.bias           Frozen  :  True
vision_tower.vision_model.encoder.layers.20.mlp.fc1.weight             Frozen  :  True
vision_tower.vision_model.encoder.layers.20.mlp.fc1.bias               Frozen  :  True
vision_tower.vision_model.encoder.layers.20.mlp.fc2.weight             Frozen  :  True
vision_tower.vision_model.encoder.layers.20.mlp.fc2.bias               Frozen  :  True
vision_tower.vision_model.encoder.layers.20.layer_norm2.weight         Frozen  :  True
vision_tower.vision_model.encoder.layers.20.layer_norm2.bias           Frozen  :  True
vision_tower.vision_model.encoder.layers.21.self_attn.k_proj.weight    Frozen  :  True
vision_tower.vision_model.encoder.layers.21.self_attn.k_proj.bias      Frozen  :  True
vision_tower.vision_model.encoder.layers.21.self_attn.v_proj.weight    Frozen  :  True
vision_tower.vision_model.encoder.layers.21.self_attn.v_proj.bias      Frozen  :  True
vision_tower.vision_model.encoder.layers.21.self_attn.q_proj.weight    Frozen  :  True
vision_tower.vision_model.encoder.layers.21.self_attn.q_proj.bias      Frozen  :  True
vision_tower.vision_model.encoder.layers.21.self_attn.out_proj.weight  Frozen  :  True
vision_tower.vision_model.encoder.layers.21.self_attn.out_proj.bias    Frozen  :  True
vision_tower.vision_model.encoder.layers.21.layer_norm1.weight         Frozen  :  True
vision_tower.vision_model.encoder.layers.21.layer_norm1.bias           Frozen  :  True
vision_tower.vision_model.encoder.layers.21.mlp.fc1.weight             Frozen  :  True
vision_tower.vision_model.encoder.layers.21.mlp.fc1.bias               Frozen  :  True
vision_tower.vision_model.encoder.layers.21.mlp.fc2.weight             Frozen  :  True
vision_tower.vision_model.encoder.layers.21.mlp.fc2.bias               Frozen  :  True
vision_tower.vision_model.encoder.layers.21.layer_norm2.weight         Frozen  :  True
vision_tower.vision_model.encoder.layers.21.layer_norm2.bias           Frozen  :  True
vision_tower.vision_model.encoder.layers.22.self_attn.k_proj.weight    Frozen  :  True
vision_tower.vision_model.encoder.layers.22.self_attn.k_proj.bias      Frozen  :  True
vision_tower.vision_model.encoder.layers.22.self_attn.v_proj.weight    Frozen  :  True
vision_tower.vision_model.encoder.layers.22.self_attn.v_proj.bias      Frozen  :  True
vision_tower.vision_model.encoder.layers.22.self_attn.q_proj.weight    Frozen  :  True
vision_tower.vision_model.encoder.layers.22.self_attn.q_proj.bias      Frozen  :  True
vision_tower.vision_model.encoder.layers.22.self_attn.out_proj.weight  Frozen  :  True
vision_tower.vision_model.encoder.layers.22.self_attn.out_proj.bias    Frozen  :  True
vision_tower.vision_model.encoder.layers.22.layer_norm1.weight         Frozen  :  True
vision_tower.vision_model.encoder.layers.22.layer_norm1.bias           Frozen  :  True
vision_tower.vision_model.encoder.layers.22.mlp.fc1.weight             Frozen  :  True
vision_tower.vision_model.encoder.layers.22.mlp.fc1.bias               Frozen  :  True
vision_tower.vision_model.encoder.layers.22.mlp.fc2.weight             Frozen  :  True
vision_tower.vision_model.encoder.layers.22.mlp.fc2.bias               Frozen  :  True
vision_tower.vision_model.encoder.layers.22.layer_norm2.weight         Frozen  :  True
vision_tower.vision_model.encoder.layers.22.layer_norm2.bias           Frozen  :  True
vision_tower.vision_model.encoder.layers.23.self_attn.k_proj.weight    Frozen  :  True
vision_tower.vision_model.encoder.layers.23.self_attn.k_proj.bias      Frozen  :  True
vision_tower.vision_model.encoder.layers.23.self_attn.v_proj.weight    Frozen  :  True
vision_tower.vision_model.encoder.layers.23.self_attn.v_proj.bias      Frozen  :  True
vision_tower.vision_model.encoder.layers.23.self_attn.q_proj.weight    Frozen  :  True
vision_tower.vision_model.encoder.layers.23.self_attn.q_proj.bias      Frozen  :  True
vision_tower.vision_model.encoder.layers.23.self_attn.out_proj.weight  Frozen  :  True
vision_tower.vision_model.encoder.layers.23.self_attn.out_proj.bias    Frozen  :  True
vision_tower.vision_model.encoder.layers.23.layer_norm1.weight         Frozen  :  True
vision_tower.vision_model.encoder.layers.23.layer_norm1.bias           Frozen  :  True
vision_tower.vision_model.encoder.layers.23.mlp.fc1.weight             Frozen  :  True
vision_tower.vision_model.encoder.layers.23.mlp.fc1.bias               Frozen  :  True
vision_tower.vision_model.encoder.layers.23.mlp.fc2.weight             Frozen  :  True
vision_tower.vision_model.encoder.layers.23.mlp.fc2.bias               Frozen  :  True
vision_tower.vision_model.encoder.layers.23.layer_norm2.weight         Frozen  :  True
vision_tower.vision_model.encoder.layers.23.layer_norm2.bias           Frozen  :  True
vision_tower.vision_model.encoder.layers.24.self_attn.k_proj.weight    Frozen  :  True
vision_tower.vision_model.encoder.layers.24.self_attn.k_proj.bias      Frozen  :  True
vision_tower.vision_model.encoder.layers.24.self_attn.v_proj.weight    Frozen  :  True
vision_tower.vision_model.encoder.layers.24.self_attn.v_proj.bias      Frozen  :  True
vision_tower.vision_model.encoder.layers.24.self_attn.q_proj.weight    Frozen  :  True
vision_tower.vision_model.encoder.layers.24.self_attn.q_proj.bias      Frozen  :  True
vision_tower.vision_model.encoder.layers.24.self_attn.out_proj.weight  Frozen  :  True
vision_tower.vision_model.encoder.layers.24.self_attn.out_proj.bias    Frozen  :  True
vision_tower.vision_model.encoder.layers.24.layer_norm1.weight         Frozen  :  True
vision_tower.vision_model.encoder.layers.24.layer_norm1.bias           Frozen  :  True
vision_tower.vision_model.encoder.layers.24.mlp.fc1.weight             Frozen  :  True
vision_tower.vision_model.encoder.layers.24.mlp.fc1.bias               Frozen  :  True
vision_tower.vision_model.encoder.layers.24.mlp.fc2.weight             Frozen  :  True
vision_tower.vision_model.encoder.layers.24.mlp.fc2.bias               Frozen  :  True
vision_tower.vision_model.encoder.layers.24.layer_norm2.weight         Frozen  :  True
vision_tower.vision_model.encoder.layers.24.layer_norm2.bias           Frozen  :  True
vision_tower.vision_model.encoder.layers.25.self_attn.k_proj.weight    Frozen  :  True
vision_tower.vision_model.encoder.layers.25.self_attn.k_proj.bias      Frozen  :  True
vision_tower.vision_model.encoder.layers.25.self_attn.v_proj.weight    Frozen  :  True
vision_tower.vision_model.encoder.layers.25.self_attn.v_proj.bias      Frozen  :  True
vision_tower.vision_model.encoder.layers.25.self_attn.q_proj.weight    Frozen  :  True
vision_tower.vision_model.encoder.layers.25.self_attn.q_proj.bias      Frozen  :  True
vision_tower.vision_model.encoder.layers.25.self_attn.out_proj.weight  Frozen  :  True
vision_tower.vision_model.encoder.layers.25.self_attn.out_proj.bias    Frozen  :  True
vision_tower.vision_model.encoder.layers.25.layer_norm1.weight         Frozen  :  True
vision_tower.vision_model.encoder.layers.25.layer_norm1.bias           Frozen  :  True
vision_tower.vision_model.encoder.layers.25.mlp.fc1.weight             Frozen  :  True
vision_tower.vision_model.encoder.layers.25.mlp.fc1.bias               Frozen  :  True
vision_tower.vision_model.encoder.layers.25.mlp.fc2.weight             Frozen  :  True
vision_tower.vision_model.encoder.layers.25.mlp.fc2.bias               Frozen  :  True
vision_tower.vision_model.encoder.layers.25.layer_norm2.weight         Frozen  :  True
vision_tower.vision_model.encoder.layers.25.layer_norm2.bias           Frozen  :  True
vision_tower.vision_model.encoder.layers.26.self_attn.k_proj.weight    Frozen  :  True
vision_tower.vision_model.encoder.layers.26.self_attn.k_proj.bias      Frozen  :  True
vision_tower.vision_model.encoder.layers.26.self_attn.v_proj.weight    Frozen  :  True
vision_tower.vision_model.encoder.layers.26.self_attn.v_proj.bias      Frozen  :  True
vision_tower.vision_model.encoder.layers.26.self_attn.q_proj.weight    Frozen  :  True
vision_tower.vision_model.encoder.layers.26.self_attn.q_proj.bias      Frozen  :  True
vision_tower.vision_model.encoder.layers.26.self_attn.out_proj.weight  Frozen  :  True
vision_tower.vision_model.encoder.layers.26.self_attn.out_proj.bias    Frozen  :  True
vision_tower.vision_model.encoder.layers.26.layer_norm1.weight         Frozen  :  True
vision_tower.vision_model.encoder.layers.26.layer_norm1.bias           Frozen  :  True
vision_tower.vision_model.encoder.layers.26.mlp.fc1.weight             Frozen  :  True
vision_tower.vision_model.encoder.layers.26.mlp.fc1.bias               Frozen  :  True
vision_tower.vision_model.encoder.layers.26.mlp.fc2.weight             Frozen  :  True
vision_tower.vision_model.encoder.layers.26.mlp.fc2.bias               Frozen  :  True
vision_tower.vision_model.encoder.layers.26.layer_norm2.weight         Frozen  :  True
vision_tower.vision_model.encoder.layers.26.layer_norm2.bias           Frozen  :  True
vision_tower.vision_model.post_layernorm.weight                        Frozen  :  True
vision_tower.vision_model.post_layernorm.bias                          Frozen  :  True
multi_modal_projector.linear.weight                                    Frozen  :  True
multi_modal_projector.linear.bias                                      Frozen  :  True
language_model.model.embed_tokens.weight                               Frozen  :  True
language_model.model.layers.0.self_attn.q_proj.weight                  Frozen  :  False
language_model.model.layers.0.self_attn.k_proj.weight                  Frozen  :  False
language_model.model.layers.0.self_attn.v_proj.weight                  Frozen  :  False
language_model.model.layers.0.self_attn.o_proj.weight                  Frozen  :  False
language_model.model.layers.0.mlp.gate_proj.weight                     Frozen  :  True
language_model.model.layers.0.mlp.up_proj.weight                       Frozen  :  True
language_model.model.layers.0.mlp.down_proj.weight                     Frozen  :  True
language_model.model.layers.0.input_layernorm.weight                   Frozen  :  True
language_model.model.layers.0.post_attention_layernorm.weight          Frozen  :  True
language_model.model.layers.1.self_attn.q_proj.weight                  Frozen  :  False
language_model.model.layers.1.self_attn.k_proj.weight                  Frozen  :  False
language_model.model.layers.1.self_attn.v_proj.weight                  Frozen  :  False
language_model.model.layers.1.self_attn.o_proj.weight                  Frozen  :  False
language_model.model.layers.1.mlp.gate_proj.weight                     Frozen  :  True
language_model.model.layers.1.mlp.up_proj.weight                       Frozen  :  True
language_model.model.layers.1.mlp.down_proj.weight                     Frozen  :  True
language_model.model.layers.1.input_layernorm.weight                   Frozen  :  True
language_model.model.layers.1.post_attention_layernorm.weight          Frozen  :  True
language_model.model.layers.2.self_attn.q_proj.weight                  Frozen  :  False
language_model.model.layers.2.self_attn.k_proj.weight                  Frozen  :  False
language_model.model.layers.2.self_attn.v_proj.weight                  Frozen  :  False
language_model.model.layers.2.self_attn.o_proj.weight                  Frozen  :  False
language_model.model.layers.2.mlp.gate_proj.weight                     Frozen  :  True
language_model.model.layers.2.mlp.up_proj.weight                       Frozen  :  True
language_model.model.layers.2.mlp.down_proj.weight                     Frozen  :  True
language_model.model.layers.2.input_layernorm.weight                   Frozen  :  True
language_model.model.layers.2.post_attention_layernorm.weight          Frozen  :  True
language_model.model.layers.3.self_attn.q_proj.weight                  Frozen  :  False
language_model.model.layers.3.self_attn.k_proj.weight                  Frozen  :  False
language_model.model.layers.3.self_attn.v_proj.weight                  Frozen  :  False
language_model.model.layers.3.self_attn.o_proj.weight                  Frozen  :  False
language_model.model.layers.3.mlp.gate_proj.weight                     Frozen  :  True
language_model.model.layers.3.mlp.up_proj.weight                       Frozen  :  True
language_model.model.layers.3.mlp.down_proj.weight                     Frozen  :  True
language_model.model.layers.3.input_layernorm.weight                   Frozen  :  True
language_model.model.layers.3.post_attention_layernorm.weight          Frozen  :  True
language_model.model.layers.4.self_attn.q_proj.weight                  Frozen  :  False
language_model.model.layers.4.self_attn.k_proj.weight                  Frozen  :  False
language_model.model.layers.4.self_attn.v_proj.weight                  Frozen  :  False
language_model.model.layers.4.self_attn.o_proj.weight                  Frozen  :  False
language_model.model.layers.4.mlp.gate_proj.weight                     Frozen  :  True
language_model.model.layers.4.mlp.up_proj.weight                       Frozen  :  True
language_model.model.layers.4.mlp.down_proj.weight                     Frozen  :  True
language_model.model.layers.4.input_layernorm.weight                   Frozen  :  True
language_model.model.layers.4.post_attention_layernorm.weight          Frozen  :  True
language_model.model.layers.5.self_attn.q_proj.weight                  Frozen  :  False
language_model.model.layers.5.self_attn.k_proj.weight                  Frozen  :  False
language_model.model.layers.5.self_attn.v_proj.weight                  Frozen  :  False
language_model.model.layers.5.self_attn.o_proj.weight                  Frozen  :  False
language_model.model.layers.5.mlp.gate_proj.weight                     Frozen  :  True
language_model.model.layers.5.mlp.up_proj.weight                       Frozen  :  True
language_model.model.layers.5.mlp.down_proj.weight                     Frozen  :  True
language_model.model.layers.5.input_layernorm.weight                   Frozen  :  True
language_model.model.layers.5.post_attention_layernorm.weight          Frozen  :  True
language_model.model.layers.6.self_attn.q_proj.weight                  Frozen  :  False
language_model.model.layers.6.self_attn.k_proj.weight                  Frozen  :  False
language_model.model.layers.6.self_attn.v_proj.weight                  Frozen  :  False
language_model.model.layers.6.self_attn.o_proj.weight                  Frozen  :  False
language_model.model.layers.6.mlp.gate_proj.weight                     Frozen  :  True
language_model.model.layers.6.mlp.up_proj.weight                       Frozen  :  True
language_model.model.layers.6.mlp.down_proj.weight                     Frozen  :  True
language_model.model.layers.6.input_layernorm.weight                   Frozen  :  True
language_model.model.layers.6.post_attention_layernorm.weight          Frozen  :  True
language_model.model.layers.7.self_attn.q_proj.weight                  Frozen  :  False
language_model.model.layers.7.self_attn.k_proj.weight                  Frozen  :  False
language_model.model.layers.7.self_attn.v_proj.weight                  Frozen  :  False
language_model.model.layers.7.self_attn.o_proj.weight                  Frozen  :  False
language_model.model.layers.7.mlp.gate_proj.weight                     Frozen  :  True
language_model.model.layers.7.mlp.up_proj.weight                       Frozen  :  True
language_model.model.layers.7.mlp.down_proj.weight                     Frozen  :  True
language_model.model.layers.7.input_layernorm.weight                   Frozen  :  True
language_model.model.layers.7.post_attention_layernorm.weight          Frozen  :  True
language_model.model.layers.8.self_attn.q_proj.weight                  Frozen  :  False
language_model.model.layers.8.self_attn.k_proj.weight                  Frozen  :  False
language_model.model.layers.8.self_attn.v_proj.weight                  Frozen  :  False
language_model.model.layers.8.self_attn.o_proj.weight                  Frozen  :  False
language_model.model.layers.8.mlp.gate_proj.weight                     Frozen  :  True
language_model.model.layers.8.mlp.up_proj.weight                       Frozen  :  True
language_model.model.layers.8.mlp.down_proj.weight                     Frozen  :  True
language_model.model.layers.8.input_layernorm.weight                   Frozen  :  True
language_model.model.layers.8.post_attention_layernorm.weight          Frozen  :  True
language_model.model.layers.9.self_attn.q_proj.weight                  Frozen  :  False
language_model.model.layers.9.self_attn.k_proj.weight                  Frozen  :  False
language_model.model.layers.9.self_attn.v_proj.weight                  Frozen  :  False
language_model.model.layers.9.self_attn.o_proj.weight                  Frozen  :  False
language_model.model.layers.9.mlp.gate_proj.weight                     Frozen  :  True
language_model.model.layers.9.mlp.up_proj.weight                       Frozen  :  True
language_model.model.layers.9.mlp.down_proj.weight                     Frozen  :  True
language_model.model.layers.9.input_layernorm.weight                   Frozen  :  True
language_model.model.layers.9.post_attention_layernorm.weight          Frozen  :  True
language_model.model.layers.10.self_attn.q_proj.weight                 Frozen  :  False
language_model.model.layers.10.self_attn.k_proj.weight                 Frozen  :  False
language_model.model.layers.10.self_attn.v_proj.weight                 Frozen  :  False
language_model.model.layers.10.self_attn.o_proj.weight                 Frozen  :  False
language_model.model.layers.10.mlp.gate_proj.weight                    Frozen  :  True
language_model.model.layers.10.mlp.up_proj.weight                      Frozen  :  True
language_model.model.layers.10.mlp.down_proj.weight                    Frozen  :  True
language_model.model.layers.10.input_layernorm.weight                  Frozen  :  True
language_model.model.layers.10.post_attention_layernorm.weight         Frozen  :  True
language_model.model.layers.11.self_attn.q_proj.weight                 Frozen  :  False
language_model.model.layers.11.self_attn.k_proj.weight                 Frozen  :  False
language_model.model.layers.11.self_attn.v_proj.weight                 Frozen  :  False
language_model.model.layers.11.self_attn.o_proj.weight                 Frozen  :  False
language_model.model.layers.11.mlp.gate_proj.weight                    Frozen  :  True
language_model.model.layers.11.mlp.up_proj.weight                      Frozen  :  True
language_model.model.layers.11.mlp.down_proj.weight                    Frozen  :  True
language_model.model.layers.11.input_layernorm.weight                  Frozen  :  True
language_model.model.layers.11.post_attention_layernorm.weight         Frozen  :  True
language_model.model.layers.12.self_attn.q_proj.weight                 Frozen  :  False
language_model.model.layers.12.self_attn.k_proj.weight                 Frozen  :  False
language_model.model.layers.12.self_attn.v_proj.weight                 Frozen  :  False
language_model.model.layers.12.self_attn.o_proj.weight                 Frozen  :  False
language_model.model.layers.12.mlp.gate_proj.weight                    Frozen  :  True
language_model.model.layers.12.mlp.up_proj.weight                      Frozen  :  True
language_model.model.layers.12.mlp.down_proj.weight                    Frozen  :  True
language_model.model.layers.12.input_layernorm.weight                  Frozen  :  True
language_model.model.layers.12.post_attention_layernorm.weight         Frozen  :  True
language_model.model.layers.13.self_attn.q_proj.weight                 Frozen  :  False
language_model.model.layers.13.self_attn.k_proj.weight                 Frozen  :  False
language_model.model.layers.13.self_attn.v_proj.weight                 Frozen  :  False
language_model.model.layers.13.self_attn.o_proj.weight                 Frozen  :  False
language_model.model.layers.13.mlp.gate_proj.weight                    Frozen  :  True
language_model.model.layers.13.mlp.up_proj.weight                      Frozen  :  True
language_model.model.layers.13.mlp.down_proj.weight                    Frozen  :  True
language_model.model.layers.13.input_layernorm.weight                  Frozen  :  True
language_model.model.layers.13.post_attention_layernorm.weight         Frozen  :  True
language_model.model.layers.14.self_attn.q_proj.weight                 Frozen  :  False
language_model.model.layers.14.self_attn.k_proj.weight                 Frozen  :  False
language_model.model.layers.14.self_attn.v_proj.weight                 Frozen  :  False
language_model.model.layers.14.self_attn.o_proj.weight                 Frozen  :  False
language_model.model.layers.14.mlp.gate_proj.weight                    Frozen  :  True
language_model.model.layers.14.mlp.up_proj.weight                      Frozen  :  True
language_model.model.layers.14.mlp.down_proj.weight                    Frozen  :  True
language_model.model.layers.14.input_layernorm.weight                  Frozen  :  True
language_model.model.layers.14.post_attention_layernorm.weight         Frozen  :  True
language_model.model.layers.15.self_attn.q_proj.weight                 Frozen  :  False
language_model.model.layers.15.self_attn.k_proj.weight                 Frozen  :  False
language_model.model.layers.15.self_attn.v_proj.weight                 Frozen  :  False
language_model.model.layers.15.self_attn.o_proj.weight                 Frozen  :  False
language_model.model.layers.15.mlp.gate_proj.weight                    Frozen  :  True
language_model.model.layers.15.mlp.up_proj.weight                      Frozen  :  True
language_model.model.layers.15.mlp.down_proj.weight                    Frozen  :  True
language_model.model.layers.15.input_layernorm.weight                  Frozen  :  True
language_model.model.layers.15.post_attention_layernorm.weight         Frozen  :  True
language_model.model.layers.16.self_attn.q_proj.weight                 Frozen  :  False
language_model.model.layers.16.self_attn.k_proj.weight                 Frozen  :  False
language_model.model.layers.16.self_attn.v_proj.weight                 Frozen  :  False
language_model.model.layers.16.self_attn.o_proj.weight                 Frozen  :  False
language_model.model.layers.16.mlp.gate_proj.weight                    Frozen  :  True
language_model.model.layers.16.mlp.up_proj.weight                      Frozen  :  True
language_model.model.layers.16.mlp.down_proj.weight                    Frozen  :  True
language_model.model.layers.16.input_layernorm.weight                  Frozen  :  True
language_model.model.layers.16.post_attention_layernorm.weight         Frozen  :  True
language_model.model.layers.17.self_attn.q_proj.weight                 Frozen  :  False
language_model.model.layers.17.self_attn.k_proj.weight                 Frozen  :  False
language_model.model.layers.17.self_attn.v_proj.weight                 Frozen  :  False
language_model.model.layers.17.self_attn.o_proj.weight                 Frozen  :  False
language_model.model.layers.17.mlp.gate_proj.weight                    Frozen  :  True
language_model.model.layers.17.mlp.up_proj.weight                      Frozen  :  True
language_model.model.layers.17.mlp.down_proj.weight                    Frozen  :  True
language_model.model.layers.17.input_layernorm.weight                  Frozen  :  True
language_model.model.layers.17.post_attention_layernorm.weight         Frozen  :  True
language_model.model.norm.weight                                       Frozen  :  True
```

# accelerate settings
By passing device_map="auto", we tell Accelerate to determine automatically
where to put each layer of the model depending on the available resources:

* first we use the maximum space available on the GPU(s)
if we still need space, we store the remaining weights on the CPU
if there is not enough RAM,
we store the remaining weights on the hard drive as memory-mapped tensors
> see...... but it maxes out my gpu and does not store on my harddisk at all.....[] 

* dont split any blocks that have residual connections
no_split_module_classes=["GPTJBlock"] indicates that the modules that are
GPTJBlock should not be split on different devices. You should set here all
blocks that include a residual connection of some kind.

* use device map to see where they are :
```python
model.hf_device_map
```

```python
{'vision_tower': 0, 'multi_modal_projector': 0,
'language_model.model.embed_tokens': 0, 'language_model.lm_head': 0,
'language_model.model.layers.0': 0, 'langu age_model.model.layers.1': 0,
'language_model.model.layers.2': 0, 'language_model.model.layers.3': 0,
'language_model.model.layers.4': 0, 'language_model.model .layers.5': 0,
'language_model.model.layers.6': 0, 'language_model.model.layers.7': 0,
'language_model.model.layers.8': 0, 'language_model.model.layers.9': 0,
'language_model.model.layers.10': 0, 'language_model.model.layers.11': 0,
'language_model.model.layers.12': 0, 'language_model.model.layers.13': 'cpu',
'language_model.model.layers.14': 'cpu', 'language_model.model.layers.15':
'cpu', 'language_model.model.layers.16': 'cpu',
'language_model.model.layers.17': 'cpu', 'language_model.model.norm': 'cpu'}
```


[liar??]
Behind the scenes, 🤗 Accelerate adds hooks to the model, so that:

    at each layer, the inputs are put on the right device (so even if your model is spread across several GPUs, it works)
    for the weights offloaded on the CPU, they are put on a GPU just before the forward pass, and cleaned up just after
    for the weights offloaded on the hard drive, they are loaded in RAM then put on a GPU just before the forward pass, and cleaned up just after

This way, you model can run for inference even if it doesn’t fit on one of the GPUs or the CPU RAM!

device_map "auto" should be enough

First note that you can limit the memory used on each GPU by using the
max_memory argument (available in infer_auto_device_map() and in all functions
using it). When setting max_memory, you should pass along a dictionary
containing the GPU identifiers (for instance 0, 1 etc.) and the "cpu" key for
the maximum RAM you want used for CPU offload. The values can either be an
integer (in bytes) or a string representing a number with its unit, such as
"10GiB" or "10GB".


```python
from accelerate import infer_auto_device_map
device_map = infer_auto_device_map(my_model, max_memory={0: "10GiB", 1: "10GiB", "cpu": "30GiB"})
```

# well this is not for fine-tuning

* from slides
base_model = AutoModel.from_pretrained(model_id, device_map="auto")
from peft import LoraConfig , get_peft_model
config = LoraConfig(r=...)
model = get_peft_model(base_model, config)
optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
for batch in dataloader:
Note 3/4 of training size is reserved for gradients
If N number of params, then we need:
* N number of gradients (current gradient)
* 2N number of optimization states (also includes gradient if method is like momentum)
So 3/4 of training memory is reserved for gradients and optimization states

## useful Quantization notes
FP32 requires 4bytes
bf16 , fp16 requires 2bytes

int8 / int4 requires 1byte/ 0.5bytes [but this is only used at inference time]
training quantized models is not possible (as they are integers .. gradients cant be calculated)
Lora lets you do this i dont know how...

## shape and size of the layers of paligemma at bf16
without quant.train() -> 16 gigs
with -> 2.1 gig

peft continue training [notebook][https://colab.research.google.com/drive/12pMorxvLV-VwjuNBM76L4xXnzVYg57iB?usp=sharing#scrollTo=6CWAxZoubb9p]
