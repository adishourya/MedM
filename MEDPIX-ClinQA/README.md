
---

# MEDPIX Visual Question Answering (VQA) Dataset

## Overview

This dataset builds on the **MEDPIX 2.0** dataset to create a Visual Question Answering (VQA) resource for medical imagery. It complements existing datasets like [VQA-RAD](https://huggingface.co/datasets/flaviagiammarino/vqa-rad), which provides domain-expert validated QA pairs from a subset of MEDPIX. While VQA-RAD offers high-quality data, it may not have enough volume for many use cases. This dataset expands on the original captions, topics, and descriptions in MEDPIX by generating two types of question sets for each image (10 questions per image):
Get the images from their github : [MedPix-2.0][https://github.com/CHILab1/MedPix-2.0] 

1. **Pre-training Questions**: These questions are derived directly from the MEDPIX description and case files. These are designed for use in early epochs for  getting good **next-token generation**.

2. **Alignment Questions**: These questions incorporate more context, aiming to help the model better handle open-ended and direct questions. They are generated using the **Llama 3.1 8B model** and are intended for later epochs to improve model alignment.


## Citations

If you use this dataset, please credit the original MEDPIX 2.0 work by Siragusa et al.:

```
@misc{siragusa2024medpix20comprehensivemultimodal,
      title={MedPix 2.0: A Comprehensive Multimodal Biomedical Dataset for Advanced AI Applications}, 
      author={Irene Siragusa and Salvatore Contino and Massimo La Ciura and Rosario Alicata and Roberto Pirrone},
      year={2024},
      eprint={2407.02994},
      archivePrefix={arXiv},
      primaryClass={cs.DB},
      url={https://arxiv.org/abs/2407.02994}, 
}
```

For the Llama-generated alignment QA:

```bibtex
@misc{llama31,
  title={Llama 3.1: Large Language Model},
  author={Meta AI},
  year={2024},
  note={8B Model},
  url={https://ai.meta.com/llama/}
}
```

---

