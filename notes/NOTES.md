# NOTES

* DSRI access pending [got it]

* vpn [🫠]

  ![image-20240918120459672](/home/adi/.config/Typora/typora-user-images/image-20240918120459672.png)

> Purpose of report ?? :
>
> ​	xfolds:
>
> * Do 4m on medpix with scaling law
>
> * compare 4m caption generation with previous models (but all the previous models all use big lang model for vqa)
>
> * compare other modality generations with their soa. [havent found yet]
>
> secondary:
>
> * give effective making techniques allowing for longer sequence lengths (compare)[masking operations are not fused]
> * page 4

![image-20240915095043352](/home/adi/.config/Typora/typora-user-images/image-20240915095043352.png)

## Plan was 

Input img :

out of the box expected modalities

* Bounding Box on the interested region & classification (comes from caption generation)

* segmentation.[why ]

* conditional caption generation. to make it like vqa will need a really big lang model.


Caption generation is the most interesting modality ? [also gives us chance to do vqa]

captions :

![image-20240915104727295](/home/adi/.config/Typora/typora-user-images/image-20240915104727295.png)

```json


// we have captions for each image we have :

// topic captions [about what the individual has] [what we expect with image and no caption]

    {
        "Type": "CT",
        "U_id": "MPX1009",
        "image": "MPX1009_synpic46283",
        "Description": {
            "ACR Codes": "8.-1",
            "Age": "73",
            "Caption": "The prostate is enlarged with several calcifications  noted within.  No dominant prostate mass is evident.",
            "Figure Part": null,
            "Modality": "CT - noncontrast",
            "Plane": "Coronal",
            "Sex": "male"
        },
        "Location": "Genitourinary",
        "Location Category": "Reproductive and Urinary System"
    },

// -----------------------------

// case captions [about the case the individual has][history][exams done][findings][literature]
    {
        "Case": {
            "Title": "Bladder Diverticulum",
            "History": "73-year-old male with hematuria and numerous white blood cells found on UA",
            "Exam": "N/A",
            "Findings": "Bladder with thickened wall and diverticulum on the right.  Diverticulum is mostly likely secondary to chronic outflow obstruction.\n\nProstate enlargement.",
            "Differential Diagnosis": "Bladder Diverticulum",
            "Case Diagnosis": "Bladder Diverticulum",
            "Diagnosis By": "N/A"
        },
        "Topic": {
            "Title": "Bladder Diverticulum",
            "Disease Discussion": "Bladder diverticula most often occur as a result of outlet obstruction.  Occasionally, a congenital weakness in the bladder wall adjacent to the ureteral orifice results in a diverticulum.  This is termed a \"Hutch\" diverticulum.\nIn children, outlet obstruction causing a diverticulum is rare and can be seen with urethral valves.  In men, diverticula are associated with outlet obstruction from urethral stricture, prostatic hypertrophy, prostatic carcinoma etc.  acquired diverticula are rare in women.\nDiverticula usually occur on the lateral bladder walls, rarely the dome.  They are often multiple.  Large diverticula often displace the bladder and or ureters.  \ndiverticula can have wide or narrow necks.  The wide necked variety empty urine readily.  The narrow neck type are slow to empty and therefore are more likely to have urinary stasis.\nInfection, tumor and stone formation can occur as a result of urine stasis within a diverticulum.  Tumor formation in a diverticulum is more likely to spread beyond the bladder because the diverticulum wall consists only of urothelium without muscle.\nBladder diverticula can be evaluated with excretory urography, ultrasound, CT and cystoscopy.\n\nRef:\nDunnick, R., McCallum, R., Sandler, C., Textbook of Uroradiology.",
            "ACR Code": "8.9",
            "Category": "Diverticulum"
        }
    },



```



## comparing caption generation:

MedClip https://huggingface.co/flaviagiammarino/pubmed-clip-vit-base-patch32

PubMedCLIP [Dec 2021] was trained on the [Radiology Objects in COntext (ROCO)](https://github.com/razorx89/roco-dataset) dataset, a large-scale multimodal medical imaging dataset. The ROCO dataset includes diverse imaging modalities (such as X-Ray,  MRI, ultrasound, fluoroscopy, etc.) from various human body regions  (such as head, spine, chest, abdomen, etc.)  captured from open-access [PubMed](https://pubmed.ncbi.nlm.nih.gov/) articles.

![image-20240915112018894](/home/adi/.config/Typora/typora-user-images/image-20240915112018894.png)

PubMedCLIP was trained for 50 epochs with a batch size of 64 using the Adam optimizer with a learning rate of 10−5.  The authors have released three different pre-trained models at this [link](https://1drv.ms/u/s!ApXgPqe9kykTgwD4Np3-f7ODAot8?e=zLVlJ2)  which use ResNet-50, ResNet-50x4 and ViT32 as image encoders. This  repository includes only the ViT32 variant of the PubMedCLIP model.

![image-20240915111936424](/home/adi/.config/Typora/typora-user-images/image-20240915111936424.png)

- **Repository:** [PubMedCLIP Official GitHub Repository](https://github.com/sarahESL/PubMedCLIP)
- **Paper:** [Does CLIP Benefit Visual Question Answering in the Medical Domain as Much as it Does in the General Domain?](https://arxiv.org/abs/2112.13906)
- Current approaches for this multimodal task adopt deep neural encoders to interpret the image and the question and then pick a corresponding answer. They typically consist of four main components: a visual encoder, question encoder, attention-based fusion of vision and text features, and an answer classiﬁer Vu et al. [2020], Zhan et al. [2020], Nguyen et al. [2019], Pan et al. [2021], Liu et al. [2021b].[mostly 1 word answer]



[paligemma] 2024 june Google [https://arxiv.org/abs/2407.07726]

* 4m instead of siglip VIT model 

  >[multi modality]
  >Additionally,
  >as shown in the literature, by converting more
  >
  >complex structured outputs into “text”, this API
  >can also cover more tasks such as: detection [21],
  >instance segmentation, panoptic seg-
  >mentation, depth prediction, colorization, and
  >many more. This conversion can be
  >hand-engineered and task-specific, such as done
  >in pix2seq [21] for detection, or learned as is
  >the case for segmentation [54] and dense output
  >tasks in general. [page 2] [same as what 4m says]
  >
  >
  >
  >Adapt to new inputs such as multiple im-
  >ages (NLVR2) or bounding boxes draw in the
  >image (WidgetCap). Or it could take the form of
  >instruction [67] or even chat [44] tuning. [page 5]
  >
  >--------------
  >
  >There are generally two ways to build vision-language models (VLMs): the first option is to connect a
  >vision encoder to a large language model while the second option is to use a transformer decoder-only
  >architecture to handle both vision and language modalities. [page 28]
  >
  >----------------
  >
  >[smaller size]
  >
  >Following PaLI-3’s strong experimental results,
  >we use a SigLIP image encoder. While PaLI-3
  >(and others [6, 25]) use a large image model such
  >as ViT-G, we use the much smaller but similarly
  >strong “shape optimized” ViT-So400m model.
  >
  >
  >
  >However,
  >PaLI-3 has shown that across a wide range of
  >vision-language tasks, a well-trained small 5 B
  >model (2 B vision + 3 B language) can attain the
  >same performance as the much larger 55 B PaLI-X
  >(22 B vision + 32 B language) and 562 B PaLM-E
  >(22 B vision + 540 B language), including tasks
  >such as ScienceQA. With PaliGemma we continue
  >this push for smaller models and show that we
  >can keep the same performance with less than
  >3 B total parameters.
  >
  >------------------
  >
  >Hence, again
  >in the spirit of learning more skills during pre-
  >training, we depart from common practice and
  >do not freeze the image encoder. [page 4]
  >
  >
  >
  >avoid destructive supervision signal from the ini-
  >tially unaligned language model, we use a slow
  >linear warm-up for the image encoder’s learning-
  >rate (Figure 3), which ensures that the image
  >encoder’s quality is not deteriorated from the ini-
  >tially misaligned gradients coming through the
  >LLM.

  

  

  

  

![image-20240915123415831](/home/adi/.config/Typora/typora-user-images/image-20240915123415831.png)

fine tune : https://colab.research.google.com/github/google-research/big_vision/blob/main/big_vision/configs/proj/paligemma/finetune_paligemma.ipynb

* we get prompt tokenizer and lang model for free.
* i coded a small siglip VIT without linear project

## other notes

* find benchmark datasets if you dont use ultraSound , flouroscopy



## GPU

> The experiments were conducted using
> one 32 GB GPU (Nvidia DGX1 8x Tesla V100) in an OKD 4.6 cluster
> under the Maastricht University Data Science Research Infrastructure
>
> ![image-20240915104133992](/home/adi/.config/Typora/typora-user-images/image-20240915104133992.png)
>
> hours of training ?





```
To compute the performance in petaflop days (PF-days) for an Nvidia DGX-1 with 8x Tesla V100 GPUs, we can break it down as follows:
1. Understand PFLOPS for Nvidia DGX-1 with 8x V100:

    A Tesla V100 (32 GB) GPU can deliver up to 15.7 teraflops (TFLOPS) of single-precision (FP32) performance.

    The DGX-1 has 8 V100 GPUs, so the total peak single-precision performance of the DGX-1 is:
    Total TFLOPS=15.7×8=125.6 TFLOPS
    Total TFLOPS=15.7×8=125.6 TFLOPS

    Convert TFLOPS to petaflops (PFLOPS):
    Total PFLOPS=125.61000=0.1256 PFLOPS
    Total PFLOPS=1000125.6​=0.1256 PFLOPS

2. Time Calculation for PF-days:

    1 PF-day is equal to 1 petaflop sustained for 1 day (i.e., 24 hours or 86,400 seconds).

    Now, you can calculate how many PF-days the DGX-1 can achieve in a given time period. For 1 day, the DGX-1 can achieve:
    PF-days=0.1256 PFLOPS×1 day=0.1256 PF-days/day
This means the Nvidia DGX-1 with 8x Tesla V100 GPUs can deliver 0.1256 petaflop-days of computation in one day.

-- chatgpt [its a bit wrong!]
```

from 4m
* All modalities are mapped to sets or sequences of discrete tokens (indices of a
vocabulary) through the use of modality specific tokenizer
* captions and bounding boxes are both treated as text and tokenized by wordpiece
* Encoder is a std transformer `but features modality-specific learnable input embedding layers to map token indices to vectors` .
* Non learnable sin-cos position embedding.



* The decoder handles tokens from both dense image-like and sequence-like modalities, with each type requiring a different approach



# Second Meet

from previous :

* 4m reasons to do segmentation, or other tasks we have
* more gaps! 
* if i cant make production grade i would atleast need to show one downstream task as an example.
* who would this be targeted to ?


* Released Dataset [includes for pretraining and alignment]
* Fine tuned Gemma [not released]
* benchmark []
