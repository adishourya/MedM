# Second Meeting

- 26th Sep / Thursday

-------------------

so we will go over :

	* Last Week
	* This Week
	* Quick Demo from yesterday's checkpoint [very likely that it might be funny]

------------------

## Administrative stuff

* vpn [still 🫠]

  ![image-20240918120459672](/home/adi/.config/Typora/typora-user-images/image-20240918120459672.png)

> ::



## from previous meeting

1. who would use our model ? [grad students / radiologist / non medical]
   1. for multimodality : I said segmentation.. and you asked why ? [and i didnt have an answer]
2. Resarch gaps [besides using different kind of models]
3. Areas of improvement [efficieny or usage]

>1. [Radiologist and patients][hope its not generic.. this is still very informal]
>
>   -------------
>
>   
>
>   Radiologists typically know which bone or anatomical region they will be examining when they suggest the patient get a CT scan or an MRI
>
>   [so they dont need image segmentation]
>
>   
>
>   However, the patient might.
>
>   [better in clinics which have long latency]
>
>   * After the scan, the patient can receive a copy of the scan and our model while they wair to see their doctor.
>   * During this time, they can use our model to educate themselves on these basic, broad labels. This will help them prepare better, more informed questions for their doctor.
>
>   
>
>   Potential questions the patient might ask:
>
>   - **What am I looking at?** [would allow us to do : semantic segmentation]
>   - **How does a healthy [xyz] look?** [allows us to do: text to image but since they can look a bit cartoonish an even better option would be for the clinic to provide a healthy example of the [xyz] label for comparison, instead of them relying on googling image]
>   - **What condition might I have?** [allows us to do VQA but we will also have to release a guarded model so that it doesnt say horrible stuff while they wait for the doctor .. or more intensive allignment after pretraining ]
>
>   
>
>   -----------
>
>   
>
>   After consultation, we can have a human readable report generation:
>
>   - The scan image alongside its segmentation to ensure clarity in discussions.
>   - A summary of the questions the patient asked, providing them with a clear record of the conversation.
>   - and more... [this can probably be different for diffferent clinics]
>
>----------------
>
>But since 4m can do many modality generation.
>
>* we only need to give them 1 model -> so they wont need as much resource.
>* and in future if they need support for other modality -> they can finetune themselves solving their privacy issues
>* ... more maybe



![image-20240926094100798](/home/adi/.config/Typora/typora-user-images/image-20240926094100798.png)



> For improvement :
>
> --------
>
> * masking [there is a lot of masking]
>   * unified masking but not fused.
>   * even we can improve by percentage this could save a lot. [since the generic models are pretrained on Millions of images]
>   * 4m did on CC12M



## Last Week

* Released a VQA Dataset

* we had a lot of captions

  ```json
  
  // U id is just patient id . each patient might have multiple images .
  // but we are not doing that ...🥶
  
  // captions per image
      {
          "Type": "CT",
          "U_id": "MPX1009",
          "image": "MPX1009_synpic46283",
          "Description": {
              "ACR Codes": "8.-1",
              "Age": "73",
              "Caption": "The prostate is enlarged with several calcifications  noted within.  No dominant prostate mass is evident.",
              "Figure Part": null,       <---------- This is almost always null
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

* Medpix also demos a vqa model [screenshot from the paper]

  ![image-20240926105929164](/home/adi/.config/Typora/typora-user-images/image-20240926105929164.png)

* I was struggling to load in the layers of the model

* Layerstats [base model]:

  > ![image-20240926110509030](/home/adi/.config/Typora/typora-user-images/image-20240926110509030.png)
  >
  > Total model size: 5.4454 GB 
  >
  > But this didnt even fit colab gpu [T4 Gpu 16 Gigs of HBM]

  hugging face accelerate [mostly for inference]

  ![image-20240926110908901](/home/adi/.config/Typora/typora-user-images/image-20240926110908901.png)

  > allows certain layers to be :
  >
  > * cpu offload
  > * disk offload
  >
  > so i thought this is great i can train in full resolution in fp32. if i fit 1 layer at a time albeit it would have been slow

  > But i couldnt
  >
  > because we cant because we need 4 times the layer size to train the layer.
  >
  > * 1 layer
  > * 1 gradients
  > * 2 optimizer states (adam also holds previous gradients)

  * LORA training

    ![image-20240926111358509](/home/adi/.config/Typora/typora-user-images/image-20240926111358509.png)

    ```
    We propose Low-Rank Adaptation, or LoRA, which freezes the pre-
    trained model weights and injects trainable rank decomposition matrices into each
    layer of the Transformer architecture, greatly reducing the number of trainable pa-
    rameters for downstream tasks. Compared to GPT-3 175B fine-tuned with Adam,
    LoRA can reduce the number of trainable parameters by 10,000 times and the
    GPU memory requirement by 3 times. LoRA performs on-par or better than fine-
    tuning in model quality on RoBERTa, DeBERTa, GPT-2, and GPT-3, despite hav-
    ing fewer trainable parameters, a higher training throughput, and, unlike adapters,
    no additional inference latency.
    --  From Abstract
    ```

    ![image-20240926111603359](/home/adi/.config/Typora/typora-user-images/image-20240926111603359.png)

    > so now we only have to store 4 times of dxr and rxd which is smaller than 4 times of dxd
    >
    > as r <<< d

> And here is how it compares to finetuning :
> ![image-20240926112510390](/home/adi/.config/Typora/typora-user-images/image-20240926112510390.png)

> ![image-20240926112205487](/home/adi/.config/Typora/typora-user-images/image-20240926112205487.png)
>
> Total quantized model size: 2.1095 GB
>
> And i was able to fit this even on my gpu



## This week

* 1 epoch pretrain [experimentation with unreleased hyperparameters]

> ![image-20240926112332528](/home/adi/.config/Typora/typora-user-images/image-20240926112332528.png)
>
> I missed a detail on gradient accumulation ?

* clipping [with no accumulation] at 1 killed all the gradients ...

  ![image-20240926112913239](/home/adi/.config/Typora/typora-user-images/image-20240926112913239.png)

## Quick Demo

* i wrote a quick inference on how we will load in a checkpoint.
* 
