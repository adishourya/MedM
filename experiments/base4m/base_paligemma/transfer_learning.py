# we will write a training loop ... we will unfreeze some of the layers as per the paper

# below are the
# The result of Stages 1 and 2 is a family of three PaliGemma checkpoints, at
# 224px, 448px, and 896px resolution, which are pre-equipped with broad visual
# knowledge. However, these check- points are not “user (or benchmark) friendly”
# as their pretraining has focused solely on density of learning signal, as
# opposed to usable interface. These base models need to be transferred to serve
# their intended final purpose. That could take the form of fine-tuning on a
# specific, spe- cialized task, such as COCO Captions, Remote Sensing VQA, Video
# Captioning, or Infograph- icQA. Adapt to new inputs such as multiple im- ages
# (NLVR2) or bounding boxes draw in the image (WidgetCap). Or it could take the
# form of instruction [67] or even chat [44] tuning. To show the effectiveness of
# the base mod- els, we transfer them to a wide range of indi- vidual academic
# benchmarks, using a simple uni- fied transfer recipe with few hyper-parameters.
# And to showcase the versatility beyond academic tasks, we also provide a “mix”
# transfer checkpoint, which transfers to a subset of these tasks at the same
# time, along with detailed captioning and long question-answering data. While
# this is not instruction tuning, it is a step in that direction. We also
# transfer PaliGemma to tasks which take multiple images as input. NLVR2 is one
# such task, which asks one question about two images, and requires looking at
# both to give the correct an- swer. Other such tasks are standard short-video
# understanding tasks subsampled to 16 frames. In all these cases, we follow
# PaLI-3 and encode each image separately, then concatenate the image tokens
# without any special separator or embed- ding tokens. Thus, 16 frames at 224px
# resolution result in 𝑁img = 4096 image tokens, the same amount as a single
# image at 896px resolution. For all transfers, we perform fine-tuning of all the
# model parameters. The hyper-parameters we modify per-task are the following, in
# decreasing order of importance: • Resolution (i.e. checkpoint): 224, 448, 896.
# Epochs: 1, 3, 10, 30, 100. Learning-rate: 3e-5, 1e-5, 3e-6. Label-smoothing:
# 0.0, 0.1, 0.3. Dropout in the LLM: 0.0, 0.1, 0.3. Weight decay: 0.0 or 0.1 ×
# learning-rate. Freeze ViT: false, true. Beam-search may benefit captioning. The
# above are typical values we suggest exploring, with the recommended initial
# attempt value in bold. We provide the best setting for each individ- ual task
# in Appendix J. We study the sensitivity to transfer hyper-parameters in Section
# 6.2, and the “transferability” in general in Section 6, show- ing that good
# results can be achieved with the aforementioned initial attempt values.
