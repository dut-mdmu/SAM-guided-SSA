# SAS: SAM-ASSISTED SEMANTIC LEARNING FOR SURGICAL SKILL ASSESSMENT

## 1. Abstract

​	Surgical skill assessment is essential for training and evaluation, yet video-based methods often fail to concentrate on instruments and operative fields, limiting accuracy in complex scenes. We propose SAS that couples SAM-assisted surgical semantics with mask-guided context–semantic fusion. First, a subset of videos is annotated to fine-tune SAM2, and CLIP text prompts generate reliable masks of instruments and the operative field with minimal supervision. These masks guide feature extraction via masked normalization and dual attentions, producing task-aware fused representations that better capture instrument–tissue interactions. The fused features are then processed by a temporal modeling and regression head to predict skill scores, and the prompt set can be readily extended to additional tools. Experiments on JIGSAWS show consistent gains over strong baselines across most tasks and metrics, with ablations confirming the contributions of both components. 

## 2. Code Running

### SAM2 Dataset

​-**Path:** `SAM2_Surgical/datasets`

​-**Download:** https://drive.google.com/file/d/1G8C2s1SI6z0QM9IQLWkI19dSuoW2roYN/view?usp=drive_link

### SAM2 Model & Checkpoints

​-**Path:** `Skill Assessment DatasetSAM2_Surgical/...`

​-**Contents:** SAM2 model implementation & Fine-tuned checkpoints

​-**Download:** https://drive.google.com/file/d/1iiLUeqbMsyresXPHtt4bdiv7PDytUqK0/view?usp=drive_link

### Skill Assessment Dataset

​-**Path:** `SAM2_Surgical/data` 

​-**Download:** https://drive.google.com/file/d/1iW-2yUxpMxBpJ6K7sIiTbyASP6bpOm05/view?usp=drive_link

## **3. Notes**

​-Ensure all datasets are placed in the correct directory structure before running the code.

​-Pretrained weights are required for both SAM2 and the skill assessment model.

​-The framework is modular and can be extended to additional surgical tools by modifying text prompts.


