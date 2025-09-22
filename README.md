## RETFound - A foundation model for retinal images


Official repo including a series of foundation models and applications for retinal images.<br>
`[RETFound-MAE]`:[RETFound: a foundation model for generalizable disease detection from retinal images](https://www.nature.com/articles/s41586-023-06555-x).<br>
`[RETFound-DINOv2]`:[Revealing the Impact of Pre-training Data on Medical Foundation Models](https://www.researchsquare.com/article/rs-6080254/v1).<br>
`[DINOv2]`:[General-purpose vision foundation models DINOv2 by Meta](https://github.com/facebookresearch/dinov2).<br>
`[DINOv3]`:[General-purpose vision foundation models DINOv3 by Meta](https://github.com/facebookresearch/dinov3).<br>


Please contact 	**ykzhoua@gmail.com** or **yukun.zhou.19@ucl.ac.uk** if you have questions.


### 📝Key features

- RETFound is pre-trained on 1.6 million retinal images with self-supervised learning
- RETFound has been validated in multiple disease detection tasks
- RETFound can be efficiently adapted to customised tasks


### 🎉News

- 🐉2025/09: **Preprint benchmarking DINOv3, DINOv2, and RETFound is [available](https://arxiv.org/abs/2509.03421)!**
- 🐉2025/09: **We included state-of-the-art DINOv3 into fine-tuning pipeline for retinal applications!**
- 🐉2025/02: **We organised the model weights on HuggingFace, no more manual downloads needed!**
- 🐉2025/02: **Multiple [pre-trained weights](https://huggingface.co/YukunZhou), including MAE-based and DINOV2-based, are added!**
- 🐉2025/02: **We update the version of packages, such as CUDA12+ and PyTorch 2.3+!**
- 🐉2024/01: [Feature vector notebook](https://github.com/rmaphoh/RETFound_MAE/blob/main/latent_feature.ipynb) are now online!
- 🐉2024/01: [Data split and model checkpoints](BENCHMARK.md) for public datasets are now online!
- 🎄2023/12: [Colab notebook](https://colab.research.google.com/drive/1_X19zdMegmAlqPAEY0Ao659fzzzlx2IZ?usp=sharing) is now online - free GPU & simple operation!


### 🔧Install environment

1. Create environment with conda:

```
conda create -n retfound python=3.11.0 -y
conda activate retfound
```

2. Install dependencies

```
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
git clone https://github.com/rmaphoh/RETFound/
cd RETFound
pip install -r requirements.txt
```


### 🌱Fine-tuning with RETFound weights

1. Get access to the pre-trained models on HuggingFace (register an account and fill in the form) and go to step 2:
<table><tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<th valign="bottom"></th>
<th valign="bottom">ViT-Large</th>
<th valign="bottom">Source</th>
<!-- TABLE BODY -->
<tr><td align="left">RETFound_mae_natureCFP</td>
<td align="center"><a href="https://huggingface.co/YukunZhou/RETFound_mae_natureCFP">access</a></td>
<td align="center"><a href="https://www.nature.com/articles/s41586-023-06555-x">Nature RETFound paper</a></td>
</tr>
<!-- TABLE BODY -->
<tr><td align="left">RETFound_mae_natureOCT</td>
<td align="center"><a href="https://huggingface.co/YukunZhou/RETFound_mae_natureOCT">access</a></td>
<td align="center"><a href="https://www.nature.com/articles/s41586-023-06555-x">Nature RETFound paper</a></td>
</tr>
<!-- TABLE BODY -->
<tr><td align="left">RETFound_mae_meh</td>
<td align="center"><a href="https://huggingface.co/YukunZhou/RETFound_mae_meh">access</a></td>
<td align="center"><a href="https://www.researchsquare.com/article/rs-6080254/v1">FM data paper</a></td>
</tr>
<!-- TABLE BODY -->
<tr><td align="left">RETFound_mae_shanghai</td>
<td align="center"><a href="https://huggingface.co/YukunZhou/RETFound_mae_shanghai">access</a></td>
<td align="center"><a href="https://www.researchsquare.com/article/rs-6080254/v1">FM data paper</a></td>
</tr>
<!-- TABLE BODY -->
<tr><td align="left">RETFound_dinov2_meh</td>
<td align="center"><a href="https://huggingface.co/YukunZhou/RETFound_dinov2_meh">access</a></td>
<td align="center"><a href="https://www.researchsquare.com/article/rs-6080254/v1">FM data paper</a></td>
</tr>
<!-- TABLE BODY -->
<tr><td align="left">RETFound_dinov2_shanghai</td>
<td align="center"><a href="https://huggingface.co/YukunZhou/RETFound_dinov2_shanghai">access</a></td>
<td align="center"><a href="https://www.researchsquare.com/article/rs-6080254/v1">FM data paper</a></td>
</tr>
</tbody></table>

2. Login in your HuggingFace account, where HuggingFace token can be [created and copied](https://huggingface.co/settings/tokens).
```
huggingface-cli login --token YOUR_HUGGINGFACE_TOKEN
```

**Optional**: if your machine and server cannot access HuggingFace due to internet wall, run the command below (Do not run it if you can access):
```
export HF_ENDPOINT=https://hf-mirror.com
```

3. If you would like to fine-tune [DINOv2](https://github.com/facebookresearch/dinov2) and [DINOv3](https://github.com/facebookresearch/dinov3), please visit their GitHub repositories to download the model weights and put them in the RETFound folder.

4. Organise your data into this directory structure (Public datasets used in this study can be [downloaded here](BENCHMARK.md))

```
├── data folder
    ├──train
        ├──class_a
        ├──class_b
        ├──class_c
    ├──val
        ├──class_a
        ├──class_b
        ├──class_c
    ├──test
        ├──class_a
        ├──class_b
        ├──class_c
``` 



5. Start fine-tuning by running `sh train.sh`.


In `train.sh`, the model can be selected by changing the hyperparameters `MODEL`, `MODEL_ARCH`, `FINETUNE`:

**RETFound**:

| MODEL           | MODEL_ARCH               | FINETUNE                 | SIZE                     |
|-----------------|--------------------------|--------------------------|--------------------------|
| RETFound_mae    | retfound_mae             | RETFound_mae_natureCFP   | ~300M                    |
| RETFound_mae    | retfound_mae             | RETFound_mae_natureOCT   | ~300M                    |
| RETFound_mae    | retfound_mae             | RETFound_mae_meh         | ~300M                    |
| RETFound_mae    | retfound_mae             | RETFound_mae_shanghai    | ~300M                    |
| RETFound_dinov2 | retfound_dinov2          | RETFound_dinov2_meh      | ~300M                    |
| RETFound_dinov2 | retfound_dinov2          | RETFound_dinov2_shanghai | ~300M                    |


**DINOv3**:

| MODEL           | MODEL_ARCH               | FINETUNE                         | SIZE                     |
|-----------------|--------------------------|----------------------------------|--------------------------|
| Dinov3          | dinov3_vits16            | dinov3_vits16_pretrain.pth       | ~21M                     |
| Dinov3          | dinov3_vits16plus        | dinov3_vits16plus_pretrain.pth   | ~29M                     |
| Dinov3          | dinov3_vitb16            | dinov3_vitb16_pretrain.pth       | ~86M                     |
| Dinov3          | dinov3_vitl16            | dinov3_vitl16_pretrain.pth       | ~300M                    |
| Dinov3          | dinov3_vith16plus        | dinov3_vith16plus_pretrain.pth   | ~840M                    |
| Dinov3          | dinov3_vit7b16           | dinov3_vit7b16_pretrain.pth      | ~6.7B                    |


**DINOv2**:

| MODEL           | MODEL_ARCH               | FINETUNE                     | SIZE                     |
|-----------------|--------------------------|------------------------------|--------------------------|
| Dinov2          | dinov2_vits14            | dinov2_vits14_pretrain.pth   | ~21M                     |
| Dinov2          | dinov2_vitb14            | dinov2_vitb14_pretrain.pth   | ~86M                     |
| Dinov2          | dinov2_vitl14            | dinov2_vitl14_pretrain.pth   | ~300M                    |
| Dinov2          | dinov2_vitg14            | dinov2_vitg14_pretrain.pth   | ~1.1B                    |

### 🧠 Multi-task fine-tuning with Excel annotations

For datasets where all images live inside a single folder and the task labels are stored in an Excel sheet, you can fine-tune
RETFound for multiple related predictions with the `multitask_finetune.py` script introduced in this repository.

1. Place all images inside a directory such as `./TRS-Image`.
2. Create an Excel file containing the following columns (one row per image):
   - `image_id`: the file name without extension (e.g. `anhuayi_QD_01`).
   - `diagnosis`: categorical value encoded as 0=正常, 1=PACG, 2=APAC, 3=POAG, 4=继发性G.
   - `md_stage`: 1=≥-6, 2=-6~-12, 3=< -12.
   - `vfi_grade`: 1=80-100%, 2=60-80%, 3=40-60%, 4=20-40%, 5=0-20%.
   - `ght_class`: 0=正常, 1=异常, 2=边缘, 3=普遍敏感性下降.
   If the Excel file omits the image extension, provide it via `--image-ext` (for example `--image-ext .png`).

Run the multi-task trainer:

```bash
python multitask_finetune.py \
  --images-dir ./TRS-Image \
  --annotations labels.xlsx \
  --finetune /path/to/RETFound_mae_natureCFP.pth \
  --output-dir ./multitask_output \
  --batch-size 16 --epochs 50
```

The script splits the data into train/validation/test subsets, fine-tunes RETFound with four classification heads, and produces
`training_history.json`, `best_model.pth`, and detailed evaluation metrics (`test_metrics.json`, `test_predictions.csv`) in the
output directory.


Change the DATA_PATH to your dataset directory.

```
# ==== Model settings ====
# adaptation {finetune,lp}
ADAPTATION="finetune"
MODEL="RETFound_dinov2"
MODEL_ARCH="retfound_dinov2"
FINETUNE="RETFound_dinov2_meh"

# ==== Data settings ====
# change the dataset name and corresponding class number
DATASET="MESSIDOR2"
NUM_CLASS=5

# =======================
DATA_PATH="PATH TO THE DATASET"
TASK="${MODEL_ARCH}_${DATASET}_${ADAPTATION}"

torchrun --nproc_per_node=1 --master_port=48766 main_finetune.py \
  --model "${MODEL}" \
  --model_arch "${MODEL_ARCH}" \
  --finetune "${FINETUNE}" \
  --savemodel \
  --global_pool \
  --batch_size 24 \
  --world_size 1 \
  --epochs 50 \
  --nb_classes "${NUM_CLASS}" \
  --data_path "${DATA_PATH}" \
  --input_size 224 \
  --task "${TASK}" \
  --adaptation "${ADAPTATION}" 

```



6. For evaluation only (download data and model checkpoints [here](BENCHMARK.md); change the DATA_PATH below)


```
# ==== Model/settings (match training) ====
ADAPTATION="finetune"
MODEL="RETFound_dinov2"
MODEL_ARCH="retfound_dinov2"
FINETUNE="RETFound_dinov2_meh"

# ==== Data/settings (match training) ====
DATASET="MESSIDOR2"
NUM_CLASS=5

# =======================
DATA_PATH="PATH TO THE DATASET"
TASK="${MODEL_ARCH}_${DATASET}_${ADAPTATION}"

# Path to the trained checkpoint (adjust if you saved elsewhere)
CKPT="./output_dir/${TASK}/checkpoint-best.pth"

# ==== Evaluation only ====
torchrun --nproc_per_node=1 --master_port=48766 main_finetune.py \
  --model "${MODEL}" \
  --model_arch "${MODEL_ARCH}" \
  --savemodel \
  --global_pool \
  --batch_size 128 \
  --world_size 1 \
  --nb_classes "${NUM_CLASS}" \
  --data_path "${DATA_PATH}" \
  --input_size 224 \
  --task "${TASK}" \
  --adaptation "${ADAPTATION}" \
  --eval \
  --resume "${CKPT}"

```


### 📃Citation

If you find this repository useful, please consider citing this paper:


```
@article{zhou2023foundation,
  title={A foundation model for generalizable disease detection from retinal images},
  author={Zhou, Yukun and Chia, Mark A and Wagner, Siegfried K and Ayhan, Murat S and Williamson, Dominic J and Struyven, Robbert R and Liu, Timing and Xu, Moucheng and Lozano, Mateo G and Woodward-Court, Peter and others},
  journal={Nature},
  volume={622},
  number={7981},
  pages={156--163},
  year={2023},
  publisher={Nature Publishing Group UK London}
}
```

```
@misc{zhou2025generalistversusspecialistvision,
      title={Generalist versus Specialist Vision Foundation Models for Ocular Disease and Oculomics}, 
      author={Yukun Zhou and Paul Nderitu and Jocelyn Hui Lin Goh and Justin Engelmann and Siegfried K. Wagner and Anran Ran and Hongyang Jiang and Lie Ju and Ke Zou and Sahana Srinivasan and Hyunmin Kim and Takahiro Ninomiya and Zheyuan Wang and Gabriel Dawei Yang and Eden Ruffell and Dominic Williamson and Rui Santos and Gabor Mark Somfai and Carol Y. Cheung and Tien Yin Wong and Daniel C. Alexander and Yih Chung Tham and Pearse A. Keane},
      year={2025},
      eprint={2509.03421},
      archivePrefix={arXiv},
      primaryClass={eess.IV},
      url={https://arxiv.org/abs/2509.03421}, 
}
```
