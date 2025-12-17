# UltraFedFM
## [From pretraining to privacy: federated ultrasound foundation model with self-supervised learning]()


[![Arxiv](https://img.shields.io/badge/arXiv-2305.17100-B21A1B)](https://arxiv.org/abs/2411.16380)


UltraFedFM is a privacy-preserving federated foundation model designed for generalist ultrasound intelligence. It enables collaborative pre-training across multiple institutions without data sharing, and supports efficient adaptation to diverse downstream ultrasound tasks including diagnosis and segmentation.
### :sparkles: A Quick Overview
![framework](./asset/UltraFedFM.png) 

🔑 **Key Features**

🔒 **Federated Learning for Privacy Preservation**

Multi-center collaborative pre-training without sharing raw ultrasound data.

🧠 **Foundation Model for Ultrasound**

A generalist backbone supporting multi-organ, multi-task, and multi-modal ultrasound applications.

⚡**Efficient Downstream Adaptation**

Ready-to-use fine-tuning and evaluation pipelines for diagnosis and segmentation tasks.

### :mag: Installation
1. Clone this repository and navigate to the BiomedGPT folder
```shell
# clone project
git clone https://github.com/yuncheng97/UltraFedFM.git
cd UltraFedFM/
```

2. Install required packages
```shell
# create conda environment and install dependencies
conda env create -f environment.yaml
conda activate UltraFedFM
```

### :file_folder: Data Preparation
For Pre-training and fine-tuning stages, download the public datasets provided by the links in "Data availability" section or prepare your own datassets. Then structure the datasets file folders, you can follow the directory setting below: 
```
Dataset/
├── Pretrain/
│   ├── 16_clients/
│       └── split_1
│           └── client_0.txt
│           └── ...
│       └── ...
│   ├── train/
│       └── dataset1
│       └── ...
├── finetune/
│   ├── diagnosis/
│   │   ├── dataset1/
│   │   └── ...
│   ├── segmentation/
│   │   ├── dataset1/
│   │   └── ...
└── ...
```
Each line in `client_n.txt` corresponds to a single ultrasound image.

Images should be stored under `Dataset/Pretrain/train/`.

### :wrench: Generate augmented dataset
Generate the augmented dataset by *adaptive scanning model augmentation*, Please noted that you need to specify each image as "linear" or "convex". The generated images should put in the ./Pretrain/train/ folder. Then the image paths of the augmented client dataset should add in the "client_n.txt" file of the original client dataset.

```shell
python util/scan_mode_convert.py
```

### :zap: Quick start with checkpoints
We provid pretrained and finetuned checkpoints of UltraFedFM ([Onedrive](https://cuhko365-my.sharepoint.com/:f:/g/personal/220019054_link_cuhk_edu_cn/ErEPqzsR_3ZLr3Q18htiG5QBEFYtO0zgMb2OzxNITg6aqw?e=xEE7eB)), which can be put in the output_dir/ folder for further development. 


### :rocket: Pre-training

```shell
bash scripts/pretrain.sh
```

### :mortar_board: Downstreams
We provide the run scripts of fine-tuning and inference. There will be log files during execution. please refer to
<details>
    <summary><b>Ultrasound Image Diagnosis</b></summary>
<pre>
# for fine-tuning
bash scripts/diagnosis.sh
# for inference using fine-tuned weights
bash scripts/eval_diagnosis.sh
</pre>
</details>
<details>
    <summary><b>Binary Ultrasound Image Segmentation</b></summary>
<pre>
# for fine-tuning
bash scripts/binary_segmentation.sh
# for inference using fine-tuned weights
bash scripts/eval_binary_segmentation.sh
# for visualization using fine-tuned weights
bash scripts/plot_binary_segmentation.sh
</pre>
</details>
<details>
    <summary><b>Multi-Class Ultrasound Image Segmentation</b></summary>
<pre>
# for fine-tuning
bash scripts/multi_segmentation.sh
# for inference using fine-tuned weights
bash scripts/eval_multi_segmentation.sh
# for visualization using fine-tuned weights
bash scripts/plot_multi_segmentation.sh
</pre>
</details>

## :pray: Acknowledgement
This code of repository is built on [MAE](https://github.com/facebookresearch/mae) and [SSL-FL](https://github.com/rui-yan/SSL-FL). Thanks for their valuble contributions.

## :book: Citation

```bibtex
@article{jiang2025pretraining,
  title={From pretraining to privacy: federated ultrasound foundation model with self-supervised learning},
  author={Jiang, Yuncheng and Feng, Chun-Mei and Ren, Jinke and Wei, Jun and Zhang, Zixun and Hu, Yiwen and Liu, Yunbi and Sun, Rui and Tang, Xuemei and Du, Juan and others},
  journal={npj Digital Medicine},
  volume={8},
  number={1},
  pages={714},
  year={2025},
  publisher={Nature Publishing Group UK London}
}

