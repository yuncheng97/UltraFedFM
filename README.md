# UltraFedFM
## [Privacy-Preserving Federated Foundation Model for Generalist Ultrasound Artificial Intelligence]()

UltraFedFM is an innovative privacy-preserving foundation model collaboratively pre-trained via federated learning with multi-modal & multi-center & multi-organ ultrasound datasets, aiming for efficient adaptation on downstream diagnosis and segmentation tasks.

### :sparkles: A Quick Overview
![framework](./asset/UltraFedFM.png) 

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

### :file_folder: Prepare dataset
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
The "client_n.txt" file contain the file paths of all images from one client.

The dataset for evaluation can be downloaded here:


### :wrench: Generate augmented dataset
Generate the augmented dataset by *adaptive scanning model augmentation*, Please noted that you need to specify each image as "linear" or "convex". The generated images should put in the ./Pretrain/train/ folder. Then the image paths of the augmented client dataset should add in the "client_n.txt" file of the original client dataset.

```shell
python util/scan_mode_convert.py
```

### :zap: Quick start with checkpoints
We provid pretrained and finetuned checkpoints of UltraFedFM (Dropbox), which can be put in the output_dir/ folder for further development. 


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


