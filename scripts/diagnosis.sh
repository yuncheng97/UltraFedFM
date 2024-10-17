
# finetune
dataset=toy_diagnosis
CUDA_VISIBLE_DEVICES=1 python main_diagnosis.py \
        --model vit_base_patch16 \
        --note ${dataset}/vit_b_ssl_usffm \
        --batch_size 32 \
        --epochs 2 \
        --nb_classes 8 \
        --data_path ./dataset/${dataset} \
        --finetune ./output_dir/pretrained_ultrafedfm/log_2024-07-16_13:53:08/checkpoint.pth
