
# finetune
dataset=toy_diagnosis
CUDA_VISIBLE_DEVICES=1 \
python main_diagnosis.py \
        --model vit_base_patch16 \
        --batch_size 32 \
        --nb_classes 8 \
        --data_path ./dataset/${dataset} \
        --resume ./output_dir/toy_diagnosis/vit_b_ssl_usffm/log_2024-09-28_16:37:08/checkpoint-best_acc.pth \
        --eval \