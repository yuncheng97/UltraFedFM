DATASET='toy_segmentation' # dataset name
CUDA_VISIBLE_DEVICES='1' python main_binary_segmentation.py \
    --datapath ./dataset/${DATASET}/ \
    --savepath ./output_dir/${DATASET} \
    --batch_size 96 \
    --epoch 128 \
    --note vit_b_ssl_usffm \
    --pretrained ./output_dir/pretrained_ultrafedfm/log_2024-07-16_13:53:08/checkpoint.pth