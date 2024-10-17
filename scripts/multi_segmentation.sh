DATASET='toy_multi_segmentation' # dataset name
CUDA_VISIBLE_DEVICES='1' python main_multi_segmentation.py \
    --datapath ./dataset/${DATASET}/ \
    --savepath ./output_dir/${DATASET} \
    --batch_size 64 \
    --epoch 32 \
    --nb_classes 3 \
    --note vit_b_ssl_usffm \
    --pretrained ./output_dir/pretrained_ultrafedfm/log_2024-07-16_13:53:08/checkpoint.pth
