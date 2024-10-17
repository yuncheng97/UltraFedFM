DATASET='toy_multi_segmentation' # dataset name
CUDA_VISIBLE_DEVICES='1' python main_multi_segmentation.py \
    --datapath ./dataset/${DATASET}/ \
    --savepath ./output_dir/${DATASET} \
    --batch_size 32 \
    --nb_classes 3 \
    --note vit_b_ssl_usffm \
    --resume ./output_dir/toy_multi_segmentation/vit_b_ssl_usffm/log_2024-07-20_20:56:42/epoch_bestDice.pth \
    --plot
