DATASET='toy_segmentation' # dataset name
CUDA_VISIBLE_DEVICES='0' python main_binary_segmentation.py \
    --datapath ./dataset/${DATASET}/ \
    --savepath ./output_dir/${DATASET} \
    --batch_size 32 \
    --note vit_b_ssl_usffm \
    --resume ./output_dir/toy_segmentation/vit_b_ssl_usffm/log_2024-07-31_16:18:46/epoch_bestDice.pth \
    --plot
