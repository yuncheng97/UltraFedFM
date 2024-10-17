FED_MODEL='ultrafedfm'

DATASET='US_Pretrain' # dataset name
DATA_PATH="/220019054/Dataset/${DATASET}/" # change DATA_PATH to the path where the data were stored

SPLIT_TYPE='split_3' # chosen from {'central', 'split_1', 'split_2', 'split_3'}
N_CLIENTS=10 # number of clients in the federated setting
MASK_RATIO=0.75 # masking ratio for Fed-MAE pre-training
N_GPUS=4 # the number of GPUs used for model training

# ------------------ UltraFedFM pretraining ----------------- #
# you can directly use the saved pre-trained checkpoints from our github and skip this step

EPOCHS=600
BLR='1.5e-4'
BATCH_SIZE=16
# change OUTPUT_PATH to your path where the pre-trained checkpoints will be stored
OUTPUT_PATH="./output_dir/${FED_MODEL}/pretrained_epoch${EPOCHS}_${SPLIT_TYPE}_blr${BLR}_bs${BATCH_SIZE}_ratio${MASK_RATIO}_dis${N_GPUS}"

# change the CUDA devices available for model training
CUDA_VISIBLE_DEVICES='0,1,2,3' OMP_NUM_THREADS=1 python -m torch.distributed.launch --nproc_per_node=${N_GPUS} --use_env main_pretrain.py \
        --data_path ${DATA_PATH} \
        --data_set ${DATASET} \
        --output_dir ${OUTPUT_PATH} \
        --blr ${BLR} \
        --batch_size ${BATCH_SIZE} \
        --save_ckpt_freq 10 \
        --max_communication_rounds ${EPOCHS} \
        --epochs ${EPOCHS} \
        --split_type ${SPLIT_TYPE} \
        --mask_ratio ${MASK_RATIO} \
        --model mae_vit_base_patch16 \
        --warmup_epochs 40 \
        --weight_decay 0.05 \
        --norm_pix_loss --sync_bn \
        --n_clients ${N_CLIENTS} --E_epoch 1  --num_local_clients -1 \
        # > pretrain_fed.out 2>&1 &

