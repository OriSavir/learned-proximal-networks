#!/bin/bash
GPU_ID=${1:-0}

for NOISE in 0.1; do
    CUDA_VISIBLE_DEVICES=$GPU_ID python lpn/train.py \
    --exp_dir exps/bsd/models/ordinary_drunet/s=${NOISE} \
    --dataset_config_path exps/bsd/configs/dataset.json \
    --model_config_path exps/bsd/configs/model_ordinary_drunet.json \
    --train_batch_size 8 \
    --dataloader_num_workers 8 \
    --num_steps 40000 \
    --num_steps_pretrain 40000 \
    --pretrain_lr 1e-4 \
    --lr 1e-4 \
    --num_stages 4 \
    --save_every_n_steps 1000 \
    --validate_every_n_steps 1000 \
    --image_size 128 \
    --num_channels 3 \
    --sigma_noise ${NOISE}
done
