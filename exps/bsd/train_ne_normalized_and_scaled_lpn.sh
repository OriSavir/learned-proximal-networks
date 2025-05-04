for NOISE in 0.1; do
    python lpn/train.py \
    --exp_dir exps/bsd/models/ne_norm_and_scaled_lpn/s=${NOISE} \
    --dataset_config_path exps/bsd/configs/dataset.json \
    --model_config_path exps/bsd/configs/model_ne_normalized_and_scaled.json \
    --train_batch_size 64 \
    --dataloader_num_workers 8 \
    --num_steps 36000 \
    --num_steps_pretrain 5000 \
    --pretrain_lr 1e-3 \
    --lr 1e-4 \
    --num_stages 4 \
    --save_every_n_steps 1000 \
    --validate_every_n_steps 1000 \
    --image_size 128 \
    --num_channels 3 \
    --sigma_noise ${NOISE} \
    --disable_sigma_schedule \
    --disable_sigma_schedule_after 14900
done
