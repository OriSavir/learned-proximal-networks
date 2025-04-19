for NOISE in 0.1; do
    python lpn/train.py \
    --exp_dir exps/celeba/models/ne_lpn_64ch_sgd/s=${NOISE} \
    --dataset_config_path exps/celeba/configs/dataset.json \
    --model_config_path exps/celeba/configs/model_ne.json \
    --train_batch_size 64 \
    --dataloader_num_workers 8 \
    --num_steps 34000 \
    --num_steps_pretrain 20000 \
    --pretrain_lr 1e-3 \
    --lr 1e-4 \
    --num_stages 4 \
    --save_every_n_steps 1000 \
    --validate_every_n_steps 1000 \
    --image_size 128 \
    --num_channels 3 \
    --optimizer sgd \
    --sigma_noise ${NOISE}
done
