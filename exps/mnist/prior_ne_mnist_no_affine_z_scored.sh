
for perturb in convex blur gaussian; do
    python lpn/evaluate_prior.py \
    --model_config_path exps/mnist/configs/mnist/model_ne_mnist_no_affine_z_scored.json \
    --out_dir exps/mnist/experiments/ne_mnist/prior \
    --perturb_config_path exps/mnist/configs/mnist/prior/perturb/${perturb}.json \
    --dataset_config_path exps/mnist/configs/mnist/prior/dataset.json \
    --model_path exps/mnist/experiments/ne_mnist_no_affine_z_scored/model.pt \
    --inv_alg cvx_cg 
done
