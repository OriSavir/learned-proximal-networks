
for perturb in convex blur gaussian; do
    python lpn/evaluate_prior.py \
    --model_config_path exps/mnist/configs/mnist/model_old_ne.json \
    --out_dir exps/mnist/experiments/old_ne_mnist/prior \
    --perturb_config_path exps/mnist/configs/mnist/prior/perturb/${perturb}.json \
    --dataset_config_path exps/mnist/configs/mnist/prior/dataset.json \
    --model_path exps/mnist/experiments/old_ne_mnist/model.pt \
    --inv_alg cvx_cg 
done
