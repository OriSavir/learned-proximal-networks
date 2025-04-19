import json
import numpy as np
from omegaconf import OmegaConf
from lpn.inverse_celeba import main_celeba
from tqdm import tqdm

# Define sigma_noise values to test
sigma_noise_values = [0.02, 0.05, 0.1, 0.2, 0.3, 0.5]

# Define a range of sigma_blur values to test
sigma_blur = 2.0

scale_range = np.logspace(-2, np.log10(2), 10)

# Initialize dictionaries to store the optimal blur values
optimal_scale_LPN = {}
optimal_scale_NE = {}

# Iterate over each noise level
for sigma_noise in tqdm(sigma_noise_values):
    
    best_psnr_NE, best_psnr_LPN = -np.inf, -np.inf
    best_scale_NE, best_scale_LPN = None, None

    for scale in scale_range:
        # Update parameters for NE model
        args = OmegaConf.create()
        args.dataset_config = OmegaConf.create()
        args.dataset_config.dataset = "celeba"
        args.dataset_config.root = "data/celeba"
        args.dataset_config.start_idx = 79
        args.dataset_config.num_imgs = 20

        args.operator_config = OmegaConf.create()
        args.operator_config.operator = "blur"
        args.operator_config.sigma_blur = float(sigma_blur)
        args.operator_config.image_size = 128
        args.sigma_noise = sigma_noise

        args.prox_config = OmegaConf.create()
        args.prox_config.prox = "lpn"
        args.prox_config.model_path = "exps/celeba/models/ne_lpn_64ch/s=0.1/model.pt"

        args.admm_config = OmegaConf.create()
        args.admm_config.rho = 0.1
        args.admm_config.maxiter = 20
        args.admm_config.x0 = "adjoint"
        args.admm_config.scale = float(scale)
        args.admm_config.order = "132"

        args.model_config = OmegaConf.create()
        args.model_config.model = "ne_128"
        args.model_config.params = OmegaConf.create()
        args.model_config.params.in_dim = 3
        args.model_config.params.hidden = 64
        args.model_config.params.beta = 100
        args.model_config.params.alpha = 1e-6

        args.seed = 0
        args.out_dir = f"exps/celeba/scale_find_ne_results/inverse/deblur/blur={float(sigma_blur)}_noise={sigma_noise}/admm"
        args.measure = False
        args.solver = "admm"
        args.data_dir = None

        NE_results = main_celeba(args)

        if NE_results["PSNR_mean"] > best_psnr_NE:
            best_psnr_NE = NE_results["PSNR_mean"]
            best_scale_NE = scale

        args_lpn = OmegaConf.create()
        args_lpn.dataset_config = OmegaConf.create()
        args_lpn.dataset_config.dataset = "celeba"
        args_lpn.dataset_config.root = "data/celeba"
        args_lpn.dataset_config.start_idx = 79
        args_lpn.dataset_config.num_imgs = 20

        args_lpn.operator_config = OmegaConf.create()
        args_lpn.operator_config.operator = "blur"
        args_lpn.operator_config.sigma_blur = float(sigma_blur)
        args_lpn.operator_config.image_size = 128
        args_lpn.sigma_noise = sigma_noise

        args_lpn.prox_config = OmegaConf.create()
        args_lpn.prox_config.prox = "lpn"
        args_lpn.prox_config.model_path = "exps/celeba/models/lpn/s=0.1/model.pt"

        args_lpn.admm_config = OmegaConf.create()
        args_lpn.admm_config.rho = 0.1
        args_lpn.admm_config.maxiter = 20
        args_lpn.admm_config.x0 = "adjoint"
        args_lpn.admm_config.scale = float(scale)
        args_lpn.admm_config.order = "132"

        args_lpn.model_config = OmegaConf.create()
        args_lpn.model_config.model = "lpn_128"
        args_lpn.model_config.params = OmegaConf.create()
        args_lpn.model_config.params.in_dim = 3
        args_lpn.model_config.params.hidden = 256
        args_lpn.model_config.params.beta = 100
        args_lpn.model_config.params.alpha = 1e-6

        args_lpn.seed = 0
        args_lpn.out_dir = f"exps/celeba/scale_find_lpn_results/inverse/deblur/blur={float(sigma_blur)}_noise={sigma_noise}/admm"
        args_lpn.measure = False
        args_lpn.solver = "admm"
        args_lpn.data_dir = None

        LPN_results = main_celeba(args_lpn)

        if LPN_results["PSNR_mean"] > best_psnr_LPN:
            best_psnr_LPN = LPN_results["PSNR_mean"]
            best_scale_LPN = scale

    optimal_scale_NE[sigma_noise] = best_scale_NE
    optimal_scale_LPN[sigma_noise] = best_scale_LPN

with open("optimal_scale_NE.json", "w") as f:
    json.dump(optimal_scale_NE, f, indent=4)

with open("optimal_scale_LPN.json", "w") as f:
    json.dump(optimal_scale_LPN, f, indent=4)

print("Optimal scale values saved to JSON files.")
