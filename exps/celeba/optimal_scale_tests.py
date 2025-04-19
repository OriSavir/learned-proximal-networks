import numpy as np
from omegaconf import OmegaConf
from lpn.inverse_celeba import main_celeba
from tqdm import tqdm
import os

# Optimal scale values
optimal_scales = {
    "NE": {
        0.02: 2.0,
        0.05: 0.3419951893353393,
        0.1: 0.10536102768906645,
        0.2: 0.058480354764257315,
        0.3: 0.058480354764257315,
        0.5: 0.0324593634702017
    },
    "LPN": {
        0.02: 2.0,
        0.05: 0.3419951893353393,
        0.1: 0.10536102768906645,
        0.2: 0.0324593634702017,
        0.3: 0.01801648230654411,
        0.5: 0.01
    },
    "LPN_64ch": {
        0.02: 2.0,
        0.05: 0.3419951893353393,
        0.1: 0.10536102768906645,
        0.2: 0.0324593634702017,
        0.3: 0.01801648230654411,
        0.5: 0.01
    }
}

# Test configuration
sigma_blur = 2.0
sigma_noise_values = [0.05, 0.1, 0.3, 0.5]

# Containers
results_dict = {
    "NE": {"PSNR": [], "SSIM": []},
    "LPN": {"PSNR": [], "SSIM": []},
    "LPN_64ch": {"PSNR": [], "SSIM": []}
}

# Run tests
for sigma_noise in tqdm(sigma_noise_values):
    for model_type in ["NE", "LPN", "LPN_64ch"]:
        args = OmegaConf.create()

        args.dataset_config = OmegaConf.create()
        args.dataset_config.dataset = "celeba"
        args.dataset_config.root = "data/celeba"
        args.dataset_config.start_idx = 100
        args.dataset_config.num_imgs = 20

        args.operator_config = OmegaConf.create()
        args.operator_config.operator = "blur"
        args.operator_config.sigma_blur = sigma_blur
        args.operator_config.image_size = 128
        args.sigma_noise = sigma_noise

        args.prox_config = OmegaConf.create()
        args.prox_config.prox = "lpn"
        if model_type == "NE":
            args.prox_config.model_path = "exps/celeba/models/ne_lpn_64ch/s=0.1/model.pt"
        elif model_type == "LPN":
            args.prox_config.model_path = "exps/celeba/models/lpn/s=0.1/model.pt"
        else:  # LPN_64ch
            args.prox_config.model_path = "exps/celeba/models/lpn_64ch/s=0.1/model.pt"

        args.admm_config = OmegaConf.create()
        args.admm_config.rho = 0.1
        args.admm_config.maxiter = 20
        args.admm_config.x0 = "adjoint"
        args.admm_config.scale = optimal_scales[model_type][sigma_noise]
        args.admm_config.order = "132"

        args.model_config = OmegaConf.create()
        args.model_config.params = OmegaConf.create()
        args.model_config.params.in_dim = 3
        args.model_config.params.hidden = 64 if model_type != "LPN" else 256
        args.model_config.params.beta = 100
        args.model_config.params.alpha = 1e-6
        args.model_config.model = "ne_128" if model_type == "NE" else "lpn_128"

        args.seed = 0
        args.measure = False
        args.solver = "admm"
        args.data_dir = None

        args.out_dir = f"exps/celeba/optimal_scale_results/{model_type}/deblur/blur={sigma_blur}_noise={sigma_noise}/admm"

        print(f"\n[{model_type}] blur={sigma_blur}, noise={sigma_noise}, scale={args.admm_config.scale}")
        results = main_celeba(args)

        results_dict[model_type]["PSNR"].append((results["PSNR_mean"], results["PSNR_std"]))
        results_dict[model_type]["SSIM"].append((results["SSIM_mean"], results["SSIM_std"]))

# Save results
save_base = "exps/celeba/optimal_scale_results"
for model_type in results_dict:
    np.save(os.path.join(save_base, f"{model_type}_PSNRs.npy"), np.array(results_dict[model_type]["PSNR"], dtype=np.float64))
    np.save(os.path.join(save_base, f"{model_type}_SSIMs.npy"), np.array(results_dict[model_type]["SSIM"], dtype=np.float64))
