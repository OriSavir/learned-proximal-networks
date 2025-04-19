import numpy as np
from omegaconf import OmegaConf
from lpn.inverse_celeba import main_celeba
from tqdm import tqdm
import os

# Sweep settings
sigma_blur = 2.0
sigma_noise_values = [0.1, 0.3, 0.5]
constant_scale = 0.5

# Initialize result containers
NE_PSNRs, NE_SSIMs = [], []
LPN_PSNRs, LPN_SSIMs = [], []
LPN_small_PSNRs, LPN_small_SSIMs = [], []

# Run tests for each sigma_noise
for sigma_noise in tqdm(sigma_noise_values):
    for model_type in ['NE', 'LPN', 'LPN_64ch']:
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
        if model_type == 'NE':
            args.prox_config.model_path = "exps/celeba/models/ne_lpn_64ch/s=0.1/model.pt"
        elif model_type == 'LPN':
            args.prox_config.model_path = "exps/celeba/models/lpn/s=0.1/model.pt"
        elif model_type == 'LPN_64ch':
            args.prox_config.model_path = "exps/celeba/models/lpn_64ch/s=0.1/model.pt"

        args.admm_config = OmegaConf.create()
        args.admm_config.rho = 0.1
        args.admm_config.maxiter = 20
        args.admm_config.x0 = "adjoint"
        args.admm_config.scale = constant_scale
        args.admm_config.order = "132"

        args.model_config = OmegaConf.create()
        if model_type == 'NE':
            args.model_config.model = "ne_128"
            args.model_config.params = OmegaConf.create()
            args.model_config.params.in_dim = 3
            args.model_config.params.hidden = 64
            args.model_config.params.beta = 100
            args.model_config.params.alpha = 1e-6
        elif model_type == 'LPN':
            args.model_config.model = "lpn_128"
            args.model_config.params = OmegaConf.create()
            args.model_config.params.in_dim = 3
            args.model_config.params.hidden = 256
            args.model_config.params.beta = 100
            args.model_config.params.alpha = 1e-6
        elif model_type == 'LPN_64ch':
            args.model_config.model = "lpn_128"
            args.model_config.params = OmegaConf.create()
            args.model_config.params.in_dim = 3
            args.model_config.params.hidden = 64
            args.model_config.params.beta = 100
            args.model_config.params.alpha = 1e-6

        args.seed = 0
        args.out_dir = f"exps/celeba/constant_scale_results/{model_type}/deblur/blur={sigma_blur}_noise={sigma_noise}/admm"
        args.measure = False
        args.solver = "admm"
        args.data_dir = None

        print(f"\n[{model_type}] blur={sigma_blur}, noise={sigma_noise}, scale={constant_scale}")
        results = main_celeba(args)

        if model_type == 'NE':
            NE_PSNRs.append((results["PSNR_mean"], results["PSNR_std"]))
            NE_SSIMs.append((results["SSIM_mean"], results["SSIM_std"]))
        elif model_type == 'LPN':
            LPN_PSNRs.append((results["PSNR_mean"], results["PSNR_std"]))
            LPN_SSIMs.append((results["SSIM_mean"], results["SSIM_std"]))
        else:
            LPN_small_PSNRs.append((results["PSNR_mean"], results["PSNR_std"]))
            LPN_small_SSIMs.append((results["SSIM_mean"], results["SSIM_std"]))

# Save results as NumPy arrays
np.save("exps/celeba/constant_scale_results/NE_PSNRs.npy", np.array(NE_PSNRs, dtype=np.float64))
np.save("exps/celeba/constant_scale_results/LPN_PSNRs.npy", np.array(LPN_PSNRs, dtype=np.float64))
np.save("exps/celeba/constant_scale_results/LPN_small_PSNRs.npy", np.array(LPN_small_PSNRs, dtype=np.float64))

np.save("exps/celeba/constant_scale_results/NE_SSIMs.npy", np.array(NE_SSIMs, dtype=np.float64))
np.save("exps/celeba/constant_scale_results/LPN_SSIMs.npy", np.array(LPN_SSIMs, dtype=np.float64))
np.save("exps/celeba/constant_scale_results/LPN_small_SSIMs.npy", np.array(LPN_small_SSIMs, dtype=np.float64))
