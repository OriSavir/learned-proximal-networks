import matplotlib.pyplot as plt
import numpy as np

from omegaconf import OmegaConf
from pprint import pp

from lpn.inverse_celeba import main_celeba

# set sigma blur values and corresponding sigma noise values
# set admm scale

sigma_blur = [2.0, 2.0, 2.0, 2.0, 2.0, 2.0]
sigma_noise = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
admm_scale = 0.5

# set parameters for NE test manually

args = OmegaConf.create()
args.dataset_config = OmegaConf.create()
args.dataset_config.dataset = "celeba"
args.dataset_config.root = "data/celeba"
args.dataset_config.start_idx = 100
args.dataset_config.num_imgs = 20

args.operator_config = OmegaConf.create()
args.operator_config.operator = "blur"
args.operator_config.sigma_blur = 1.0
args.operator_config.image_size = 128
args.sigma_noise = 0.02

args.prox_config = OmegaConf.create()
args.prox_config.prox = "lpn"
args.prox_config.model_path = "exps/celeba/models/ne_lpn_64ch/s=0.1/model.pt"

args.admm_config = OmegaConf.create()
args.admm_config.rho = 0.1
args.admm_config.maxiter = 20
args.admm_config.x0 = "adjoint"
args.admm_config.scale = admm_scale
args.admm_config.order = "132"


args.model_config = OmegaConf.create()
args.model_config.model = "ne_128"
args.model_config.params = OmegaConf.create()
args.model_config.params.in_dim = 3
args.model_config.params.hidden = 64
args.model_config.params.beta = 100
args.model_config.params.alpha = 1e-6


args.seed = 0
args.out_dir = None
args.measure = False
args.solver = "admm"
args.data_dir = None


# set original LPN arguments manually
args_lpn = OmegaConf.create()
args_lpn.dataset_config = OmegaConf.create()
args_lpn.dataset_config.dataset = "celeba"
args_lpn.dataset_config.root = "data/celeba"
args_lpn.dataset_config.start_idx = 100
args_lpn.dataset_config.num_imgs = 20

args_lpn.operator_config = OmegaConf.create()
args_lpn.operator_config.operator = "blur"
args_lpn.operator_config.sigma_blur = 1.0
args_lpn.operator_config.image_size = 128
args_lpn.sigma_noise = 0.02

args_lpn.prox_config = OmegaConf.create()
args_lpn.prox_config.prox = "lpn"
args_lpn.prox_config.model_path = "exps/celeba/models/lpn/s=0.1/model.pt"

args_lpn.admm_config = OmegaConf.create()
args_lpn.admm_config.rho = 0.1
args_lpn.admm_config.maxiter = 20
args_lpn.admm_config.x0 = "adjoint"
args_lpn.admm_config.scale = admm_scale
args_lpn.admm_config.order = "132"

args_lpn.model_config = OmegaConf.create()
args_lpn.model_config.model = "lpn_128"
args_lpn.model_config.params = OmegaConf.create()
args_lpn.model_config.params.in_dim = 3
args_lpn.model_config.params.hidden = 256
args_lpn.model_config.params.beta = 100
args_lpn.model_config.params.alpha = 1e-6

args_lpn.seed = 0
args_lpn.out_dir = None
args_lpn.measure = False
args_lpn.solver = "admm"
args_lpn.data_dir = None

# run the tests on the Norm-Equiv and original LPN

NE_PSNRs = []
LPN_PSNRs = []

NE_SSIMs = []
LPN_SSIMs = []

for sigma_blur, sigma_noise in zip(sigma_blur, sigma_noise):
    args.operator_config.sigma_blur = sigma_blur
    args.sigma_noise = sigma_noise
    args.out_dir = f"exps/celeba/results/inverse/deblur/blur={sigma_blur}_noise={sigma_noise}/admm"
    NE_results = main_celeba(args)
    NE_PSNRs.append((NE_results["PSNR_mean"], NE_results["PSNR_std"]))
    NE_SSIMs.append((NE_results["SSIM_mean"], NE_results["SSIM_std"]))

    args_lpn.operator_config.sigma_blur = sigma_blur
    args_lpn.sigma_noise = sigma_noise
    args_lpn.out_dir = f"exps/celeba/results/inverse/deblur/blur={sigma_blur}_noise={sigma_noise}/admm"
    LPN_results = main_celeba(args_lpn)
    LPN_PSNRs.append((LPN_results["PSNR_mean"], LPN_results["PSNR_std"]))
    LPN_SSIMs.append((LPN_results["SSIM_mean"], LPN_results["SSIM_std"]))


# save the results as np arrays
NE_PSNRs = np.array(NE_PSNRs, dtype=np.float64)
LPN_PSNRs = np.array(LPN_PSNRs, dtype=np.float64)

NE_SSIMs = np.array(NE_SSIMs, dtype=np.float64)
LPN_SSIMs = np.array(LPN_SSIMs, dtype=np.float64)

np.save("exps/celeba/results/NE_PSNRs.npy", NE_PSNRs)
np.save("exps/celeba/results/LPN_PSNRs.npy", LPN_PSNRs)

np.save("exps/celeba/results/NE_SSIMs.npy", NE_SSIMs)
np.save("exps/celeba/results/LPN_SSIMs.npy", LPN_SSIMs)

# plot the reslts into two graphs

fig, ax = plt.subplots(1, 2, figsize=(12, 6))
ax[0].errorbar(sigma_noise, NE_PSNRs[:, 0], yerr=NE_PSNRs[:, 1], label='NE PSNR', fmt='-o')
ax[0].errorbar(sigma_noise, LPN_PSNRs[:, 0], yerr=LPN_PSNRs[:, 1], label='LPN PSNR', fmt='-o')
ax[0].set_xlabel('Sigma Noise')
ax[0].set_ylabel('PSNR')
ax[0].set_title('PSNR vs Sigma Noise')
ax[0].legend()
ax[0].grid(True)

# Plot SSIM vs sigma noise
ax[1].errorbar(sigma_noise, NE_SSIMs[:, 0], yerr=NE_SSIMs[:, 1], label='NE SSIM', fmt='-o')
ax[1].errorbar(sigma_noise, LPN_SSIMs[:, 0], yerr=LPN_SSIMs[:, 1], label='LPN SSIM', fmt='-o')
ax[1].set_xlabel('Sigma Noise')
ax[1].set_ylabel('SSIM')
ax[1].set_title('SSIM vs Sigma Noise')
ax[1].legend()
ax[1].grid(True)

plt.tight_layout()
plt.savefig("exps/celeba/results/PSNR_SSIM_vs_Sigma_Noise.png")
plt.show()





