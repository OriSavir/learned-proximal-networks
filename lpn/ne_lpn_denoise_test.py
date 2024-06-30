import argparse
from pprint import pp
import torch
from torch import nn
from tqdm import tqdm
import numpy as np
import json
import os
from omegaconf import OmegaConf
from torch.utils.tensorboard import SummaryWriter

from utils import load_dataset, load_config
from utils import get_model
from utils import get_loss_hparams_and_lr, get_loss
from utils import trainer
from utils import utils
import matplotlib.pyplot as plt


#set torch device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


#code in this file will serve as a denoising test of the trained ne_lpn model

dataset_config_path = "../exps/mnist/configs/mnist/test_dataset.json"


def run_test(network_type, sigma, model_path, dataset_config_path, model_weight_path):
    # load model and dataset
    model_config = load_config(model_path)
    model = get_model(model_config)
    model.load_state_dict(torch.load(model_weight_path)["model_state_dict"])

    dataset_config = load_config(dataset_config_path)
    test_dataset = load_dataset(dataset_config, "test")

    #get the data loader
    test_data_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=1, shuffle=False, num_workers=4
    )

    total_loss = 0

    # test for batches and apply guassian noise to each
    for step, batch in enumerate(test_data_loader):
        clean_images = batch["image"].to(device)
        noise = torch.randn_like(clean_images)
        noised_images = clean_images + noise * sigma
        denoised_images = model(noised_images)
        # calculate the difference between the clean images and the denoised images using MSE loss (if that is correct)
        mse_loss = nn.MSELoss()
        loss = mse_loss(clean_images, denoised_images)
        total_loss += loss.item()
        #print(f"Step: {step}, Loss: {loss.item()}")
        if step == 0:
            #save the images using matplotlib
            plt.figure()
            plt.subplot(1, 3, 1)
            plt.imshow(clean_images[0, 0].cpu().detach().numpy(), cmap="gray")
            plt.title("Clean Image")
            plt.subplot(1, 3, 2)
            plt.imshow(noised_images[0, 0].cpu().detach().numpy(), cmap="gray")
            plt.title("Noised Image")
            plt.subplot(1, 3, 3)
            plt.imshow(denoised_images[0, 0].cpu().detach().numpy(), cmap="gray")
            plt.title("Denoised Image")
            plt.show()
            img_name = f"denoised_image_{network_type}_{sigma}.png"
            plt.savefig(img_name)
            break

            

    average_loss = total_loss / (step + 1)
    print(f"Average Loss: {average_loss}")
    return average_loss

ne_model_config_path = "../exps/mnist/configs/mnist/model_ne.json"
ne_model_weight_path = "../exps/mnist/experiments/ne_mnist/model.pt"
lpn_model_config_path = "../exps/mnist/configs/mnist/model.json"
lpn_model_weight_path = "../exps/mnist/experiments/mnist/model.pt"
dataset_config_path = "../exps/mnist/configs/mnist/test_dataset.json"


print("NE Model Test")
avg_loss_ne = run_test("NE", 0.05, ne_model_config_path, dataset_config_path, ne_model_weight_path)

print("LPN Model Test")
avg_loss_lpn = run_test("LPN", 0.05, lpn_model_config_path, dataset_config_path, lpn_model_weight_path)

print("Difference (NE - LPN):", avg_loss_ne - avg_loss_lpn)


