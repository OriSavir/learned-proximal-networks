import torch
import numpy as np

from lpn.utils import metrics
from lpn.utils import get_loss_hparams_and_lr, get_loss


class Validator:
    """Class for validation."""

    def __init__(self, dataloader, writer, sigma_noise):
        self.dataloader = dataloader
        self.writer = writer
        assert type(sigma_noise) == float
        self.sigma_noise = sigma_noise

    def _validate(self, model, args, step):
        """Validate the model on the validation set."""

        model.eval()
        device = next(model.parameters()).device
        
        loss_hparams, _ = get_loss_hparams_and_lr(args, step)
        loss_func = get_loss(loss_hparams)

        psnr_list = []
        ssim_list = []
        loss_list = []
        for batch in self.dataloader:
            clean_images = batch["image"].to(device)
            noise = torch.randn_like(clean_images)
            noisy_images = clean_images + noise * self.sigma_noise
            out = model(noisy_images)
            loss = loss_func(out, clean_images)

            psnr_, ssim_ = self.compute_metrics(clean_images, out)
            psnr_list.extend(psnr_)
            ssim_list.extend(ssim_)
            loss_list.append(loss.detach().item())

        print(f"PSNR: {np.mean(psnr_list)}")
        print(f"SSIM: {np.mean(ssim_list)}")
        self.psnr_list = psnr_list
        self.ssim_list = ssim_list
        self.loss_list = loss_list

    def compute_metrics(self, gt, out):
        """gt, out: batch, channel, height, width. torch.Tensor."""
        gt = gt.cpu().detach().numpy().transpose(0, 2, 3, 1)
        out = out.cpu().detach().numpy().transpose(0, 2, 3, 1)

        psnr_ = [metrics.compute_psnr(gt_, out_) for gt_, out_ in zip(gt, out)]
        ssim_ = [metrics.compute_ssim(gt_, out_) for gt_, out_ in zip(gt, out)]

        return psnr_, ssim_

    def _log(self, step):
        """Log the validation metrics."""

        self.writer.add_scalar("val/psnr", np.mean(self.psnr_list), step)
        self.writer.add_scalar("val/ssim", np.mean(self.ssim_list), step)
        self.writer.add_scalar("val/loss", np.mean(self.loss_list), step)

    def validate(self, model, args, step):
        """Validate the model and log the metrics."""

        self._validate(model, args, step)
        self._log(step)
