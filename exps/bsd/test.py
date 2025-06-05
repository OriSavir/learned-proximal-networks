import os
import argparse
from omegaconf import OmegaConf
from pprint import pp
from lpn.inverse_bsd import main

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sigma_blur", type=float, required=True)
    parser.add_argument("--sigma_noise", type=float, required=True)
    parser.add_argument("--model_name", type=str, required=True, default="lpn_128")
    parser.add_argument("--image_size", type=int, default=128)
    parser.add_argument("--start_idx", type=int, default=100)
    parser.add_argument("--num_imgs", type=int, default=20)
    parser.add_argument("--split", type=str, default='test')
    parser.add_argument("--solver", type=str, default='admm', choices=['admm', 'pgd'])
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()

def build_config(args):
    conf = OmegaConf.create()

    conf.dataset_config = OmegaConf.create({
        "dataset": "bsds500",
        "root": "data/bsds500",
        "start_idx": args.start_idx,
        "num_imgs": args.num_imgs,
        "split": args.split,
        "image_size": args.image_size,
    })

    conf.operator_config = OmegaConf.create({
        "operator": "blur",
        "sigma_blur": args.sigma_blur,
        "image_size": args.image_size,
    })

    # Determine model prefix path based on model name
    if args.model_name == "lpn_128":
        model_prefix = "lpn"
    elif args.model_name == "ne_lpn_128":
        model_prefix = "ne_lpn"
    elif args.model_name == "ne_normalized_and_scaled_lpn_128":
        model_prefix = "ne_norm_and_scaled_lpn"
    elif args.model_name == "ne_by_forward_lpn_128":
        model_prefix = "ne_by_forward_lpn"
    elif args.model_name == "drunet":
        model_prefix = "drunet"
    else:
        raise ValueError(f"Unknown model name: {args.model_name}")


    # create the config
    if "lpn" in args.model_name:
        model_params = {
            "in_dim": 3,
            "hidden": 256,
            "beta": 100,
            "alpha": 1e-6,
        }
    elif args.model_name == "drunet":
        model_params = {
            "in_nc": 3,
            "out_nc": 3,
            "nc": [64, 128, 256, 512],
            "nb": 4,
            "blind": True,
            "mode": "norm-equiv"
        }
    else:
        raise ValueError(f"No parameter config defined for model: {args.model_name}")

    conf.model_config = OmegaConf.create({
        "model": args.model_name,
        "model_prefix": model_prefix,
        "params": model_params,
        "model_prefix_path": f"exps/bsd/models/{model_prefix}"
    })




    conf.prox_config = OmegaConf.create({
        "prox": "lpn" if "lpn" in args.model_name else "drunet",
        "model_path": None  # Set below by set_best()
    })

    conf.admm_config = OmegaConf.create({
        "rho": 0.1,
        "maxiter": 20,
        "x0": "adjoint",
        "scale": 0.5,  # Default, will be overwritten
        "order": "132",
    })

    conf.sigma_noise = args.sigma_noise
    conf.seed = args.seed
    conf.measure = False
    conf.solver = args.solver
    conf.data_dir = None

    conf.out_dir = f"exps/bsd/results/inverse/{args.model_name}/deblur/blur={args.sigma_blur}_noise={args.sigma_noise}/{args.solver}"
    os.makedirs(conf.out_dir, exist_ok=True)

    return conf

def set_best(conf):
    key = (conf.operator_config.sigma_blur, conf.sigma_noise)
    model_prefix = conf.model_config.model_prefix

    if conf.solver.lower() == "admm":
        admm_params = {
            (1.0, 0.04): (f"exps/bsd/models/{model_prefix}/s=0.1/model.pt", 0.5, 20),
            (2.0, 0.02): (f"exps/bsd/models/{model_prefix}/s=0.1/model.pt", 2.0, 20),
            (2.0, 0.04): (f"exps/bsd/models/{model_prefix}/s=0.1/model.pt", 0.5, 20),
        }

        if key not in admm_params:
            raise ValueError(f"Best ADMM parameters not defined for sigma_blur={key[0]}, sigma_noise={key[1]}")
        
        model_path, scale, maxiter = admm_params[key]
        conf.prox_config.model_path = model_path
        conf.admm_config.scale = scale
        conf.admm_config.maxiter = maxiter

    elif conf.solver.lower() == "pgd":
        pgd_params = {
            (1.0, 0.02): (f"exps/bsd/models/{model_prefix}/s=0.1/model.pt", 2.0, 20),
            (1.0, 0.04): (f"exps/bsd/models/{model_prefix}/s=0.1/model.pt", 2.0, 20),
            (2.0, 0.02): (f"exps/bsd/models/{model_prefix}/s=0.1/model.pt", 2.0, 20),
            (2.0, 0.04): (f"exps/bsd/models/{model_prefix}/s=0.1/model.pt", 2.0, 20),
        }

        if key not in pgd_params:
            raise ValueError(f"Best PGD parameters not defined for sigma_blur={key[0]}, sigma_noise={key[1]}")

        model_path, eta, maxiter = pgd_params[key]
        conf.prox_config.model_path = model_path
        conf.pgd_config = OmegaConf.create()
        conf.pgd_config.eta = eta
        conf.pgd_config.maxiter = maxiter

    else:
        raise ValueError(f"Unknown solver: {conf.solver}")

    return conf


if __name__ == "__main__":
    args = parse_args()
    config = build_config(args)
    config = set_best(config)
    pp(config)
    main(config)
