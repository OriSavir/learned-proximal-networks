import numpy as np
import torch


def prox(x, model):
    """Evaluate the learned proximal operator at x.
    Inputs:
        x: (n, ), a vector of n points, numpy array
        model: an LPN model
    Outputs:
        y: (n, ), a vector of n points, numpy array
    """
    device = next(model.parameters()).device
    x = torch.tensor(x).unsqueeze(1).float().to(device)
    return model(x).squeeze(1).detach().cpu().numpy()


def cvx(x, model):
    """Evaluate the learned convex function at x.
    Inputs:
        x: (n, ), a vector of n points, numpy array
        model: an LPN model
    Outputs:
        y: (n, ), a vector of n values, numpy array
    """
    device = next(model.parameters()).device
    x = torch.tensor(x).unsqueeze(1).float().to(device)
    return model.scalar(x).squeeze(1).detach().cpu().numpy()


def prior(x, model):
    """Evaluate the learned prior function at x.
    Inputs:
        x: (n, ), a vector of n points, numpy array
        model: an LPN model
    Outputs:
        y: (n, ), a vector of n values, numpy array
    """
    # psi(y) = <y, f(y)> - 1/2 ||f(y)||^2 - phi(f(y))
    y = invert_mse(x, model)
    psi = cvx(y, model)
    q = 0.5 * (x**2)  # quadratic term
    print(y.shape, x.shape, q.shape, psi.shape)
    out = y * x - q - psi

    return out


def invert_mse(x, model):
    """Invert the learned proximal operator at x by minimizing the MSE.
    Inputs:
        x: (n, ), a vector of n points, numpy array
        model: an LPN model
    Outputs:
        y: (n, ), a vector of n points, numpy array
    """
    device = next(model.parameters()).device
    x = torch.tensor(x).unsqueeze(1).float().to(device)
    y = torch.zeros_like(x).to(device) + 0.1

    optimizer = torch.optim.Adam([y], lr=1e-2)

    for i in range(10000):
        optimizer.zero_grad()
        loss = (model(y) - x).pow(2).mean()
        loss.backward()
        optimizer.step()
        if i % 100 == 0:
            print("mse", loss.item())
    print("final mse", loss.item())

    return y.squeeze(1).detach().cpu().numpy()


def gt_cvx(x):
    """Ground-truth convex function for the negative log-prior of N(0,1).
    x: numpy array, shape (n,)
    """
    return (x**2) / 4


def prox_gaussian(x, lam=1.0):
    """
    Proximal operator of the negative log of N(0,1).

    For phi(z) = 0.5 * z^2, the prox is:
        prox_{lam * phi}(x) = x / (1 + lam)

    Args:
        x (np.ndarray): input array
        lam (float): regularization parameter λ

    Returns:
        np.ndarray: prox result
    """
    return x / (1 + lam)

def split_normal_log_prior(x, mean=0.0, std_left=1.0, std_right=2.0):
    """
    Compute the log-prior of a split normal distribution.

    Args:
        x (np.ndarray): input array
        mean (float): mean of the distribution
        std_left (float): standard deviation for x < mean
        std_right (float): standard deviation for x >= mean

    Returns:
        np.ndarray: log-prior values
    """
    norm_const = np.log(np.sqrt(2 / np.pi)  * 1 / (std_left + std_right))
    log_prior = np.where(
        x < mean,
        -1 * norm_const + 0.5 * ((x - mean) / std_left) ** 2,
        -1 * norm_const + 0.5 * ((x - mean) / std_right) ** 2
    )
    return log_prior

def prox_split_normal(x, lam=1.0, mean=0.0, std_left=1.0, std_right=2.0):
    """
    Proximal operator for the split normal distribution.

    Args:
        x (np.ndarray): input array
        mean (float): mean of the distribution
        std_left (float): standard deviation for x < mean
        std_right (float): standard deviation for x >= mean

    Returns:
        np.ndarray: prox result
    """
    a_left  = lam / (std_left**2)
    a_right = lam / (std_right**2)
    return np.where(
        x < mean,
        (x + a_left * mean)  / (1 + a_left),
        (x + a_right * mean) / (1 + a_right)
    )

def gt_cvx_split_normal(x, mean=0.0, std_left=1.0, std_right=2.0):
    """
    Get the convex function for the split normal distribution.

    Args:
        mean (float): mean of the distribution
        std_left (float): standard deviation for x < mean
        std_right (float): standard deviation for x >= mean

    Returns:
        function: convex function
    """
    return np.where(
        x < mean,
        (x**2) / 4,
        2 * (x**2) / 5
    )