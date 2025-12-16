import torch
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "mps")


def get_ddpm_schedule(timesteps):
    """
    Helper function to get forward process coeffecients

    inputs:
        timesteps (int) : time-horization of forward and reverse process

    returns:
        dictionary :
            sqrt_alphas_cumprod: mean scaler
            sqrt_one_minus_alphas_cumprod: variance scaler
    """
    betas = torch.linspace(1e-4, 2e-2, timesteps)

    alphas = 1 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=-1)

    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod).to(device=device)
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1 - alphas_cumprod).to(device=device)

    return {
        "sqrt_alphas_cumprod": sqrt_alphas_cumprod,
        "sqrt_one_minus_alphas_cumprod": sqrt_one_minus_alphas_cumprod,
    }


def forward_process(x0, times, schedules):
    """
    Adds noise to input image

    inputs:
        x0 (tensor) : batch of clean images (B, 1, 28, 28)
        t (int) : batch of timesteps (B, 1)
        schedule : dictionary(tensor)

    returns:
        tuple(tensor) : batch of noisy images (B, 1, 28, 28) and batch of noise added (B, 1, 28, 28)
    """
    sqrt_alpha_bar = (
        schedules["sqrt_alphas_cumprod"].gather(-1, times - 1).reshape(-1, 1, 1, 1)
    )
    sqrt_one_minus_alpha_bar = (
        schedules["sqrt_one_minus_alphas_cumprod"]
        .gather(-1, times - 1)
        .reshape(-1, 1, 1, 1)
    )

    epsilon = torch.randn_like(x0)

    x_t = (sqrt_alpha_bar * x0) + (sqrt_one_minus_alpha_bar * epsilon)

    return x_t, epsilon
