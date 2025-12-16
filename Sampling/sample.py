import torch
import os
import torchvision.utils as vutils


def sample(conditions, model, filename):
    """
    Denoising Reverse Process to generate/sample images and write to file

    inputs
        conditions (tensor): (B, 2, 28, 28) Input Digits to be Subtracted
        model (UNet) : UNet that predicts noise added at time step t
    """
    model.eval()

    with torch.no_grad():

        device = next(model.parameters()).device

        noisy = torch.randn(size=(conditions.shape[0], 1, 28, 28)).to(device=device)

        betas = torch.linspace(1e-4, 2e-2, steps=1000).to(device=device)
        alphas = 1 - betas
        cum_prod_alphas = torch.cumprod(alphas, dim=0).to(device=device)

        for t in range(1000, 0, -1):

            time_tensor = torch.full(
                (conditions.shape[0],), t, dtype=torch.long, device=device
            )
            conditioned_noisy = torch.cat([noisy, conditions], dim=1)

            prediced_epsilon = model(conditioned_noisy, time_tensor)

            beta = betas[t - 1]
            alpha = 1 - beta
            alpha_bar = cum_prod_alphas[t - 1]

            noisy = torch.pow(alpha, -0.5) * (
                noisy - (beta / torch.sqrt(1 - alpha_bar)) * prediced_epsilon
            )

            if t > 1:
                variance = ((1.0 - cum_prod_alphas[t - 2]) / (1.0 - alpha_bar)) * beta
                sigma_t = torch.sqrt(variance)

                z = torch.randn_like(noisy)

                noisy = noisy + sigma_t * z

        final_image_tensor = noisy + 1.0

        final_image_tensor = final_image_tensor / 2.0

        final_image_tensor = torch.clamp(final_image_tensor, 0.0, 1.0)

        input_1 = conditions[:, 0:1, :, :]
        input_2 = conditions[:, 1:2, :, :]

        input_1 = (input_1 + 1.0) / 2.0
        input_2 = (input_2 + 1.0) / 2.0

        comparison_tensor = torch.cat([input_1, input_2, final_image_tensor], dim=3)

        vutils.save_image(comparison_tensor, filename, nrow=1, padding=4, pad_value=0.5)

        print(f"Generated images saved to {os.path.abspath(filename)}")

    model.train()
