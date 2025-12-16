import torch
import torch.nn as nn
import torch.nn.functional as F


def CreateTimeEmbeddings(timesteps, dim=128, max_period=10000):
    """
    Generates the sinusoidal time embedding vector based on the formula:
    PE(t, 2i) = sin(t / 10000^(2i/dim))
    PE(t, 2i + 1) = cos(t / 10000^(2i/dim))

    Args:
        timesteps (torch.Tensor): A tensor of timesteps (e.g., [1, 500, 999]).
        dim (int): The target dimension of the embedding vector (e.g., 256).

    Returns:
        torch.Tensor: The final time embedding tensor of shape (N, dim).
    """

    assert dim % 2 == 0

    half_dim = dim // 2

    exponent = (
        torch.arange(half_dim, dtype=torch.float32, device=timesteps.device) / half_dim
    )

    denominator = max_period**exponent

    timesteps = timesteps.unsqueeze(1).float()

    time_arg = timesteps / denominator

    sin_part = torch.sin(time_arg)
    cos_part = torch.cos(time_arg)

    embedding = torch.cat([sin_part, cos_part], dim=1)

    return embedding


class TimeEmbeddingFiLm(nn.Module):
    """
    MLP to transform Time Embedding into Shifting and Scaling for FiLm

    Args:
        dim (int): The dimension of the embedding vector
        channels (int): The number of channels to do FiLm Normalization for

    Returns:
        Torch.tensor tuple:
            1. The final scale tensor (channels)
            2. The final shift tensor (channels)
    """

    def __init__(self, dim, channels):
        super().__init__()
        self.f1 = nn.Linear(dim, 4 * dim)
        self.f2 = nn.Linear(4 * dim, dim)
        self.scale = nn.Linear(dim, channels)
        self.shift = nn.Linear(dim, channels)

    def forward(self, x):

        # x = [B, dim]

        x = self.f1(x)
        x = F.silu(x)
        x = self.f2(x)
        x = F.silu(x)
        scale = self.scale(x)
        shift = self.shift(x)
        return scale, shift


class ResidualBlock(nn.Module):
    """
    Convolutional Block that uses residual trick for better gradient flow and function learning

    Args:
        in_channels (int): The number of channels of the incoming tensor
        out_channels(int): The number of channels for the outgoing tensor
        kernel_size (tuple(int)): Size of kernel for convolution operation (must be odd and square)

    Returns:
        torch.tensor: The final feature map output
    """

    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        self.padding = kernel_size[0] // 2
        self.gn1 = nn.GroupNorm(
            num_groups=min(in_channels, 8), num_channels=in_channels
        )
        self.c1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=self.padding,
        )

        self.c2 = nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=self.padding,
        )
        self.gn2 = nn.GroupNorm(
            num_groups=min(out_channels, 8), num_channels=out_channels
        )
        self.FiLm = TimeEmbeddingFiLm(128, out_channels)
        if in_channels != out_channels:
            self.identity = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1))
        else:
            self.identity = nn.Identity()

    def forward(self, x, timesteps):
        residual = self.gn1(x)
        residual = F.silu(residual)
        residual = self.c1(residual)
        residual = self.gn2(residual)

        time_vector = CreateTimeEmbeddings(timesteps, 128)
        scales, shifts = self.FiLm(time_vector)
        scales_b = scales.unsqueeze(-1).unsqueeze(-1)
        shifts_b = shifts.unsqueeze(-1).unsqueeze(-1)
        adaptive_residual = (scales_b + 1) * residual + shifts_b
        adaptive_residual = F.silu(adaptive_residual)
        adaptive_residual = self.c2(adaptive_residual)

        shortcut = self.identity(x)

        return adaptive_residual + shortcut


class UpsampleBlock(nn.Module):
    """
    Convolutional block that upsamples image
    """

    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        self.tc1 = nn.ConvTranspose2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=2,
        )
        self.gn1 = nn.GroupNorm(
            num_groups=min(out_channels, 8), num_channels=out_channels
        )

    def forward(self, x):
        x = self.tc1(x)
        x = self.gn1(x)
        return F.silu(x)


class UNet(nn.Module):
    def __init__(self):
        super().__init__()

        # Encode
        self.rb1 = ResidualBlock(3, 16, (3, 3))
        self.mp1 = nn.MaxPool2d(kernel_size=(2, 2))
        self.rb2 = ResidualBlock(16, 32, (3, 3))
        self.mp2 = nn.MaxPool2d(kernel_size=(2, 2))

        # Decode
        self.us1 = UpsampleBlock(32, 16, kernel_size=(2, 2))
        self.rb3 = ResidualBlock(48, 32, kernel_size=(3, 3))
        self.us2 = UpsampleBlock(32, 16, kernel_size=(2, 2))
        self.rb4 = ResidualBlock(32, 1, kernel_size=(3, 3))

    def forward(self, x, times):

        # Encode Block 1
        block1 = self.rb1(x, times)  # (B, 3, 28, 28) -> (B, 16, 28, 28)
        skip1 = block1
        pooled_block1 = self.mp1(block1)  # (B, 16, 28, 28) -> (B, 16, 14, 14)

        # Encode Block 2
        block2 = self.rb2(pooled_block1, times)  # (B, 16, 14, 14) - > (B, 32, 14, 14)
        skip2 = block2
        pooled_block2 = self.mp2(block2)  # (B, 32, 14, 14) -> (B, 32, 7, 7)

        # Decode Block 1
        upsample1 = self.us1(pooled_block2)  # (B, 32, 7, 7) -> (B, 16, 14, 14)
        upsample1_skip2 = torch.cat(
            [upsample1, skip2], dim=1
        )  # (B, 16, 14, 14) -> (B, 48, 14, 14)
        block3 = self.rb3(upsample1_skip2, times)  # (B, 48, 14, 14) -> (B, 32, 14, 14)

        # Decode Block 2
        upsample2 = self.us2(block3)  # (B, 32, 14, 14) -> (B, 16, 28, 28)
        upsample2_skip2 = torch.cat(
            [upsample2, skip1], dim=1
        )  # (B, 16, 28, 28) -> (B, 32, 28, 28)
        block4 = self.rb4(upsample2_skip2, times)  # (B, 32, 28, 28) -> (B, 1, 28, 28)

        return block4
