from __future__ import annotations

from pathlib import Path
from typing import Final

import torch
import torch.nn as nn


class VAE(nn.Module):
    def __init__(self, latent_dim: int) -> None:
        super().__init__()
        self.latent_dim: Final[int] = latent_dim
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )
        self.fc_mu = nn.Linear(256 * 4 * 4, latent_dim)
        self.fc_log_var = nn.Linear(256 * 4 * 4, latent_dim)
        self.fc_decode = nn.Linear(latent_dim, 256 * 4 * 4)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid(),
        )

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        decoded = self.fc_decode(z).view(-1, 256, 4, 4)
        return self.decoder(decoded)


class GANGenerator(nn.Module):
    def __init__(self, latent_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, 512, kernel_size=4, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh(),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class GeneratorService:
    def __init__(self, checkpoint_path: Path) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        if not isinstance(checkpoint, dict):
            raise ValueError("Checkpoint invalide: format non supporte")

        latent_dim_object = checkpoint.get("latent_dim", 128)
        if not isinstance(latent_dim_object, int):
            raise ValueError("Checkpoint invalide: latent_dim doit etre un entier")

        self.is_gan = "generator_state_dict" in checkpoint
        if self.is_gan:
            model_state_object = checkpoint["generator_state_dict"]
            if not isinstance(model_state_object, dict):
                raise ValueError("Checkpoint GAN invalide: generator_state_dict n'est pas un dictionnaire")
            self.model = GANGenerator(latent_dim_object).to(self.device)
        elif "model_state_dict" in checkpoint:
            model_state_object = checkpoint["model_state_dict"]
            if not isinstance(model_state_object, dict):
                raise ValueError("Checkpoint VAE invalide: model_state_dict n'est pas un dictionnaire")
            self.model = VAE(latent_dim_object).to(self.device)
        else:
            raise ValueError("Checkpoint invalide: aucune cle de modele reconnue")

        self.model.load_state_dict(model_state_object)
        self.model.eval()
        self.latent_dim = latent_dim_object

    def make_latent(self, batch_size: int) -> torch.Tensor:
        if self.is_gan:
            return torch.randn(batch_size, self.latent_dim, 1, 1, device=self.device)
        return torch.randn(batch_size, self.latent_dim, device=self.device)

    @torch.inference_mode()
    def generate(self) -> torch.Tensor:
        latent = self.make_latent(batch_size=1)
        if self.is_gan:
            image = self.model(latent)
            return (((image[0] + 1.0) * 0.5).cpu()).clamp(0.0, 1.0)
        image = self.model.decode(latent)
        return image[0].cpu().clamp(0.0, 1.0)
