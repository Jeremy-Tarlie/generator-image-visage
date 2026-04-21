from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import TypedDict

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
from PIL import ImageOps
from torch.utils.data import DataLoader, Dataset
from torchvision.utils import make_grid, save_image


@dataclass(frozen=True)
class TrainConfig:
    data_dir: Path = Path(__file__).resolve().parent / "data"
    output_model_path: Path = Path(__file__).resolve().parent / "gan_faces.pt"
    output_preview_path: Path = Path(__file__).resolve().parent / "gan_faces_samples.png"
    image_size: int = 64
    latent_dim: int = 96
    batch_size: int = 64
    epochs: int = 20
    learning_rate: float = 2e-4
    num_workers: int = 0
    seed: int = 42


class GANCheckpoint(TypedDict):
    generator_state_dict: dict[str, torch.Tensor]
    discriminator_state_dict: dict[str, torch.Tensor]
    optimizer_g_state_dict: dict[str, object]
    optimizer_d_state_dict: dict[str, object]
    latent_dim: int
    image_size: int
    last_epoch: int


class FaceImageDataset(Dataset[torch.Tensor]):
    def __init__(self, root_dir: Path, image_size: int) -> None:
        self.image_paths = sorted(
            [
                image_path
                for image_path in root_dir.rglob("*")
                if image_path.is_file() and image_path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
            ]
        )
        if not self.image_paths:
            raise FileNotFoundError(f"Aucune image detectee dans: {root_dir}")

        resize_size = int(image_size * 1.25)
        self.transform = transforms.Compose(
            [
                transforms.Resize((resize_size, resize_size)),
                transforms.CenterCrop(image_size),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int) -> torch.Tensor:
        path = self.image_paths[index]
        with Image.open(path) as image:
            rgb_image = ImageOps.exif_transpose(image).convert("RGB")
        return self.transform(rgb_image)


class Generator(nn.Module):
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


class Discriminator(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.utils.spectral_norm(nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.utils.spectral_norm(nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.utils.spectral_norm(nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.utils.spectral_norm(nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=0),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).view(-1, 1)


def init_weights(module: nn.Module) -> None:
    if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.normal_(module.weight.data, mean=0.0, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias.data)
    elif isinstance(module, nn.BatchNorm2d):
        nn.init.normal_(module.weight.data, mean=1.0, std=0.02)
        nn.init.zeros_(module.bias.data)


def denormalize(images: torch.Tensor) -> torch.Tensor:
    return ((images + 1.0) * 0.5).clamp(0.0, 1.0)


def load_checkpoint_if_available(
    generator: Generator,
    discriminator: Discriminator,
    optimizer_g: torch.optim.Optimizer,
    optimizer_d: torch.optim.Optimizer,
    checkpoint_path: Path,
) -> int:
    if not checkpoint_path.exists():
        return 0
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if not isinstance(checkpoint, dict):
        return 0
    required_keys = {
        "generator_state_dict",
        "discriminator_state_dict",
        "optimizer_g_state_dict",
        "optimizer_d_state_dict",
        "last_epoch",
    }
    if not required_keys.issubset(set(checkpoint.keys())):
        return 0
    generator.load_state_dict(checkpoint["generator_state_dict"])
    discriminator.load_state_dict(checkpoint["discriminator_state_dict"])
    optimizer_g.load_state_dict(checkpoint["optimizer_g_state_dict"])
    optimizer_d.load_state_dict(checkpoint["optimizer_d_state_dict"])
    last_epoch = checkpoint.get("last_epoch", 0)
    if not isinstance(last_epoch, int):
        return 0
    return max(last_epoch, 0)


def save_samples(generator: Generator, latent_dim: int, device: torch.device, output_path: Path) -> None:
    generator.eval()
    with torch.inference_mode():
        z = torch.randn(16, latent_dim, 1, 1, device=device)
        fake = generator(z).cpu()
        grid = make_grid(denormalize(fake), nrow=4)
        save_image(grid, output_path)


def train(config: TrainConfig, resume: bool) -> None:
    torch.manual_seed(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = FaceImageDataset(config.data_dir, config.image_size)
    dataloader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=True,
    )

    generator = Generator(config.latent_dim).to(device)
    discriminator = Discriminator().to(device)
    generator.apply(init_weights)
    discriminator.apply(init_weights)
    criterion = nn.BCEWithLogitsLoss()
    fixed_z = torch.randn(16, config.latent_dim, 1, 1, device=device)

    optimizer_g = torch.optim.Adam(generator.parameters(), lr=config.learning_rate, betas=(0.5, 0.999))
    optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=config.learning_rate * 0.5, betas=(0.5, 0.999))

    start_epoch = (
        load_checkpoint_if_available(
            generator, discriminator, optimizer_g, optimizer_d, config.output_model_path
        )
        if resume
        else 0
    )
    print(
        f"Dataset: {len(dataset)} images | Device: {device.type} | "
        f"Resume: {'oui' if start_epoch > 0 else 'non'} (epoch {start_epoch})",
        flush=True,
    )

    for epoch in range(start_epoch + 1, config.epochs + 1):
        generator.train()
        discriminator.train()
        d_epoch_loss = 0.0
        g_epoch_loss = 0.0

        for real_images in dataloader:
            real_images = real_images.to(device)
            current_batch_size = real_images.size(0)

            real_targets = torch.full((current_batch_size, 1), 0.9, device=device)
            fake_targets = torch.zeros(current_batch_size, 1, device=device)

            z = torch.randn(current_batch_size, config.latent_dim, 1, 1, device=device)
            fake_images = generator(z)

            optimizer_d.zero_grad()
            d_real = criterion(discriminator(real_images), real_targets)
            d_fake = criterion(discriminator(fake_images.detach()), fake_targets)
            d_loss = 0.5 * (d_real + d_fake)
            d_loss.backward()
            optimizer_d.step()

            optimizer_g.zero_grad()
            g_targets = torch.ones(current_batch_size, 1, device=device)
            g_loss = criterion(discriminator(fake_images), g_targets)
            g_loss.backward()
            optimizer_g.step()

            # 2e update du generateur pour aider a converger en peu d'epochs.
            z2 = torch.randn(current_batch_size, config.latent_dim, 1, 1, device=device)
            fake_images2 = generator(z2)
            optimizer_g.zero_grad()
            g_loss_2 = criterion(discriminator(fake_images2), g_targets)
            g_loss_2.backward()
            optimizer_g.step()

            d_epoch_loss += d_loss.item()
            g_epoch_loss += 0.5 * (g_loss.item() + g_loss_2.item())

        num_batches = len(dataloader)
        avg_d = d_epoch_loss / num_batches
        avg_g = g_epoch_loss / num_batches

        checkpoint: GANCheckpoint = {
            "generator_state_dict": generator.state_dict(),
            "discriminator_state_dict": discriminator.state_dict(),
            "optimizer_g_state_dict": optimizer_g.state_dict(),
            "optimizer_d_state_dict": optimizer_d.state_dict(),
            "latent_dim": config.latent_dim,
            "image_size": config.image_size,
            "last_epoch": epoch,
        }
        torch.save(checkpoint, config.output_model_path)

        generator.eval()
        with torch.inference_mode():
            fake = generator(fixed_z).cpu()
            grid = make_grid(denormalize(fake), nrow=4)
            save_image(grid, config.output_preview_path)
        print(f"Epoch {epoch:03d}/{config.epochs} | D={avg_d:.6f} | G={avg_g:.6f}", flush=True)

    print(f"[OK] Modele GAN sauvegarde: {config.output_model_path}", flush=True)
    print(f"[OK] Apercu GAN sauvegarde: {config.output_preview_path}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Entrainement GAN sur dataset de visages")
    parser.add_argument("--epochs", type=int, default=20, help="Nombre d'epoques")
    parser.add_argument("--batch-size", type=int, default=64, help="Taille de batch")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reprend l'entrainement depuis gan_faces.pt s'il existe",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(TrainConfig(epochs=args.epochs, batch_size=args.batch_size), resume=args.resume)
