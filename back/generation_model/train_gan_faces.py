from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, TypedDict

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
    latent_dim: int = 128
    batch_size: int = 64
    epochs: int = 160
    learning_rate: float = 2e-4
    ema_decay: float = 0.999
    num_workers: int = max(2, min(8, os.cpu_count() or 2))
    seed: int = 42


class GANCheckpoint(TypedDict, total=False):
    generator_state_dict: dict[str, torch.Tensor]
    discriminator_state_dict: dict[str, torch.Tensor]
    optimizer_g_state_dict: dict[str, object]
    optimizer_d_state_dict: dict[str, object]
    ema_generator_state_dict: dict[str, torch.Tensor]
    latent_dim: int
    image_size: int
    last_epoch: int


DeviceArg = Literal["auto", "cpu", "cuda"]


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
            if image.mode == "P" and "transparency" in image.info:
                image = image.convert("RGBA")
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
        return self.net(x).view(-1)


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


def discriminator_hinge_loss(real_logits: torch.Tensor, fake_logits: torch.Tensor) -> torch.Tensor:
    return torch.relu(1.0 - real_logits).mean() + torch.relu(1.0 + fake_logits).mean()


def generator_hinge_loss(fake_logits: torch.Tensor) -> torch.Tensor:
    return -fake_logits.mean()


def update_ema(ema_model: nn.Module, current_model: nn.Module, decay: float) -> None:
    with torch.no_grad():
        for ema_param, model_param in zip(ema_model.parameters(), current_model.parameters()):
            ema_param.lerp_(model_param, 1.0 - decay)


def resolve_device(device_arg: DeviceArg) -> torch.device:
    if device_arg == "cpu":
        return torch.device("cpu")
    if device_arg == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA demandee mais indisponible. Verifie les drivers/CUDA Toolkit.")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_checkpoint_if_available(
    generator: Generator,
    ema_generator: Generator,
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
    if "ema_generator_state_dict" in checkpoint:
        ema_generator.load_state_dict(checkpoint["ema_generator_state_dict"])
    else:
        ema_generator.load_state_dict(generator.state_dict())
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


def train(config: TrainConfig, resume: bool, device_arg: DeviceArg) -> None:
    torch.manual_seed(config.seed)
    device = resolve_device(device_arg)
    use_amp = device.type == "cuda"
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")

    dataset = FaceImageDataset(config.data_dir, config.image_size)
    loader_kwargs: dict[str, object] = {}
    if config.num_workers > 0:
        loader_kwargs["prefetch_factor"] = 4
        loader_kwargs["persistent_workers"] = True
    dataloader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=True,
        **loader_kwargs,
    )

    generator = Generator(config.latent_dim).to(device)
    ema_generator = Generator(config.latent_dim).to(device)
    discriminator = Discriminator().to(device)
    if device.type == "cuda":
        generator = generator.to(memory_format=torch.channels_last)
        ema_generator = ema_generator.to(memory_format=torch.channels_last)
        discriminator = discriminator.to(memory_format=torch.channels_last)
    generator.apply(init_weights)
    ema_generator.load_state_dict(generator.state_dict())
    for param in ema_generator.parameters():
        param.requires_grad_(False)
    discriminator.apply(init_weights)
    fixed_z = torch.randn(16, config.latent_dim, 1, 1, device=device)

    optimizer_g = torch.optim.Adam(generator.parameters(), lr=config.learning_rate, betas=(0.0, 0.9))
    optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=config.learning_rate, betas=(0.0, 0.9))
    scaler_g = torch.amp.GradScaler("cuda", enabled=use_amp)
    scaler_d = torch.amp.GradScaler("cuda", enabled=use_amp)

    start_epoch = (
        load_checkpoint_if_available(
            generator, ema_generator, discriminator, optimizer_g, optimizer_d, config.output_model_path
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
            real_images = real_images.to(
                device, non_blocking=device.type == "cuda"
            ).contiguous(memory_format=torch.channels_last)
            current_batch_size = real_images.size(0)

            z = torch.randn(current_batch_size, config.latent_dim, 1, 1, device=device)
            optimizer_d.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                fake_images = generator(z)
                d_real_logits = discriminator(real_images)
                d_fake_logits = discriminator(fake_images.detach())
                d_loss = discriminator_hinge_loss(d_real_logits, d_fake_logits)
            scaler_d.scale(d_loss).backward()
            scaler_d.step(optimizer_d)
            scaler_d.update()

            optimizer_g.zero_grad(set_to_none=True)
            z_g = torch.randn(current_batch_size, config.latent_dim, 1, 1, device=device)
            with torch.amp.autocast("cuda", enabled=use_amp):
                fake_for_g = generator(z_g)
                g_loss = generator_hinge_loss(discriminator(fake_for_g))
            scaler_g.scale(g_loss).backward()
            scaler_g.step(optimizer_g)
            scaler_g.update()
            update_ema(ema_generator, generator, config.ema_decay)

            d_epoch_loss += d_loss.item()
            g_epoch_loss += g_loss.item()

        num_batches = len(dataloader)
        avg_d = d_epoch_loss / num_batches
        avg_g = g_epoch_loss / num_batches

        checkpoint: GANCheckpoint = {
            "generator_state_dict": generator.state_dict(),
            "discriminator_state_dict": discriminator.state_dict(),
            "optimizer_g_state_dict": optimizer_g.state_dict(),
            "optimizer_d_state_dict": optimizer_d.state_dict(),
            "ema_generator_state_dict": ema_generator.state_dict(),
            "latent_dim": config.latent_dim,
            "image_size": config.image_size,
            "last_epoch": epoch,
        }
        torch.save(checkpoint, config.output_model_path)

        ema_generator.eval()
        with torch.inference_mode():
            fake = ema_generator(fixed_z).cpu()
            grid = make_grid(denormalize(fake), nrow=4)
            save_image(grid, config.output_preview_path)
        print(f"Epoch {epoch:03d}/{config.epochs} | D={avg_d:.6f} | G={avg_g:.6f}", flush=True)

    print(f"[OK] Modele GAN sauvegarde: {config.output_model_path}", flush=True)
    print(f"[OK] Apercu GAN sauvegarde: {config.output_preview_path}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Entrainement GAN sur dataset de visages")
    parser.add_argument("--epochs", type=int, default=160, help="Nombre d'epoques")
    parser.add_argument("--batch-size", type=int, default=64, help="Taille de batch")
    parser.add_argument("--num-workers", type=int, default=TrainConfig.num_workers, help="Workers DataLoader")
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="Device d'entrainement (auto=CUDA si disponible)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reprend l'entrainement depuis gan_faces.pt s'il existe",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(
        TrainConfig(epochs=args.epochs, batch_size=args.batch_size, num_workers=args.num_workers),
        resume=args.resume,
        device_arg=args.device,
    )
