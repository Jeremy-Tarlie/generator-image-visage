from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Literal, TypedDict

import torch
import torch.nn as nn
import torch.nn.functional as functional
import torchvision.transforms as transforms
from PIL import Image
from PIL import ImageOps
from torch.utils.data import DataLoader, Dataset
from torchvision.utils import make_grid, save_image


@dataclass(frozen=True)
class TrainConfig:
    data_dir: Path = Path(__file__).resolve().parent / "data"
    output_model_path: Path = Path(__file__).resolve().parent / "vae_faces.pt"
    output_preview_path: Path = Path(__file__).resolve().parent / "vae_faces_samples.png"
    output_recon_path: Path = Path(__file__).resolve().parent / "vae_faces_reconstructions.png"
    image_size: int = 64
    latent_dim: int = 128
    batch_size: int = 64
    epochs: int = 60
    learning_rate: float = 4e-4
    beta_kl_max: float = 0.002
    warmup_epochs: int = 12
    num_workers: int = max(2, min(8, os.cpu_count() or 2))
    seed: int = 42


class Checkpoint(TypedDict):
    model_state_dict: dict[str, torch.Tensor]
    optimizer_state_dict: dict[str, object]
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

        # IMPORTANT: ne pas cropper directement a 64x64 sinon on garde juste
        # le centre du visage (nez/bouche), ce qui casse la generation.
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


class VAE(nn.Module):
    def __init__(self, latent_dim: int) -> None:
        super().__init__()
        self.latent_dim: Final[int] = latent_dim

        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1),   # 64 -> 32
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),  # 32 -> 16
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1), # 16 -> 8
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),# 8 -> 4
            nn.ReLU(inplace=True),
        )

        self.fc_mu = nn.Linear(256 * 4 * 4, latent_dim)
        self.fc_log_var = nn.Linear(256 * 4 * 4, latent_dim)
        self.fc_decode = nn.Linear(latent_dim, 256 * 4 * 4)

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1), # 4 -> 8
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # 8 -> 16
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),   # 16 -> 32
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1),    # 32 -> 64
            nn.Tanh(),
        )

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.encoder(x).flatten(start_dim=1)
        return self.fc_mu(features), self.fc_log_var(features)

    def reparameterize(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        decoded = self.fc_decode(z).view(-1, 256, 4, 4)
        return self.decoder(decoded)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        reconstructed = self.decode(z)
        return reconstructed, mu, log_var


def loss_function(
    reconstructed: torch.Tensor, original: torch.Tensor, mu: torch.Tensor, log_var: torch.Tensor, beta_kl: float
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # L1 donne generalement des contours plus nets que MSE pour les visages.
    reconstruction_loss = functional.l1_loss(reconstructed, original, reduction="mean")
    kl_div = -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())
    total_loss = reconstruction_loss + beta_kl * kl_div
    return total_loss, reconstruction_loss, kl_div


def denormalize(images: torch.Tensor) -> torch.Tensor:
    return ((images + 1.0) * 0.5).clamp(0.0, 1.0)


def save_preview_images(model: VAE, device: torch.device, config: TrainConfig) -> None:
    model.eval()
    with torch.inference_mode():
        latent = torch.randn(16, config.latent_dim, device=device)
        samples = denormalize(model.decode(latent).cpu())
        grid = make_grid(samples, nrow=4)
        save_image(grid, config.output_preview_path)


def save_reconstruction_preview(
    model: VAE, dataloader: DataLoader[torch.Tensor], device: torch.device, output_path: Path
) -> None:
    model.eval()
    with torch.inference_mode():
        batch = next(iter(dataloader)).to(device)
        recon, _, _ = model(batch[:8])
        originals = denormalize(batch[:8].cpu())
        reconstructions = denormalize(recon.cpu())
        comparison = torch.cat([originals, reconstructions], dim=0)
        grid = make_grid(comparison, nrow=8)
        save_image(grid, output_path)


def current_beta(epoch: int, config: TrainConfig) -> float:
    if config.warmup_epochs <= 0:
        return config.beta_kl_max
    ratio = min(float(epoch) / float(config.warmup_epochs), 1.0)
    return config.beta_kl_max * ratio


def load_checkpoint_if_available(model: VAE, optimizer: torch.optim.Optimizer, path: Path) -> int:
    if not path.exists():
        return 0
    checkpoint = torch.load(path, map_location="cpu")
    if not isinstance(checkpoint, dict):
        return 0
    if "model_state_dict" not in checkpoint or "optimizer_state_dict" not in checkpoint:
        return 0
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    last_epoch = checkpoint.get("last_epoch", 0)
    if not isinstance(last_epoch, int):
        return 0
    return max(last_epoch, 0)


def resolve_device(device_arg: DeviceArg) -> torch.device:
    if device_arg == "cpu":
        return torch.device("cpu")
    if device_arg == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA demandee mais indisponible. Verifie les drivers/CUDA Toolkit.")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


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
        **loader_kwargs,
    )

    model = VAE(config.latent_dim).to(device)
    if device.type == "cuda":
        model = model.to(memory_format=torch.channels_last)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, betas=(0.9, 0.999))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs, eta_min=1e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    start_epoch = load_checkpoint_if_available(model, optimizer, config.output_model_path) if resume else 0

    print(
        f"Dataset: {len(dataset)} images | Device: {device.type} | "
        f"Resume: {'oui' if start_epoch > 0 else 'non'} (epoch {start_epoch})",
        flush=True,
    )

    for epoch in range(start_epoch + 1, config.epochs + 1):
        model.train()
        loss_accumulator = 0.0
        recon_accumulator = 0.0
        kl_accumulator = 0.0
        beta = current_beta(epoch, config)

        for batch in dataloader:
            images = batch.to(device, non_blocking=device.type == "cuda").contiguous(memory_format=torch.channels_last)
            optimizer.zero_grad()
            with torch.amp.autocast("cuda", enabled=use_amp):
                reconstructed, mu, log_var = model(images)
                loss, reconstruction_loss, kl_div = loss_function(
                    reconstructed, images, mu, log_var, beta
                )
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            loss_accumulator += loss.item()
            recon_accumulator += reconstruction_loss.item()
            kl_accumulator += kl_div.item()

        num_batches = len(dataloader)
        avg_loss = loss_accumulator / num_batches
        avg_recon = recon_accumulator / num_batches
        avg_kl = kl_accumulator / num_batches
        print(
            f"Epoch {epoch:03d}/{config.epochs} | "
            f"loss={avg_loss:.6f} | recon={avg_recon:.6f} | kl={avg_kl:.6f} | "
            f"beta={beta:.5f} | lr={optimizer.param_groups[0]['lr']:.6f}",
            flush=True,
        )
        scheduler.step()

        checkpoint: Checkpoint = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "latent_dim": config.latent_dim,
            "image_size": config.image_size,
            "last_epoch": epoch,
        }
        torch.save(checkpoint, config.output_model_path)

    save_preview_images(model, device, config)
    save_reconstruction_preview(model, dataloader, device, config.output_recon_path)

    print(f"[OK] Modele sauvegarde: {config.output_model_path}", flush=True)
    print(f"[OK] Apercu sauvegarde: {config.output_preview_path}", flush=True)
    print(f"[OK] Reconstructions sauvegardees: {config.output_recon_path}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Entrainement VAE sur dataset de visages")
    parser.add_argument("--epochs", type=int, default=60, help="Nombre d'epoques")
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
        help="Reprend l'entrainement depuis vae_faces.pt s'il existe",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(
        TrainConfig(epochs=args.epochs, batch_size=args.batch_size, num_workers=args.num_workers),
        resume=args.resume,
        device_arg=args.device,
    )
