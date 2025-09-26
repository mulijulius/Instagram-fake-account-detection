from dataclasses import dataclass
from typing import Optional, Callable
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np


class Generator(nn.Module):
    def __init__(self, noise_dim: int, output_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(noise_dim, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, output_dim),
        )

    def forward(self, z):
        return self.net(z)


class Discriminator(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128), nn.LeakyReLU(0.2),
            nn.Linear(128, 128), nn.LeakyReLU(0.2),
            nn.Linear(128, 1), nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x)


@dataclass
class TabularGANTrainer:
    input_dim: int
    noise_dim: int = 32
    batch_size: int = 128
    device: str = "cpu"

    def __post_init__(self):
        self.G = Generator(self.noise_dim, self.input_dim).to(self.device)
        self.D = Discriminator(self.input_dim).to(self.device)

    def train(
        self,
        fakes: np.ndarray,
        epochs: int = 50,
        lr: float = 5e-4,
        on_epoch: Optional[Callable[[int, float, float], None]] = None,
    ) -> None:
        X = np.asarray(fakes, dtype=np.float32)
        dataset = TensorDataset(torch.from_numpy(X))
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, drop_last=False)

        opt_G = torch.optim.Adam(self.G.parameters(), lr=lr, betas=(0.5, 0.999))
        opt_D = torch.optim.Adam(self.D.parameters(), lr=lr, betas=(0.5, 0.999))
        bce = nn.BCELoss()

        for epoch_index in range(1, epochs + 1):
            g_loss_sum = 0.0
            d_loss_sum = 0.0
            batch_count = 0
            for (xb,) in loader:
                xb = xb.to(self.device)
                bs = xb.size(0)

                # Train Discriminator
                z = torch.randn(bs, self.noise_dim, device=self.device)
                xg = self.G(z).detach()
                d_real = self.D(xb)
                d_fake = self.D(xg)
                loss_D = bce(d_real, torch.ones_like(d_real)) + bce(d_fake, torch.zeros_like(d_fake))
                opt_D.zero_grad(); loss_D.backward(); opt_D.step()

                # Train Generator
                z = torch.randn(bs, self.noise_dim, device=self.device)
                xg = self.G(z)
                d_fake = self.D(xg)
                loss_G = bce(d_fake, torch.ones_like(d_fake))
                opt_G.zero_grad(); loss_G.backward(); opt_G.step()
                # accumulate losses for reporting
                d_loss_sum += float(loss_D.item())
                g_loss_sum += float(loss_G.item())
                batch_count += 1
            if on_epoch is not None and batch_count > 0:
                on_epoch(epoch_index, g_loss_sum / batch_count, d_loss_sum / batch_count)

    def save(self, discriminator_path: str, generator_path: Optional[str] = None) -> None:
        torch.save({
            "input_dim": self.input_dim,
            "noise_dim": self.noise_dim,
            "D_state": self.D.state_dict(),
        }, discriminator_path)
        if generator_path is None:
            if "discriminator" in discriminator_path:
                generator_path = discriminator_path.replace("discriminator", "generator")
            else:
                generator_path = discriminator_path + ".generator.pt"
        torch.save({
            "input_dim": self.input_dim,
            "noise_dim": self.noise_dim,
            "G_state": self.G.state_dict(),
        }, generator_path)


def load_gan_discriminator(path: str, device: str = "cpu") -> Optional[Discriminator]:
    try:
        ckpt = torch.load(path, map_location=device)
    except FileNotFoundError:
        return None
    model = Discriminator(ckpt["input_dim"]).to(device)
    model.load_state_dict(ckpt["D_state"])
    model.eval()
    return model


def load_gan_generator(path: str, device: str = "cpu") -> Optional[Generator]:
    try:
        ckpt = torch.load(path, map_location=device)
    except FileNotFoundError:
        return None
    model = Generator(ckpt["noise_dim"], ckpt["input_dim"]).to(device)
    model.load_state_dict(ckpt["G_state"])
    model.eval()
    return model


def discriminator_probabilities(model: Discriminator, X: np.ndarray, device: str = "cpu") -> np.ndarray:
    with torch.no_grad():
        X = torch.from_numpy(np.asarray(X, dtype=np.float32)).to(device)
        probs = model(X).squeeze(1).cpu().numpy()
    return probs
