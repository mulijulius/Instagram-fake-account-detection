from dataclasses import dataclass
from typing import Optional
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np


class SimpleAutoencoder(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int = 16):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, max(64, input_dim * 2)),
            nn.ReLU(),
            nn.Linear(max(64, input_dim * 2), max(32, input_dim)),
            nn.ReLU(),
            nn.Linear(max(32, input_dim), latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, max(32, input_dim)),
            nn.ReLU(),
            nn.Linear(max(32, input_dim), max(64, input_dim * 2)),
            nn.ReLU(),
            nn.Linear(max(64, input_dim * 2), input_dim),
        )

    def forward(self, x):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat


@dataclass
class AutoencoderTrainer:
    input_dim: int
    latent_dim: int = 16
    batch_size: int = 128
    device: str = "cpu"

    def __post_init__(self):
        self.model = SimpleAutoencoder(self.input_dim, self.latent_dim).to(self.device)

    def train(self, X: np.ndarray, epochs: int = 25, lr: float = 1e-3) -> None:
        X = np.asarray(X, dtype=np.float32)
        dataset = TensorDataset(torch.from_numpy(X))
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, drop_last=False)
        opt = torch.optim.Adam(self.model.parameters(), lr=lr)
        loss_fn = nn.MSELoss()

        self.model.train()
        for _ in range(epochs):
            for (xb,) in loader:
                xb = xb.to(self.device)
                x_hat = self.model(xb)
                loss = loss_fn(x_hat, xb)
                opt.zero_grad()
                loss.backward()
                opt.step()

    def save(self, path: str) -> None:
        torch.save({
            "state_dict": self.model.state_dict(),
            "input_dim": self.input_dim,
            "latent_dim": self.latent_dim,
        }, path)


def load_autoencoder(path: str, device: str = "cpu") -> Optional[SimpleAutoencoder]:
    try:
        ckpt = torch.load(path, map_location=device)
    except FileNotFoundError:
        return None
    model = SimpleAutoencoder(ckpt["input_dim"], ckpt["latent_dim"]).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model


def reconstruction_errors(model: SimpleAutoencoder, X: np.ndarray, device: str = "cpu") -> np.ndarray:
    with torch.no_grad():
        X = torch.from_numpy(np.asarray(X, dtype=np.float32)).to(device)
        X_hat = model(X)
        errs = ((X_hat - X) ** 2).mean(dim=1).cpu().numpy()
    return errs
