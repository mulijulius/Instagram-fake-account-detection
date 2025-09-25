import numpy as np
from typing import Tuple

from app.ml.autoencoder import load_autoencoder, reconstruction_errors
from app.ml.gan import load_gan_discriminator, discriminator_probabilities


def minmax_normalize(values: np.ndarray) -> np.ndarray:
    vmin = float(values.min())
    vmax = float(values.max())
    if vmax - vmin < 1e-12:
        return np.zeros_like(values)
    return (values - vmin) / (vmax - vmin)


class FusionScorer:
    def __init__(self, ae_path: str, gan_discriminator_path: str, device: str = "cpu"):
        self.ae = load_autoencoder(ae_path, device=device)
        self.disc = load_gan_discriminator(gan_discriminator_path, device=device)
        self.device = device

    def score(self, X: np.ndarray, alpha: float = 0.5, beta: float = 0.5) -> np.ndarray:
        if self.ae is None or self.disc is None:
            raise RuntimeError("Models not available. Train autoencoder and GAN first.")
        R = reconstruction_errors(self.ae, X, device=self.device)
        Rn = minmax_normalize(R)
        P = discriminator_probabilities(self.disc, X, device=self.device)
        final = alpha * Rn + beta * P
        return final

    def predict(self, X: np.ndarray, alpha: float = 0.5, beta: float = 0.5, threshold: float = 0.5) -> Tuple[np.ndarray, np.ndarray]:
        scores = self.score(X, alpha=alpha, beta=beta)
        preds = (scores >= threshold).astype(int)
        return scores, preds
