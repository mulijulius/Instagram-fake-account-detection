import numpy as np
import pandas as pd
import torch

from app.ml.gan import load_gan_discriminator, load_gan_generator


def save_synthetic_samples(n: int, out_csv: str, discriminator_path: str) -> int:
    disc = load_gan_discriminator(discriminator_path)
    if disc is None:
        return 0
    # infer generator path from discriminator
    gen_path = discriminator_path.replace("discriminator", "generator")
    G = load_gan_generator(gen_path)
    input_dim = next(disc.parameters()).shape[1]
    if G is None:
        # fallback: gaussian noise
        X = np.random.normal(0.0, 1.0, size=(n, input_dim)).astype(np.float32)
    else:
        noise_dim = next(G.parameters()).shape[1]
        with torch.no_grad():
            z = torch.randn(n, noise_dim)
            X = G(z).cpu().numpy().astype(np.float32)
    df = pd.DataFrame(X, columns=[f"f_{i}" for i in range(input_dim)])
    df.to_csv(out_csv, index=False)
    return int(len(df))
