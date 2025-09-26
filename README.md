## Instagram Fake Account Detection - Desktop App

### Overview
Python desktop application implementing a hybrid detection system:
- Autoencoder trained on real accounts
- GAN trained on fake accounts for augmentation and discriminator scoring
- Fusion scoring of reconstruction error and discriminator probability

### Features
- Load labeled CSV datasets (real/fake)
  - Significance: unified entry point for datasets; ensures consistent feature selection.
- Feature extraction and normalization
  - Significance: standardizes inputs for stable learning and fair scoring.
- Train Autoencoder (real-only)
  - Significance: models normal behavior; reconstruction error is an anomaly signal.
- Train tabular GAN (fake-only)
  - Significance: discriminator probability complements AE error with a discriminative signal.
- Generate synthetic fake samples
  - Significance: supports augmentation and stress-testing of the pipeline.
- Fusion scoring and thresholding
  - Significance: combines complementary signals with weights (alpha, beta) for robust detection.
- Evaluation (Precision, Recall, F1, ROC-AUC, PR-AUC)
  - Significance: covers thresholded and threshold-free performance views.
- GUI with blue theme, wide layout, realtime training plots, and advanced visualizations
  - Significance: enables interactive experimentation and monitoring.
- Save visualizations and generated samples under `results/`
  - Significance: ensures reproducibility and reporting.

### Quick Start
1) Create virtual environment
```bash
python3 -m venv .venv && source .venv/bin/activate
```
2) Install dependencies
```bash
pip install -r requirements.txt
```
3) Launch app
```bash
python -m app
```

If you run into display issues on a remote server, ensure a desktop session is available or use an X server.

### Building a Single Executable
You can compile the entire source into a single executable using PyInstaller.

1) Install build tooling (in your virtualenv):
```bash
pip install pyinstaller
```

2) Build the app (POSIX):
```bash
pyinstaller --noconfirm \
  --name InstagramFakeAccountDetector \
  --onefile \
  --windowed \
  --add-data "data:data" \
  --add-data "results:results" \
  --add-data ".artifacts:.artifacts" \
  app/__main__.py
```

Notes:
- On Linux, the executable will land in `dist/InstagramFakeAccountDetector`.
- `--windowed` hides the console window for GUI apps.
- If packaging fails due to hidden imports, add `--hidden-import` for missing modules (e.g., some `sklearn` subpackages).

3) Run the binary:
```bash
./dist/InstagramFakeAccountDetector
```

### Data Format (CSV)
Expected columns (example, extend as needed):
- profile features: followers, following, bio_length, has_profile_pic
- metadata: account_age_days, verified, posts_count, post_frequency
- caption/text: avg_caption_len, hashtag_count
- activity: avg_likes, avg_comments, engagement_rate, posting_variance
- label: 0 for real, 1 for fake

See `data/sample_dataset.csv` for a template.

### Project Structure
```
app/
  __init__.py
  __main__.py
  gui/
    __init__.py
    main_window.py
  ml/
    __init__.py
    data.py
    features.py
    normalize.py
    autoencoder.py
    gan.py
    fusion.py
    evaluate.py
    synth.py
requirements.txt
README.md
data/
  sample_dataset.csv
results/
  (generated at runtime: plots, synthetic samples)
```

### Notes
- Models and scalers are saved under `.artifacts/` in the project root.
- Realtime training plots are visible in the GUI and can be saved as PNG into `results/`.
- Synthetic samples are written to `results/synthetic_fakes.csv`.
- This code targets CPU by default; enable GPU in PyTorch if available.

### How it works (In-depth)
1. Data ingestion: You load a labeled CSV. Feature columns are selected via `app/ml/features.py`. The label is assumed binary: `0` (real) and `1` (fake).
2. Normalization: You choose a normalization method (e.g., `zscore` or `minmax`). A scaler is fitted with the selected features and saved to `.artifacts/scaler.pkl` for reuse.
3. Autoencoder: Trained only on real samples (`y == 0`). It learns to reconstruct normal patterns. Reconstruction error serves as an anomaly score.
4. GAN: A tabular GAN is trained using the fake samples (`y == 1`). Its discriminator learns to distinguish generated/real-like patterns and provides a complementary probability signal.
5. Fusion: The fusion scorer combines the autoencoder reconstruction error and GAN discriminator probability using weights `alpha` and `beta`. With a threshold `T`, samples are classified as fake (1) or real (0).
6. Evaluation: On a labeled dataset, the app computes Precision, Recall, F1, and ROC-AUC.
7. Realtime visualization: During training, losses are reported back to the GUI and plotted live using Matplotlib (blue-themed). You can clear or save the plot to `results/` at any time.
8. Synthetic generation: After training the GAN, you can generate synthetic samples and save them to `results/synthetic_fakes.csv` for analysis or augmentation experiments.

### Using the GUI
- Dataset: Click "Browse CSV…" and pick a file under `data/` (or anywhere).
- Normalize: Choose method and click "Fit Scaler".
- Train Autoencoder: Set epochs/lr, click "Train Autoencoder". Watch the AE loss plot update.
- Train GAN: Set epochs/lr, click "Train GAN". Watch G/D losses update.
- Evaluate: Set `alpha`, `beta`, and threshold `T`, then click "Evaluate".
- Predict on CSV: Score an unlabeled CSV and view basic statistics.
- Generate Fake Samples: Choose `N synth` and generate to `results/synthetic_fakes.csv`.
- Save/Clear Plot: Save the current plot to `results/` or clear histories.

#### GUI Controls & Significance
- Browse CSV…: select input dataset; required for all subsequent steps.
- Normalize + Fit Scaler: fits and stores scaler for consistent preprocessing.
- Train Autoencoder: trains AE on real samples; monitors reconstruction MSE.
- Train GAN: trains tabular GAN on fake samples; monitors generator/discriminator loss.
- Evaluate: computes metrics and renders ROC, PR, confusion matrix, and score histogram.
- Predict on CSV…: scores unlabeled data using current fusion settings.
- Generate Fake Samples: creates synthetic fakes to `results/synthetic_fakes.csv`.
- Save Plot / Clear Plot: export all subplots or reset the canvas.

#### Visualizations and Captions
- Autoencoder Loss — caption: "Autoencoder training loss over epochs (MSE)"
- GAN Losses — caption: "Generator and Discriminator training losses (BCE)"
- ROC Curve — caption: "ROC curve (AUC computed after Evaluate)"
- Precision-Recall Curve — caption: "Precision-Recall curve (AUPRC computed after Evaluate)"
- Confusion Matrix — caption: "Confusion matrix at current threshold"
- Score Distribution — caption: "Score histogram by class; vertical line = threshold"

### Example runs
Below are three example scenarios you can try after launching the GUI with `python -m app`:

1) Quick end-to-end on the sample dataset
   - Browse to `data/sample_dataset.csv`.
   - Click "Fit Scaler" (leave `Normalize` as `zscore`).
   - Train Autoencoder for 10 epochs (AE lr: 0.001), watch the AE loss curve.
   - Train GAN for 20 epochs (GAN lr: 0.0005), watch G/D losses.
   - Click "Evaluate" with `alpha=0.5`, `beta=0.5`, `T=0.5` and review metrics.
   - Click "Save Plot" to store the combined plot under `results/`.

2) Autoencoder-focused training and evaluation
   - Use `data/train.csv` (or your dataset), "Fit Scaler" with `minmax`.
   - Train Autoencoder for 25 epochs.
   - Skip GAN training (you can evaluate with AE alone using `alpha>0`, `beta=0`).
   - Evaluate with `alpha=1.0`, `beta=0.0`, `T=0.6`. Save plot to `results/`.

3) GAN-focused training plus synthetic data generation
   - Use `data/train.csv`, "Fit Scaler" with `zscore`.
   - Train Autoencoder briefly (e.g., 5 epochs) to enable fusion later.
   - Train GAN for 50 epochs.
   - Click "Generate Fake Samples" with `N synth = 1000`.
   - Inspect `results/synthetic_fakes.csv` and save the training plot to `results/`.

# Instagram-fake-account-detection
Detect fake Instagram post from real ones
