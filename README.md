## Instagram Fake Account Detection - Desktop App

### Overview
Python desktop application implementing a hybrid detection system:
- Autoencoder trained on real accounts
- GAN trained on fake accounts for augmentation and discriminator scoring
- Fusion scoring of reconstruction error and discriminator probability

### Features
- Load labeled CSV datasets (real/fake)
- Feature extraction and normalization
- Train Autoencoder (real-only)
- Train tabular GAN (fake-only)
- Generate synthetic fake samples
- Fusion scoring and thresholding
- Evaluation (Precision, Recall, F1, ROC-AUC)
- GUI to orchestrate training, evaluation, and predictions

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
```

### Notes
- Models and scalers are saved under `.artifacts/` in the project root.
- This code targets CPU by default; enable GPU in PyTorch if available.

# Instagram-fake-account-detection
Detect fake Instagram post from real ones
