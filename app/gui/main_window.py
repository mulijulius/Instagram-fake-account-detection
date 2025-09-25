from PySide6.QtWidgets import (
    QMainWindow,
    QWidget,
    QFileDialog,
    QPushButton,
    QLabel,
    QVBoxLayout,
    QHBoxLayout,
    QLineEdit,
    QDoubleSpinBox,
    QSpinBox,
    QMessageBox,
    QTabWidget,
)
from PySide6.QtCore import Qt

from app.ml.data import load_dataset
from app.gui.data_explorer import DataExplorerWidget
from app.ml.features import select_feature_columns, split_features_labels
from app.ml.normalize import build_scaler, apply_scaler, save_scaler, load_scaler
from app.ml.autoencoder import AutoencoderTrainer
from app.ml.gan import TabularGANTrainer, load_gan_discriminator
from app.ml.fusion import FusionScorer
from app.ml.evaluate import evaluate_models
from app.ml.synth import save_synthetic_samples

import os


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Instagram Fake Account Detection")

        # File paths for persisted artifacts produced by the modeling workflow
        self.dataset_path: str | None = None
        self.scaler_path = ".artifacts/scaler.pkl"
        self.autoencoder_path = ".artifacts/autoencoder.pt"
        self.gan_discriminator_path = ".artifacts/gan_discriminator.pt"

        # Ensure expected project folders exist at runtime
        os.makedirs(".artifacts", exist_ok=True)
        os.makedirs("data", exist_ok=True)

        # Build the application as a 2-tab interface:
        # 1) Model Trainer (existing controls)
        # 2) Data Explorer (new: upload/analyze/visualize any tabular file)
        tabs = QTabWidget()
        tabs.addTab(self._build_model_tab(), "Model Trainer")
        tabs.addTab(DataExplorerWidget(), "Data Explorer")
        self.setCentralWidget(tabs)

    def _build_model_tab(self) -> QWidget:
        """Construct and return the Model Trainer tab content as a QWidget."""
        container = QWidget()
        layout = QVBoxLayout(container)

        # Dataset loader row
        dataset_row = QHBoxLayout()
        self.dataset_label = QLabel("Dataset: not selected")
        btn_browse = QPushButton("Browse Dataset…")
        btn_browse.clicked.connect(self.on_browse)
        dataset_row.addWidget(self.dataset_label)
        dataset_row.addWidget(btn_browse)

        # Normalization controls
        norm_row = QHBoxLayout()
        self.norm_method = QLineEdit("zscore")
        self.norm_method.setPlaceholderText("zscore|minmax")
        btn_fit_scaler = QPushButton("Fit Scaler")
        btn_fit_scaler.clicked.connect(self.on_fit_scaler)
        norm_row.addWidget(QLabel("Normalize:"))
        norm_row.addWidget(self.norm_method)
        norm_row.addWidget(btn_fit_scaler)

        # Training controls
        train_row = QHBoxLayout()
        self.ae_epochs = QSpinBox(); self.ae_epochs.setRange(1, 10000); self.ae_epochs.setValue(25)
        self.ae_lr = QDoubleSpinBox(); self.ae_lr.setDecimals(5); self.ae_lr.setRange(1e-6, 1.0); self.ae_lr.setSingleStep(0.0001); self.ae_lr.setValue(0.001)
        btn_train_ae = QPushButton("Train Autoencoder")
        btn_train_ae.clicked.connect(self.on_train_autoencoder)

        self.gan_epochs = QSpinBox(); self.gan_epochs.setRange(1, 10000); self.gan_epochs.setValue(50)
        self.gan_lr = QDoubleSpinBox(); self.gan_lr.setDecimals(5); self.gan_lr.setRange(1e-6, 1.0); self.gan_lr.setSingleStep(0.0001); self.gan_lr.setValue(0.0005)
        btn_train_gan = QPushButton("Train GAN")
        btn_train_gan.clicked.connect(self.on_train_gan)

        train_row.addWidget(QLabel("AE epochs")); train_row.addWidget(self.ae_epochs)
        train_row.addWidget(QLabel("AE lr")); train_row.addWidget(self.ae_lr)
        train_row.addWidget(btn_train_ae)
        train_row.addSpacing(12)
        train_row.addWidget(QLabel("GAN epochs")); train_row.addWidget(self.gan_epochs)
        train_row.addWidget(QLabel("GAN lr")); train_row.addWidget(self.gan_lr)
        train_row.addWidget(btn_train_gan)

        # Fusion and threshold row
        fusion_row = QHBoxLayout()
        self.alpha = QDoubleSpinBox(); self.alpha.setRange(0.0, 1.0); self.alpha.setSingleStep(0.05); self.alpha.setValue(0.5)
        self.beta = QDoubleSpinBox(); self.beta.setRange(0.0, 1.0); self.beta.setSingleStep(0.05); self.beta.setValue(0.5)
        self.threshold = QDoubleSpinBox(); self.threshold.setRange(0.0, 1.0); self.threshold.setSingleStep(0.01); self.threshold.setValue(0.5)
        btn_eval = QPushButton("Evaluate")
        btn_eval.clicked.connect(self.on_evaluate)
        fusion_row.addWidget(QLabel("alpha")); fusion_row.addWidget(self.alpha)
        fusion_row.addWidget(QLabel("beta")); fusion_row.addWidget(self.beta)
        fusion_row.addWidget(QLabel("Threshold T")); fusion_row.addWidget(self.threshold)
        fusion_row.addWidget(btn_eval)

        # Predict on CSV row
        predict_row = QHBoxLayout()
        btn_predict = QPushButton("Predict on CSV…")
        btn_predict.clicked.connect(self.on_predict)
        predict_row.addWidget(btn_predict)

        # Synthetic data generation row
        synth_row = QHBoxLayout()
        self.synth_num = QSpinBox(); self.synth_num.setRange(1, 100000); self.synth_num.setValue(500)
        btn_synth = QPushButton("Generate Fake Samples")
        btn_synth.clicked.connect(self.on_generate_synth)
        synth_row.addWidget(QLabel("N synth")); synth_row.addWidget(self.synth_num)
        synth_row.addWidget(btn_synth)

        # Assemble tab layout
        layout.addLayout(dataset_row)
        layout.addLayout(norm_row)
        layout.addLayout(train_row)
        layout.addLayout(fusion_row)
        layout.addLayout(predict_row)
        layout.addLayout(synth_row)

        return container

    def on_browse(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Select CSV Dataset", "data", "CSV Files (*.csv)")
        if path:
            self.dataset_path = path
            self.dataset_label.setText(f"Dataset: {os.path.basename(path)}")

    def _require_dataset(self) -> str | None:
        if not self.dataset_path:
            QMessageBox.warning(self, "Dataset", "Please select a dataset CSV first.")
            return None
        return self.dataset_path

    def on_fit_scaler(self) -> None:
        path = self._require_dataset()
        if not path:
            return
        df = load_dataset(path)
        X_cols = select_feature_columns(df)
        X, y = split_features_labels(df, X_cols)
        scaler = build_scaler(self.norm_method.text().strip() or "zscore", X)
        save_scaler(self.scaler_path, scaler, X_cols)
        QMessageBox.information(self, "Scaler", f"Scaler fitted and saved to {self.scaler_path}")

    def on_train_autoencoder(self) -> None:
        path = self._require_dataset()
        if not path:
            return
        df = load_dataset(path)
        X_cols = select_feature_columns(df)
        X, y = split_features_labels(df, X_cols)
        scaler, cols = load_scaler(self.scaler_path)
        if scaler is None:
            QMessageBox.warning(self, "Scaler", "Fit scaler first.")
            return
        Xn = apply_scaler(scaler, X[cols])
        # train autoencoder on real-only (y==0)
        trainer = AutoencoderTrainer(input_dim=Xn.shape[1])
        trainer.train(Xn[y == 0], epochs=int(self.ae_epochs.value()), lr=float(self.ae_lr.value()))
        trainer.save(self.autoencoder_path)
        QMessageBox.information(self, "Autoencoder", f"Saved to {self.autoencoder_path}")

    def on_train_gan(self) -> None:
        path = self._require_dataset()
        if not path:
            return
        df = load_dataset(path)
        X_cols = select_feature_columns(df)
        X, y = split_features_labels(df, X_cols)
        scaler, cols = load_scaler(self.scaler_path)
        if scaler is None:
            QMessageBox.warning(self, "Scaler", "Fit scaler first.")
            return
        Xn = apply_scaler(scaler, X[cols])
        trainer = TabularGANTrainer(input_dim=Xn.shape[1])
        trainer.train(fakes=Xn[y == 1], epochs=int(self.gan_epochs.value()), lr=float(self.gan_lr.value()))
        trainer.save(discriminator_path=self.gan_discriminator_path)
        QMessageBox.information(self, "GAN", f"Discriminator saved to {self.gan_discriminator_path}")

    def on_evaluate(self) -> None:
        path = self._require_dataset()
        if not path:
            return
        df = load_dataset(path)
        X_cols = select_feature_columns(df)
        X, y = split_features_labels(df, X_cols)
        scaler, cols = load_scaler(self.scaler_path)
        if scaler is None:
            QMessageBox.warning(self, "Scaler", "Fit scaler first.")
            return
        Xn = apply_scaler(scaler, X[cols])
        scorer = FusionScorer(ae_path=self.autoencoder_path, gan_discriminator_path=self.gan_discriminator_path)
        alpha = float(self.alpha.value()); beta = float(self.beta.value())
        thr = float(self.threshold.value())
        metrics = evaluate_models(scorer, Xn, y, alpha=alpha, beta=beta, threshold=thr)
        msg = "\n".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
        QMessageBox.information(self, "Evaluation", msg)

    def on_predict(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Select CSV for Prediction", "data", "CSV Files (*.csv)")
        if not path:
            return
        df = load_dataset(path)
        X_cols = select_feature_columns(df)
        X, _ = split_features_labels(df, X_cols, has_label=False)
        scaler, cols = load_scaler(self.scaler_path)
        if scaler is None:
            QMessageBox.warning(self, "Scaler", "Fit scaler first.")
            return
        Xn = apply_scaler(scaler, X[cols])
        scorer = FusionScorer(ae_path=self.autoencoder_path, gan_discriminator_path=self.gan_discriminator_path)
        alpha = float(self.alpha.value()); beta = float(self.beta.value()); thr = float(self.threshold.value())
        scores, preds = scorer.predict(Xn, alpha=alpha, beta=beta, threshold=thr)
        fake_rate = float((preds == 1).mean()) if len(preds) else 0.0
        QMessageBox.information(self, "Prediction", f"Samples: {len(preds)}\nFake rate: {fake_rate:.2%}")

    def on_generate_synth(self) -> None:
        n = int(self.synth_num.value())
        disc = load_gan_discriminator(self.gan_discriminator_path)
        if disc is None:
            QMessageBox.warning(self, "GAN", "Train GAN first.")
            return
        # Use the trainer class to sample via generator weights stored internally if implemented
        # For simplicity, we rely on synth module to generate tabular noise matching discriminator input size
        out = save_synthetic_samples(n, out_csv="data/synthetic_fakes.csv", discriminator_path=self.gan_discriminator_path)
        QMessageBox.information(self, "Synthetic", f"Saved {out} samples to data/synthetic_fakes.csv")

