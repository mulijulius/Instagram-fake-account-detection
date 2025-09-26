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
    QApplication,
)
from PySide6.QtCore import Qt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas, NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from datetime import datetime

from app.ml.data import load_dataset
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

        self.dataset_path: str | None = None
        self.scaler_path = ".artifacts/scaler.pkl"
        self.autoencoder_path = ".artifacts/autoencoder.pt"
        self.gan_discriminator_path = ".artifacts/gan_discriminator.pt"

        os.makedirs(".artifacts", exist_ok=True)
        os.makedirs("data", exist_ok=True)
        os.makedirs("results", exist_ok=True)

        # Blue theme styling
        self.setStyleSheet(
            """
            QMainWindow { background-color: #0d47a1; color: white; }
            QLabel { color: white; }
            QPushButton { background-color: #1976d2; color: white; border: none; padding: 6px 10px; }
            QPushButton:hover { background-color: #1e88e5; }
            QLineEdit, QDoubleSpinBox, QSpinBox { background: white; color: #0d47a1; }
            """
        )

        container = QWidget()
        layout = QVBoxLayout(container)

        # Dataset loader
        dataset_row = QHBoxLayout()
        self.dataset_label = QLabel("Dataset: not selected")
        btn_browse = QPushButton("Browse CSV…")
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

        # Fusion and threshold
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

        # Predict single CSV
        predict_row = QHBoxLayout()
        btn_predict = QPushButton("Predict on CSV…")
        btn_predict.clicked.connect(self.on_predict)
        predict_row.addWidget(btn_predict)

        # Synthetic data generation
        synth_row = QHBoxLayout()
        self.synth_num = QSpinBox(); self.synth_num.setRange(1, 100000); self.synth_num.setValue(500)
        btn_synth = QPushButton("Generate Fake Samples")
        btn_synth.clicked.connect(self.on_generate_synth)
        synth_row.addWidget(QLabel("N synth")); synth_row.addWidget(self.synth_num)
        synth_row.addWidget(btn_synth)

        layout.addLayout(dataset_row)
        layout.addLayout(norm_row)
        layout.addLayout(train_row)
        layout.addLayout(fusion_row)
        layout.addLayout(predict_row)
        layout.addLayout(synth_row)

        # Realtime visualization area (Matplotlib)
        self.figure = Figure(figsize=(8, 6), facecolor="#0d47a1")
        self.ax_ae = self.figure.add_subplot(2, 1, 1)
        self.ax_gan = self.figure.add_subplot(2, 1, 2)

        for ax in (self.ax_ae, self.ax_gan):
            ax.set_facecolor("#0d47a1")
            ax.tick_params(colors="white")
            for spine in ax.spines.values():
                spine.set_color("white")
            ax.title.set_color("white")
            ax.yaxis.label.set_color("white")
            ax.xaxis.label.set_color("white")
            ax.grid(True, color="white", alpha=0.12)

        self.ax_ae.set_title("Autoencoder Loss")
        self.ax_ae.set_xlabel("Epoch")
        self.ax_ae.set_ylabel("MSE")
        self.ax_gan.set_title("GAN Losses")
        self.ax_gan.set_xlabel("Epoch")
        self.ax_gan.set_ylabel("BCE")

        self.epoch_history_ae: list[int] = []
        self.loss_history_ae: list[float] = []
        self.epoch_history_gan: list[int] = []
        self.g_loss_history: list[float] = []
        self.d_loss_history: list[float] = []

        (self.line_ae,) = self.ax_ae.plot([], [], color="#64b5f6", label="AE Loss")
        (self.line_g_gan,) = self.ax_gan.plot([], [], color="#90caf9", label="G Loss")
        (self.line_d_gan,) = self.ax_gan.plot([], [], color="#1565c0", label="D Loss")
        leg = self.ax_gan.legend()
        if leg is not None:
            leg.get_frame().set_facecolor("#0d47a1")
            leg.get_frame().set_edgecolor("white")
            for t in leg.get_texts():
                t.set_color("white")

        self.canvas = FigureCanvas(self.figure)
        toolbar = NavigationToolbar(self.canvas, self)
        plot_box = QVBoxLayout()
        plot_box.addWidget(toolbar)
        plot_box.addWidget(self.canvas)
        layout.addLayout(plot_box)

        # Plot controls
        plot_controls = QHBoxLayout()
        btn_save_plot = QPushButton("Save Plot")
        btn_save_plot.clicked.connect(self.on_save_plot)
        btn_clear_plot = QPushButton("Clear Plot")
        btn_clear_plot.clicked.connect(self.on_clear_plot)
        plot_controls.addWidget(btn_save_plot)
        plot_controls.addWidget(btn_clear_plot)
        layout.addLayout(plot_controls)

        self.setCentralWidget(container)

        # Set size to at least 3/4 of available screen
        screen = QApplication.primaryScreen()
        if screen is not None:
            geo = screen.availableGeometry()
            self.resize(int(geo.width() * 0.75), int(geo.height() * 0.75))

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

        def ae_on_epoch(epoch: int, loss: float) -> None:
            self.epoch_history_ae.append(epoch)
            self.loss_history_ae.append(loss)
            self.line_ae.set_data(self.epoch_history_ae, self.loss_history_ae)
            self.ax_ae.relim(); self.ax_ae.autoscale_view()
            self.canvas.draw_idle()
            QApplication.processEvents()

        trainer.train(
            Xn[y == 0],
            epochs=int(self.ae_epochs.value()),
            lr=float(self.ae_lr.value()),
            on_epoch=ae_on_epoch,
        )
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

        def gan_on_epoch(epoch: int, g_loss: float, d_loss: float) -> None:
            self.epoch_history_gan.append(epoch)
            self.g_loss_history.append(g_loss)
            self.d_loss_history.append(d_loss)
            self.line_g_gan.set_data(self.epoch_history_gan, self.g_loss_history)
            self.line_d_gan.set_data(self.epoch_history_gan, self.d_loss_history)
            self.ax_gan.relim(); self.ax_gan.autoscale_view()
            self.canvas.draw_idle()
            QApplication.processEvents()

        trainer.train(
            fakes=Xn[y == 1],
            epochs=int(self.gan_epochs.value()),
            lr=float(self.gan_lr.value()),
            on_epoch=gan_on_epoch,
        )
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
        out = save_synthetic_samples(
            n,
            out_csv="results/synthetic_fakes.csv",
            discriminator_path=self.gan_discriminator_path,
        )
        QMessageBox.information(self, "Synthetic", f"Saved {out} samples to results/synthetic_fakes.csv")

    def on_save_plot(self) -> None:
        ts = datetime.now().strftime("%Y%m%d-%H%M%S")
        out_path = os.path.join("results", f"training_plot_{ts}.png")
        self.figure.savefig(out_path, dpi=150, facecolor=self.figure.get_facecolor())
        QMessageBox.information(self, "Plot Saved", f"Saved to {out_path}")

    def on_clear_plot(self) -> None:
        self.epoch_history_ae.clear()
        self.loss_history_ae.clear()
        self.epoch_history_gan.clear()
        self.g_loss_history.clear()
        self.d_loss_history.clear()
        self.line_ae.set_data([], [])
        self.line_g_gan.set_data([], [])
        self.line_d_gan.set_data([], [])
        self.ax_ae.relim(); self.ax_ae.autoscale_view()
        self.ax_gan.relim(); self.ax_gan.autoscale_view()
        self.canvas.draw_idle()

