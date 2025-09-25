from __future__ import annotations

from typing import List

import os
import math
import pandas as pd
import seaborn as sns

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QLabel,
    QFileDialog,
    QPlainTextEdit,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QSizePolicy,
    QMessageBox,
)

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from app.ml.data import load_table


class MplCanvas(FigureCanvas):
    """Lightweight Matplotlib canvas wrapper for embedding charts in Qt.

    We use explicit Figure objects instead of the stateful pyplot API to avoid
    global side effects across charts.
    """

    def __init__(self, width: float = 6.0, height: float = 4.0, dpi: int = 100) -> None:
        self.figure = Figure(figsize=(width, height), dpi=dpi)
        super().__init__(self.figure)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.updateGeometry()


class DataExplorerWidget(QWidget):
    """Interactive data explorer: load a file, inspect summary, and visualize.

    Supports CSV, Excel, JSON, and Parquet files.
    """

    def __init__(self) -> None:
        super().__init__()

        # UI: file selection row
        top_row = QHBoxLayout()
        self.path_label = QLabel("File: not selected")
        btn_browse = QPushButton("Browse File…")
        btn_browse.clicked.connect(self.on_browse_file)
        top_row.addWidget(self.path_label)
        top_row.addWidget(btn_browse)

        # UI: tabs for Summary/Visualizations
        self.tabs = QTabWidget()

        # Summary tab: textual info + head preview
        self.summary_text = QPlainTextEdit()
        self.summary_text.setReadOnly(True)
        self.preview_table = QTableWidget()
        self.preview_table.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)

        summary_tab = QWidget()
        summary_layout = QVBoxLayout(summary_tab)
        summary_layout.addWidget(QLabel("Summary"))
        summary_layout.addWidget(self.summary_text)
        summary_layout.addWidget(QLabel("Preview (first 50 rows)"))
        summary_layout.addWidget(self.preview_table)

        # Visualization tab: correlation, distributions, missingness
        viz_tab = QWidget()
        viz_layout = QVBoxLayout(viz_tab)

        self.canvas_corr = MplCanvas(width=6, height=4)
        self.canvas_dist = MplCanvas(width=6, height=4)
        self.canvas_missing = MplCanvas(width=6, height=3)

        viz_layout.addWidget(QLabel("Correlation (numeric columns)"))
        viz_layout.addWidget(self.canvas_corr)
        viz_layout.addWidget(QLabel("Distributions (up to 6 numeric columns)"))
        viz_layout.addWidget(self.canvas_dist)
        viz_layout.addWidget(QLabel("Missing Values per Column"))
        viz_layout.addWidget(self.canvas_missing)

        self.tabs.addTab(summary_tab, "Summary")
        self.tabs.addTab(viz_tab, "Visualizations")

        # Root layout
        root = QVBoxLayout(self)
        root.addLayout(top_row)
        root.addWidget(self.tabs)

        # Internal state
        self.df: pd.DataFrame | None = None
        self.loaded_path: str | None = None

    # ---------------------------- UI Actions ---------------------------- #
    def on_browse_file(self) -> None:
        """Open file dialog and load the selected file for analysis."""
        start_dir = "data" if os.path.isdir("data") else os.getcwd()
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Data File",
            start_dir,
            "Data Files (*.csv *.txt *.xlsx *.xls *.json *.parquet *.pq);;All Files (*)",
        )
        if not path:
            return
        try:
            df = load_table(path)
        except Exception as exc:
            QMessageBox.critical(self, "Load Error", str(exc))
            return

        self.df = df
        self.loaded_path = path
        self.path_label.setText(f"File: {os.path.basename(path)}  (rows={len(df)}, cols={df.shape[1]})")

        self._update_summary(df)
        self._update_preview(df)
        self._update_visualizations(df)

    # ---------------------------- Updaters ----------------------------- #
    def _update_summary(self, df: pd.DataFrame) -> None:
        """Compute and display a concise textual summary of the dataset."""
        # Basic shape and dtypes
        lines: List[str] = []
        lines.append(f"Shape: {df.shape[0]} rows x {df.shape[1]} columns")

        dtype_counts = df.dtypes.value_counts()
        dtype_str = ", ".join([f"{dt}: {int(c)}" for dt, c in dtype_counts.items()])
        lines.append(f"Dtype counts: {dtype_str}")

        # Missing values
        missing_series = df.isna().sum()
        total_missing = int(missing_series.sum())
        lines.append(f"Missing cells (total): {total_missing}")
        if total_missing:
            top_missing = missing_series.sort_values(ascending=False).head(10)
            top_missing_str = ", ".join([f"{c}: {int(n)}" for c, n in top_missing.items()])
            lines.append(f"Top missing by column: {top_missing_str}")

        # Numeric summary
        num_cols = df.select_dtypes(include=["number"]).columns.tolist()
        cat_cols = df.select_dtypes(exclude=["number"]).columns.tolist()
        lines.append(f"Numeric columns: {len(num_cols)}; Categorical/Other: {len(cat_cols)}")

        if num_cols:
            desc = df[num_cols].describe().T
            lines.append("\nNumeric describe() (first 10 rows):")
            # Limit to first 10 rows for brevity
            with pd.option_context("display.max_rows", 10, "display.max_columns", 8):
                lines.append(desc.head(10).to_string())

        if cat_cols:
            nunique = df[cat_cols].nunique().sort_values(ascending=False).head(10)
            cat_str = ", ".join([f"{c}: {int(n)} uniq" for c, n in nunique.items()])
            lines.append("\nTop categorical columns by cardinality (first 10):")
            lines.append(cat_str)

        self.summary_text.setPlainText("\n".join(lines))

    def _update_preview(self, df: pd.DataFrame) -> None:
        """Render a small preview of the DataFrame in a simple table widget."""
        rows = min(50, len(df))
        cols = df.shape[1]
        self.preview_table.clear()
        self.preview_table.setRowCount(rows)
        self.preview_table.setColumnCount(cols)
        self.preview_table.setHorizontalHeaderLabels([str(c) for c in df.columns])
        for r in range(rows):
            for c in range(cols):
                val = df.iat[r, c]
                item = QTableWidgetItem(str(val))
                item.setFlags(item.flags() ^ Qt.ItemIsEditable)
                self.preview_table.setItem(r, c, item)
        self.preview_table.resizeColumnsToContents()

    def _update_visualizations(self, df: pd.DataFrame) -> None:
        """Draw all charts for the current dataset into their canvases."""
        # Correlation heatmap (numeric only)
        ax = self.canvas_corr.figure.clear().add_subplot(111)
        num_df = df.select_dtypes(include=["number"]).copy()
        if not num_df.empty and num_df.shape[1] >= 2:
            corr = num_df.corr(numeric_only=True)
            sns.heatmap(corr, ax=ax, cmap="coolwarm", center=0.0)
            ax.set_title("Correlation Heatmap")
        else:
            ax.text(0.5, 0.5, "Not enough numeric columns for correlation", ha="center", va="center")
            ax.axis("off")
        self.canvas_corr.draw()

        # Distributions: up to 6 numeric columns
        self.canvas_dist.figure.clear()
        if not num_df.empty:
            cols = num_df.columns.tolist()[:6]
            n = len(cols)
            nrows = math.ceil(n / 3)
            ncols = min(3, n)
            for idx, col in enumerate(cols, start=1):
                axd = self.canvas_dist.figure.add_subplot(nrows, ncols, idx)
                sns.histplot(num_df[col].dropna(), kde=True, ax=axd)
                axd.set_title(col)
            self.canvas_dist.figure.tight_layout()
        else:
            axd = self.canvas_dist.figure.add_subplot(111)
            axd.text(0.5, 0.5, "No numeric columns to plot", ha="center", va="center")
            axd.axis("off")
        self.canvas_dist.draw()

        # Missingness per column
        axm = self.canvas_missing.figure.clear().add_subplot(111)
        missing = df.isna().sum()
        if missing.sum() > 0:
            missing = missing[missing > 0].sort_values(ascending=False)
            sns.barplot(x=missing.values, y=missing.index, ax=axm, orient="h")
            axm.set_xlabel("Missing Count")
            axm.set_ylabel("Column")
            axm.set_title("Missing Values by Column")
        else:
            axm.text(0.5, 0.5, "No missing values detected", ha="center", va="center")
            axm.axis("off")
        self.canvas_missing.draw()

