import os
import sys
import subprocess
from typing import Dict, List, Optional

from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt6.QtGui import QColor, QFont, QIcon
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMainWindow,
    QProgressBar,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget
)

from src.core.ai_models import AVAILABLE_MODELS
from src.core.batch_manager import BatchManager, EpisodeStatus, QueueItem
from src.core.detector import HardwareInfo, get_hardware_info
from src.core.scene_renamer import SceneNamingConfig


DARK_STYLESHEET = """
QMainWindow, QWidget {
    background-color: #0f172a;
    color: #f8fafc;
    font-family: 'Segoe UI', -apple-system, BlinkMacSystemFont, Roboto, sans-serif;
    font-size: 13px;
}

QGroupBox {
    background-color: #1e293b;
    border: 1px solid #334155;
    border-radius: 8px;
    margin-top: 14px;
    padding: 12px 14px 14px 14px;
    font-weight: 600;
}

QGroupBox::title {
    subcontrol-origin: margin;
    subcontrol-position: top left;
    left: 14px;
    padding: 0 6px;
    color: #38bdf8;
}

QLineEdit, QComboBox {
    background-color: #0f172a;
    border: 1px solid #475569;
    border-radius: 6px;
    padding: 6px 10px;
    color: #f8fafc;
    selection-background-color: #0284c7;
}

QLineEdit:focus, QComboBox:focus {
    border: 1px solid #38bdf8;
}

QComboBox::drop-down {
    border: none;
    padding-right: 8px;
}

QComboBox QAbstractItemView {
    background-color: #1e293b;
    border: 1px solid #475569;
    selection-background-color: #0369a1;
    color: #f8fafc;
}

QPushButton {
    background-color: #334155;
    border: 1px solid #475569;
    border-radius: 6px;
    padding: 7px 14px;
    font-weight: 600;
    color: #f8fafc;
}

QPushButton:hover {
    background-color: #475569;
    border-color: #64748b;
}

QPushButton:pressed {
    background-color: #1e293b;
}

QPushButton#btnStart {
    background-color: #059669;
    border: 1px solid #10b981;
    color: #ffffff;
    font-size: 14px;
    padding: 10px 20px;
}

QPushButton#btnStart:hover {
    background-color: #10b981;
}

QPushButton#btnStop {
    background-color: #dc2626;
    border: 1px solid #ef4444;
    color: #ffffff;
    font-size: 14px;
    padding: 10px 20px;
}

QPushButton#btnStop:hover {
    background-color: #ef4444;
}

QPushButton#btnScan {
    background-color: #0284c7;
    border: 1px solid #38bdf8;
    color: #ffffff;
}

QPushButton#btnScan:hover {
    background-color: #0369a1;
}

QTableWidget {
    background-color: #1e293b;
    border: 1px solid #334155;
    border-radius: 8px;
    gridline-color: #334155;
    selection-background-color: #075985;
    selection-color: #ffffff;
}

QTableWidget QHeaderView::section {
    background-color: #0f172a;
    color: #94a3b8;
    padding: 8px;
    border: 1px solid #334155;
    font-weight: 600;
}

QProgressBar {
    background-color: #1e293b;
    border: 1px solid #334155;
    border-radius: 6px;
    text-align: center;
    color: #ffffff;
    font-weight: 600;
}

QProgressBar::chunk {
    background-color: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #0284c7, stop:1 #38bdf8);
    border-radius: 5px;
}

QCheckBox {
    color: #cbd5e1;
    spacing: 6px;
}

QCheckBox::indicator {
    width: 16px;
    height: 16px;
    border: 1px solid #475569;
    border-radius: 4px;
    background-color: #0f172a;
}

QCheckBox::indicator:checked {
    background-color: #0284c7;
    border-color: #38bdf8;
}
"""


class UpscaleWorker(QThread):
    item_updated = pyqtSignal(object)
    batch_updated = pyqtSignal(int, int, float, str)
    finished_signal = pyqtSignal()

    def __init__(
        self,
        batch_manager: BatchManager,
        output_dir: str,
        base_dir: str,
        target_preset: str,
        video_codec: str,
        model_id: str,
        scene_config: SceneNamingConfig,
        preserve_folders: bool,
        overwrite: bool
    ):
        super().__init__()
        self.bm = batch_manager
        self.output_dir = output_dir
        self.base_dir = base_dir
        self.target_preset = target_preset
        self.video_codec = video_codec
        self.model_id = model_id
        self.scene_config = scene_config
        self.preserve_folders = preserve_folders
        self.overwrite = overwrite

    def run(self):
        self.bm.run_batch(
            output_dir=self.output_dir,
            base_dir=self.base_dir,
            target_preset=self.target_preset,
            video_codec=self.video_codec,
            model_id=self.model_id,
            scene_config=self.scene_config,
            preserve_folders=self.preserve_folders,
            overwrite=self.overwrite,
            item_callback=lambda item: self.item_updated.emit(item),
            batch_callback=lambda cur, tot, pct, status: self.batch_updated.emit(cur, tot, pct, status)
        )
        self.finished_signal.emit()


class UpscalerMainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Video Enhancer 4K — Animation & Cartoon Edition")
        self.resize(1240, 780)
        self.setMinimumSize(1000, 600)
        self.setAcceptDrops(True)

        self.hardware = get_hardware_info()
        self.batch_manager = BatchManager(self.hardware)
        self.worker: Optional[UpscaleWorker] = None

        self._init_ui()
        self.setStyleSheet(DARK_STYLESHEET)

        # Auto-detect default directory
        default_dir = r"Z:\DL\Happy Tree Friends"
        if os.path.isdir(default_dir):
            self.txt_source.setText(default_dir)
            self.txt_output.setText(r"Z:\DL\Happy Tree Friends (2160p)")
            QTimer.singleShot(200, self.on_scan_clicked)

    def _init_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(16, 16, 16, 16)
        main_layout.setSpacing(12)

        # 1. Header Banner
        header = QFrame()
        header.setStyleSheet("background-color: #1e293b; border: 1px solid #334155; border-radius: 8px;")
        h_layout = QHBoxLayout(header)
        h_layout.setContentsMargins(14, 10, 14, 10)

        title_box = QVBoxLayout()
        lbl_title = QLabel("Video Enhancer 4K — Cartoon & 2D Animation")
        lbl_title.setStyleSheet("font-size: 17px; font-weight: 700; color: #38bdf8;")
        lbl_sub = QLabel("Batch processing with line reconstruction and standardized Scene / Warez naming")
        lbl_sub.setStyleSheet("font-size: 12px; color: #94a3b8;")
        title_box.addWidget(lbl_title)
        title_box.addWidget(lbl_sub)
        h_layout.addLayout(title_box)

        h_layout.addStretch()

        # Telemetry badge
        gpu_str = f"GPU: {self.hardware.gpu_name} ({self.hardware.gpu_vram_gb} GB)"
        nvenc_str = "NVENC H.265 / H.264 Active" if self.hardware.has_nvenc_hevc else "CPU Encoding"
        lbl_hw = QLabel(f"{gpu_str}  |  {nvenc_str}")
        lbl_hw.setStyleSheet("color: #10b981; font-weight: 600; font-size: 12px; padding: 4px 10px; background-color: #064e3b; border-radius: 6px;")
        h_layout.addWidget(lbl_hw)

        main_layout.addWidget(header)

        # 2. Paths & Config Group
        grp_config = QGroupBox("Processing & Naming Configuration")
        cfg_layout = QVBoxLayout(grp_config)
        cfg_layout.setSpacing(10)

        # Path row
        row_src = QHBoxLayout()
        row_src.addWidget(QLabel("Source Folder:"))
        self.txt_source = QLineEdit()
        self.txt_source.setPlaceholderText("Drag and drop video folder or click browse...")
        row_src.addWidget(self.txt_source, 1)
        btn_browse_src = QPushButton("Browse...")
        btn_browse_src.clicked.connect(self.on_browse_source)
        row_src.addWidget(btn_browse_src)

        btn_scan = QPushButton("🔍 Scan Folder")
        btn_scan.setObjectName("btnScan")
        btn_scan.clicked.connect(self.on_scan_clicked)
        row_src.addWidget(btn_scan)
        cfg_layout.addLayout(row_src)

        row_out = QHBoxLayout()
        row_out.addWidget(QLabel("Output Folder:"))
        self.txt_output = QLineEdit()
        self.txt_output.setPlaceholderText("Destination directory for 4K videos...")
        row_out.addWidget(self.txt_output, 1)
        btn_browse_out = QPushButton("Browse...")
        btn_browse_out.clicked.connect(self.on_browse_output)
        row_out.addWidget(btn_browse_out)

        btn_open_out = QPushButton("📁 Open Folder")
        btn_open_out.clicked.connect(self.on_open_output)
        row_out.addWidget(btn_open_out)
        cfg_layout.addLayout(row_out)

        # Settings row
        row_settings = QHBoxLayout()

        # Resolution
        row_settings.addWidget(QLabel("Resolution:"))
        self.cb_resolution = QComboBox()
        self.cb_resolution.addItems(["4K UHD (3840x2160)", "1440p QHD (2560x1440)", "1080p FHD (1920x1080)", "720p HD (1280x720)"])
        self.cb_resolution.currentIndexChanged.connect(self.on_settings_changed)
        row_settings.addWidget(self.cb_resolution)

        # Codec
        row_settings.addWidget(QLabel("Codec:"))
        self.cb_codec = QComboBox()
        self.cb_codec.addItems(["H.265 (HEVC NVENC)", "H.264 (NVENC)", "H.265 (CPU x265)"])
        self.cb_codec.currentIndexChanged.connect(self.on_settings_changed)
        row_settings.addWidget(self.cb_codec)

        # Model
        row_settings.addWidget(QLabel("Model:"))
        self.cb_model = QComboBox()
        for m_id, m_spec in AVAILABLE_MODELS.items():
            self.cb_model.addItem(m_spec.name, m_id)
        row_settings.addWidget(self.cb_model)

        # Scene Tags
        row_settings.addWidget(QLabel("Scene Group:"))
        self.txt_group = QLineEdit("CYLAE")
        self.txt_group.setMaximumWidth(90)
        self.txt_group.textChanged.connect(self.on_settings_changed)
        row_settings.addWidget(self.txt_group)

        row_settings.addWidget(QLabel("Source:"))
        self.cb_source = QComboBox()
        self.cb_source.addItems(["WEB-DL", "WEBRip", "UHD.WEB-DL", "REMASTER"])
        self.cb_source.currentIndexChanged.connect(self.on_settings_changed)
        row_settings.addWidget(self.cb_source)

        # Checkboxes
        self.chk_preserve_folders = QCheckBox("Preserve subfolders")
        self.chk_preserve_folders.setChecked(True)
        row_settings.addWidget(self.chk_preserve_folders)

        self.chk_overwrite = QCheckBox("Overwrite existing")
        self.chk_overwrite.setChecked(False)
        row_settings.addWidget(self.chk_overwrite)

        row_settings.addStretch()
        cfg_layout.addLayout(row_settings)

        main_layout.addWidget(grp_config)

        # 3. Queue / Episodes Table
        table_box = QVBoxLayout()
        table_header = QHBoxLayout()
        self.lbl_queue_count = QLabel("0 episodes indexed")
        self.lbl_queue_count.setStyleSheet("font-weight: 600; color: #94a3b8;")
        table_header.addWidget(self.lbl_queue_count)
        table_header.addStretch()

        table_header.addWidget(QLabel("Filter:"))
        self.txt_filter = QLineEdit()
        self.txt_filter.setPlaceholderText("Search episode...")
        self.txt_filter.setMaximumWidth(220)
        self.txt_filter.textChanged.connect(self.on_filter_changed)
        table_header.addWidget(self.txt_filter)
        table_box.addLayout(table_header)

        self.table = QTableWidget()
        self.table.setColumnCount(6)
        self.table.setHorizontalHeaderLabels([
            "#",
            "Original File",
            "Size",
            "Target Output Name (Scene Standard)",
            "Status",
            "Detail / Progress"
        ])
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Interactive)
        self.table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeMode.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(4, QHeaderView.ResizeMode.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(5, QHeaderView.ResizeMode.Interactive)
        self.table.setColumnWidth(1, 260)
        self.table.setColumnWidth(5, 230)
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.table.setAlternatingRowColors(True)

        table_box.addWidget(self.table, 1)
        main_layout.addLayout(table_box, 1)

        # 4. Progress and Controls Bottom Bar
        bottom_bar = QFrame()
        bottom_bar.setStyleSheet("background-color: #1e293b; border: 1px solid #334155; border-radius: 8px; padding: 6px;")
        b_layout = QVBoxLayout(bottom_bar)
        b_layout.setSpacing(8)

        status_row = QHBoxLayout()
        self.lbl_overall_status = QLabel("Ready to start.")
        self.lbl_overall_status.setStyleSheet("font-weight: 600; color: #f8fafc;")
        status_row.addWidget(self.lbl_overall_status)
        status_row.addStretch()

        self.lbl_eta = QLabel("")
        self.lbl_eta.setStyleSheet("color: #38bdf8; font-weight: 600;")
        status_row.addWidget(self.lbl_eta)
        b_layout.addLayout(status_row)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setFixedHeight(22)
        b_layout.addWidget(self.progress_bar)

        btn_row = QHBoxLayout()
        self.btn_start = QPushButton("▶  START BATCH PROCESSING")
        self.btn_start.setObjectName("btnStart")
        self.btn_start.clicked.connect(self.on_start_clicked)
        btn_row.addWidget(self.btn_start, 2)

        self.btn_stop = QPushButton("⏹  STOP")
        self.btn_stop.setObjectName("btnStop")
        self.btn_stop.setEnabled(False)
        self.btn_stop.clicked.connect(self.on_stop_clicked)
        btn_row.addWidget(self.btn_stop, 1)

        b_layout.addLayout(btn_row)
        main_layout.addWidget(bottom_bar)

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event):
        urls = event.mimeData().urls()
        if urls:
            path = urls[0].toLocalFile()
            if os.path.exists(path):
                if os.path.isfile(path):
                    self.txt_source.setText(path)
                    out_dir = os.path.join(os.path.dirname(path), "(2160p)")
                else:
                    self.txt_source.setText(path)
                    out_dir = path + " (2160p)"
                self.txt_output.setText(out_dir)
                self.on_scan_clicked()

    def on_browse_source(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Source Directory", self.txt_source.text() or "Z:\\")
        if folder:
            self.txt_source.setText(folder)
            self.txt_output.setText(folder + " (2160p)")
            self.on_scan_clicked()

    def on_browse_output(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Output Directory", self.txt_output.text() or "Z:\\")
        if folder:
            self.txt_output.setText(folder)
            self.on_settings_changed()

    def on_open_output(self):
        out_dir = self.txt_output.text().strip()
        if os.path.exists(out_dir):
            os.startfile(out_dir)
        else:
            os.makedirs(out_dir, exist_ok=True)
            os.startfile(out_dir)

    def get_current_scene_config(self) -> SceneNamingConfig:
        res_idx = self.cb_resolution.currentIndex()
        res_tag = "2160p" if res_idx == 0 else ("1440p" if res_idx == 1 else ("1080p" if res_idx == 2 else "720p"))

        codec_idx = self.cb_codec.currentIndex()
        codec_tag = "H.265" if codec_idx in (0, 2) else "H.264"

        group = self.txt_group.text().strip() or "CYLAE"
        source = self.cb_source.currentText().strip() or "WEB-DL"

        return SceneNamingConfig(
            show_name="Happy.Tree.Friends",
            resolution_tag=res_tag,
            source_tag=source,
            audio_tag="AAC2.0",
            codec_tag=codec_tag,
            group_tag=group,
            container=".mp4"
        )

    def on_scan_clicked(self):
        src = self.txt_source.text().strip()
        out = self.txt_output.text().strip()
        if not src or not os.path.exists(src):
            self.lbl_overall_status.setText("Error: Source directory does not exist.")
            return

        if not out:
            out = src + " (2160p)"
            self.txt_output.setText(out)

        cfg = self.get_current_scene_config()
        preserve_folders = self.chk_preserve_folders.isChecked()

        self.lbl_overall_status.setText("Indexing files...")
        queue = self.batch_manager.scan_folder(src, out, cfg, preserve_folders=preserve_folders)
        self._populate_table(queue)

    def on_settings_changed(self):
        src = self.txt_source.text().strip()
        out = self.txt_output.text().strip()
        if src and os.path.exists(src):
            cfg = self.get_current_scene_config()
            preserve_folders = self.chk_preserve_folders.isChecked()
            queue = self.batch_manager.scan_folder(src, out, cfg, preserve_folders=preserve_folders)
            self._populate_table(queue)

    def _populate_table(self, queue: List[QueueItem]):
        self.table.setRowCount(len(queue))
        skipped_count = 0

        for r, item in enumerate(queue):
            # 0. Index
            it_idx = QTableWidgetItem(str(item.index))
            it_idx.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self.table.setItem(r, 0, it_idx)

            # 1. Source relative path
            it_src = QTableWidgetItem(item.relative_source_path)
            self.table.setItem(r, 1, it_src)

            # 2. Source Size
            it_sz = QTableWidgetItem(f"{item.file_size_mb:.1f} MB")
            it_sz.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            self.table.setItem(r, 2, it_sz)

            # 3. Target Scene Name
            it_scene = QTableWidgetItem(item.target_scene_name)
            it_scene.setForeground(QColor("#38bdf8"))
            self.table.setItem(r, 3, it_scene)

            # 4. Status
            it_status = QTableWidgetItem(item.status.value)
            it_status.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self._style_status_item(it_status, item.status)
            self.table.setItem(r, 4, it_status)

            if item.status == EpisodeStatus.SKIPPED:
                skipped_count += 1

            # 5. Detail / Progress
            it_detail = QTableWidgetItem(item.status_detail)
            self.table.setItem(r, 5, it_detail)

        total = len(queue)
        ready = total - skipped_count
        self.lbl_queue_count.setText(f"{total} episodes indexed ({ready} ready to process, {skipped_count} already present)")
        self.lbl_overall_status.setText(f"Ready. {total} episodes loaded.")

    def _style_status_item(self, item: QTableWidgetItem, status: EpisodeStatus):
        if status == EpisodeStatus.COMPLETED:
            item.setForeground(QColor("#10b981"))  # emerald
        elif status == EpisodeStatus.PROCESSING:
            item.setForeground(QColor("#f59e0b"))  # amber
        elif status == EpisodeStatus.SKIPPED:
            item.setForeground(QColor("#64748b"))  # slate gray
        elif status == EpisodeStatus.FAILED:
            item.setForeground(QColor("#ef4444"))  # red
        elif status == EpisodeStatus.CANCELLED:
            item.setForeground(QColor("#f97316"))  # orange
        else:
            item.setForeground(QColor("#cbd5e1"))  # light gray

    def on_filter_changed(self, text: str):
        filter_text = text.lower().strip()
        for r in range(self.table.rowCount()):
            match = False
            for col in (1, 3):
                item = self.table.item(r, col)
                if item and filter_text in item.text().lower():
                    match = True
                    break
            self.table.setRowHidden(r, not match)

    def on_start_clicked(self):
        if not self.batch_manager.queue:
            self.lbl_overall_status.setText("No episodes in queue.")
            return

        out_dir = self.txt_output.text().strip()
        base_dir = self.txt_source.text().strip()
        if not out_dir:
            self.lbl_overall_status.setText("Please specify an output folder.")
            return

        res_idx = self.cb_resolution.currentIndex()
        target_preset = "4k" if res_idx == 0 else ("1440p" if res_idx == 1 else ("1080p" if res_idx == 2 else "720p"))

        codec_idx = self.cb_codec.currentIndex()
        video_codec = "hevc_nvenc" if codec_idx == 0 else ("h264_nvenc" if codec_idx == 1 else "libx265")

        model_id = self.cb_model.currentData() or "auto"
        scene_cfg = self.get_current_scene_config()
        preserve_folders = self.chk_preserve_folders.isChecked()
        overwrite = self.chk_overwrite.isChecked()

        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.lbl_overall_status.setText("Starting batch processing...")

        self.worker = UpscaleWorker(
            batch_manager=self.batch_manager,
            output_dir=out_dir,
            base_dir=base_dir,
            target_preset=target_preset,
            video_codec=video_codec,
            model_id=model_id,
            scene_config=scene_cfg,
            preserve_folders=preserve_folders,
            overwrite=overwrite
        )
        self.worker.item_updated.connect(self.on_item_updated)
        self.worker.batch_updated.connect(self.on_batch_updated)
        self.worker.finished_signal.connect(self.on_finished)
        self.worker.start()

    def on_stop_clicked(self):
        self.lbl_overall_status.setText("Stopping...")
        self.batch_manager.stop()
        self.btn_stop.setEnabled(False)

    def on_item_updated(self, item: QueueItem):
        row = item.index - 1
        if 0 <= row < self.table.rowCount():
            it_status = self.table.item(row, 4)
            if it_status:
                it_status.setText(item.status.value)
                self._style_status_item(it_status, item.status)

            it_detail = self.table.item(row, 5)
            if it_detail:
                it_detail.setText(item.status_detail)

            if item.status == EpisodeStatus.PROCESSING:
                self.table.scrollToItem(it_status)

    def on_batch_updated(self, current: int, total: int, pct: float, status_str: str):
        self.progress_bar.setValue(int(pct))
        self.lbl_overall_status.setText(status_str)

    def on_finished(self):
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.lbl_overall_status.setText("Batch processing completed or stopped.")
        self.progress_bar.setValue(100)


def run_gui():
    app = QApplication(sys.argv)
    window = UpscalerMainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(run_gui())
