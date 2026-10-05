"""Dark design system — deep gradient theme with glows, pills, and animated accents.

Optimized for the sidebar + queue full-width layout.
"""

from __future__ import annotations

import os

_ASSETS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")
_CHEVRON_DOWN_SVG = os.path.join(_ASSETS_DIR, "chevron_down.svg").replace("\\", "/")
_CHEVRON_CSS = f'image: url("{_CHEVRON_DOWN_SVG}");\n    ' if os.path.isfile(_CHEVRON_DOWN_SVG) else ""

# Color tokens
COLOR_BG = "#0d0d10"
COLOR_SURFACE = "#13131a"
COLOR_CARD = "#16161e"
COLOR_CARD_ELEVATED = "#1c1c26"
COLOR_SIDEBAR = "#111118"
COLOR_BLUE = "#4f8fff"
COLOR_BLUE_BRIGHT = "#60a5fa"
COLOR_PURPLE = "#a78bfa"
COLOR_GREEN = "#34d399"
COLOR_INDIGO = "#818cf8"
COLOR_RED = "#f87171"
COLOR_ORANGE = "#fb923c"
COLOR_AMBER = "#fbbf24"
COLOR_MUTED = "#6b7280"
COLOR_BORDER = "#252535"
COLOR_BORDER_LIGHT = "#2e2e42"

# Platform compatibility tokens (Apple HIG / Fluent reference tokens)
COLOR_SYS_BG = "#1e1e1e"
COLOR_SYS_CARD = "#252528"
COLOR_APPLE_BLUE = "#0a84ff"
COLOR_APPLE_GREEN = "#30d158"
COLOR_APPLE_INDIGO = "#5e5ce6"
COLOR_APPLE_RED = "#ff453a"

DARK_STYLESHEET = """
/* ==========================================================================
   Video Enhancer 4K — Deep Dark Design System v2.1
   Sidebar layout · Gradient accents · Glow effects · 1080p optimized
   Compatible tokens: #1e1e1e, #252528, #0a84ff, #30d158, #5e5ce6, #ff453a
   ========================================================================== */

/* ---------- Base reset ---------- */
QMainWindow, QWidget {
    background-color: #0d0d10;
    color: #e2e8f0;
    font-family: "Segoe UI Variable", "Segoe UI", system-ui, -apple-system, sans-serif;
    font-size: 13px;
    selection-background-color: #4f8fff;
    selection-color: #ffffff;
}

/* ---------- Splitter divider ---------- */
QSplitter::handle:horizontal {
    background-color: #1c1c2a;
    width: 4px;
    border-radius: 2px;
}

QSplitter::handle:horizontal:hover {
    background-color: #4f8fff;
}

/* ---------- Sidebar surface ---------- */
QWidget#sidebar {
    background-color: #0f0f18;
    border-right: 1px solid #1a1a28;
}

/* ---------- Card panels ---------- */
QGroupBox, QFrame#card {
    background-color: #16161e;
    border: 1px solid #1e1e2e;
    border-radius: 12px;
    margin-top: 10px;
    padding-top: 14px;
    font-weight: 600;
}

QGroupBox#sideCard {
    background-color: #13131a;
    border: 1px solid #1c1c2a;
    border-radius: 10px;
    margin-top: 8px;
    padding: 12px 10px 10px 10px;
}

QGroupBox::title {
    subcontrol-origin: margin;
    subcontrol-position: top left;
    left: 12px;
    top: 1px;
    padding: 0 6px;
    color: #60a5fa;
    font-weight: 700;
    font-size: 11px;
    letter-spacing: 0.6px;
    text-transform: uppercase;
}

/* Header panel (top full-width bar) */
QFrame#headerPanel {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
        stop:0 #0f0f1a, stop:0.5 #121220, stop:1 #0f0f1a);
    border-bottom: 1px solid #1e1e2e;
    border-radius: 0px;
}

/* Bottom bar */
QFrame#bottomBar {
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
        stop:0 #13131e, stop:1 #0f0f18);
    border-top: 1px solid #1e1e2e;
    border-radius: 0px;
}

/* macPanel alias (legacy compat) */
QFrame#macPanel {
    background-color: #13131e;
    border: 1px solid #1e1e2e;
    border-radius: 12px;
}

/* ---------- Text inputs & combos ---------- */
QLineEdit, QComboBox, QSpinBox {
    background-color: #0f0f18;
    border: 1px solid #252535;
    border-radius: 8px;
    padding: 7px 10px;
    color: #e2e8f0;
    selection-background-color: #4f8fff;
    min-height: 16px;
}

QLineEdit:focus, QComboBox:focus, QSpinBox:focus {
    border: 1px solid #4f8fff;
    background-color: #111120;
    outline: none;
}

QLineEdit:hover, QComboBox:hover, QSpinBox:hover {
    border-color: #2e2e42;
}

QLineEdit::clear-button {
    image: none;
}

QComboBox::drop-down {
    border: none;
    padding-right: 8px;
    width: 20px;
}

QComboBox::down-arrow {
    width: 10px;
    height: 10px;
}

QComboBox QAbstractItemView {
    background-color: #1a1a26;
    border: 1px solid #252535;
    border-radius: 8px;
    selection-background-color: #2d3a5e;
    selection-color: #60a5fa;
    padding: 4px;
    outline: none;
}

QComboBox QAbstractItemView::item {
    padding: 6px 10px;
    border-radius: 4px;
    min-height: 24px;
}

QComboBox QAbstractItemView::item:hover {
    background-color: #252535;
}

/* ---------- Default buttons ---------- */
QPushButton {
    background-color: #1c1c2a;
    border: 1px solid #252535;
    border-radius: 8px;
    padding: 7px 14px;
    color: #c8d0e0;
    font-weight: 500;
    min-height: 18px;
}

QPushButton:hover {
    background-color: #222235;
    border-color: #333348;
    color: #e2e8f0;
}

QPushButton:pressed {
    background-color: #13131e;
    border-color: #1e1e2e;
}

QPushButton:disabled {
    background-color: #13131e;
    border-color: #1a1a28;
    color: #3a3a50;
}

/* ---------- Primary action: START ---------- */
QPushButton#btnStart {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
        stop:0 #2563eb, stop:0.5 #4f46e5, stop:1 #7c3aed);
    border: 1px solid #3b4fcc;
    color: #ffffff;
    font-weight: 800;
    font-size: 13px;
    border-radius: 10px;
    padding: 10px 24px;
    letter-spacing: 0.5px;
}

QPushButton#btnStart:hover {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
        stop:0 #3070ff, stop:0.5 #6058f0, stop:1 #8b47ff);
}

QPushButton#btnStart:pressed {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
        stop:0 #1d4ed8, stop:0.5 #4338ca, stop:1 #6d28d9);
}

QPushButton#btnStart:disabled {
    background: #1a1a2a;
    border-color: #252535;
    color: #3a3a50;
}

/* ---------- Stop button ---------- */
QPushButton#btnStop {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
        stop:0 #dc2626, stop:1 #b91c1c);
    border: 1px solid #991b1b;
    color: #ffffff;
    font-weight: 700;
    border-radius: 10px;
    padding: 10px 18px;
}

QPushButton#btnStop:hover {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
        stop:0 #ef4444, stop:1 #dc2626);
}

QPushButton#btnStop:pressed {
    background-color: #991b1b;
}

/* ---------- Preview / secondary action buttons ---------- */
QPushButton#btnPreview, QPushButton#btnPreviewClip, QPushButton#btnCompareFrame {
    background-color: #1a1a28;
    border: 1px solid #28283c;
    color: #e2e8f0;
    font-weight: 600;
    border-radius: 8px;
    padding: 7px 14px;
}

QPushButton#btnPreview:hover, QPushButton#btnPreviewClip:hover, QPushButton#btnCompareFrame:hover {
    background-color: #232338;
    border-color: #6366f1;
    color: #ffffff;
}

/* Preflight button */
QPushButton#btnPreflight {
    background-color: #141c28;
    border: 1px solid #202e42;
    color: #93c5fd;
    font-weight: 600;
    border-radius: 8px;
    padding: 7px 14px;
}

QPushButton#btnPreflight:hover {
    background-color: #1a2638;
    border-color: #3b82f6;
    color: #bfdbfe;
}

/* Retry button */
QPushButton#btnRetryFailed {
    background-color: #1a1f10;
    border: 1px solid #3a4a1a;
    color: #86efac;
    font-weight: 600;
    border-radius: 8px;
    padding: 7px 14px;
}

QPushButton#btnRetryFailed:hover {
    background-color: #202810;
    border-color: #4a6020;
    color: #bbf7d0;
}

/* Scan button */
QPushButton#btnScan {
    background-color: #1a1e2e;
    border: 1px solid #2d3555;
    color: #818cf8;
    font-weight: 600;
    border-radius: 8px;
    padding: 7px 14px;
}

QPushButton#btnScan:hover {
    background-color: #1e2238;
    border-color: #3d4a70;
    color: #a5b4fc;
}

QPushButton#btnScan:disabled {
    background-color: #13131e;
    border-color: #1a1a28;
    color: #3a3a50;
}

/* ---------- Expert panel toggle ---------- */
QPushButton#btnToggleExpert {
    background-color: #161622;
    border: 1px solid #252535;
    border-radius: 8px;
    padding: 5px 12px;
    color: #818cf8;
    font-weight: 600;
    font-size: 12px;
}

QPushButton#btnToggleExpert:hover {
    background-color: #1e1e30;
    border-color: #4f4fa0;
    color: #a5b4fc;
}

/* ---------- Stat filter pills (segmented control) ---------- */
QPushButton[class="statPill"] {
    background-color: #13131e;
    border: 1px solid #1e1e2e;
    border-radius: 20px;
    padding: 4px 14px;
    color: #6b7280;
    font-weight: 600;
    font-size: 12px;
    min-height: 16px;
}

QPushButton[class="statPill"]:hover {
    background-color: #1c1c2a;
    border-color: #252535;
    color: #9ca3af;
}

QPushButton[class="statPill"]:checked {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
        stop:0 #1d2d5e, stop:1 #1e1a4a);
    border: 1px solid #3b4fcc;
    color: #93c5fd;
}

QPushButton[class="statPillError"] {
    background-color: #1f1010;
    border: 1px solid #7f1d1d;
    border-radius: 20px;
    padding: 4px 14px;
    color: #f87171;
    font-weight: 600;
    font-size: 12px;
    min-height: 16px;
}

QPushButton[class="statPillError"]:checked {
    background-color: #2d1010;
    border-color: #ef4444;
}

/* ---------- Queue table ---------- */
QTableWidget {
    background-color: #0f0f18;
    alternate-background-color: #121220;
    border: 1px solid #1a1a28;
    border-radius: 10px;
    gridline-color: transparent;
    selection-background-color: #1d2d5e;
    selection-color: #bfdbfe;
    outline: none;
}

QTableWidget::item {
    padding: 7px 12px;
    border-bottom: 1px solid #141422;
}

QTableWidget::item:hover {
    background-color: rgba(79, 143, 255, 0.08);
}

QTableWidget::item:selected {
    background-color: #1d2d5e;
    color: #bfdbfe;
}

QHeaderView::section {
    background-color: #13131e;
    color: #64748b;
    padding: 8px 10px;
    border: none;
    border-bottom: 1px solid #1e1e2e;
    border-right: 1px solid #1a1a28;
    font-weight: 700;
    font-size: 11px;
    letter-spacing: 0.6px;
    text-transform: uppercase;
}

QHeaderView::section:first {
    border-radius: 0px;
}

QHeaderView {
    background-color: transparent;
}

/* ---------- Progress bar ---------- */
QProgressBar {
    background-color: #13131e;
    border: 1px solid #1e1e2e;
    border-radius: 6px;
    text-align: center;
    color: #60a5fa;
    font-weight: 700;
    font-size: 11px;
    min-height: 12px;
    max-height: 12px;
}

QProgressBar::chunk {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
        stop:0 #2563eb,
        stop:0.3 #4f46e5,
        stop:0.6 #7c3aed,
        stop:0.85 #a855f7,
        stop:1 #ec4899);
    border-radius: 6px;
}

/* ---------- Checkboxes ---------- */
QCheckBox {
    color: #9ca3af;
    spacing: 8px;
    font-size: 12px;
}

QCheckBox::indicator {
    width: 16px;
    height: 16px;
    border: 1px solid #2e2e42;
    border-radius: 5px;
    background-color: #13131e;
}

QCheckBox::indicator:hover {
    border-color: #4f8fff;
}

QCheckBox::indicator:checked {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
        stop:0 #3b82f6, stop:1 #6366f1);
    border-color: #4f8fff;
}

/* ---------- Menus ---------- */
QMenu {
    background-color: #1a1a26;
    border: 1px solid #252535;
    border-radius: 10px;
    padding: 6px;
}

QMenu::item {
    padding: 7px 20px 7px 14px;
    border-radius: 6px;
    color: #d1d5db;
    font-size: 12px;
}

QMenu::item:selected {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
        stop:0 #1d2d5e, stop:1 #1e1a4a);
    color: #93c5fd;
}

QMenu::separator {
    height: 1px;
    background: #252535;
    margin: 4px 10px;
}

QMessageBox {
    background-color: #1a1a26;
    border: 1px solid #252535;
}

/* ---------- Scrollbars ---------- */
QScrollBar:vertical {
    border: none;
    background: rgba(255, 255, 255, 0.03);
    width: 8px;
    margin: 2px;
    border-radius: 4px;
}

QScrollBar::handle:vertical {
    background: #33334d;
    min-height: 28px;
    border-radius: 4px;
}

QScrollBar::handle:vertical:hover {
    background: #4f8fff;
}

QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
    height: 0px;
}

QScrollBar:horizontal {
    border: none;
    background: rgba(255, 255, 255, 0.03);
    height: 8px;
    margin: 2px;
    border-radius: 4px;
}

QScrollBar::handle:horizontal {
    background: #33334d;
    min-width: 28px;
    border-radius: 4px;
}

QScrollBar::handle:horizontal:hover {
    background: #4f8fff;
}

QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {
    width: 0px;
}

/* ---------- Tooltips ---------- */
QToolTip {
    background-color: #1a1a26;
    color: #e2e8f0;
    border: 1px solid #333348;
    border-radius: 8px;
    padding: 6px 10px;
    font-size: 12px;
}

/* ---------- Sliders ---------- */
QSlider::groove:horizontal {
    height: 4px;
    background: #1e1e2e;
    border-radius: 2px;
}

QSlider::sub-page:horizontal {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
        stop:0 #3b82f6, stop:1 #6366f1);
    border-radius: 2px;
}

QSlider::handle:horizontal {
    background: #ffffff;
    width: 14px;
    height: 14px;
    margin-top: -5px;
    margin-bottom: -5px;
    border-radius: 7px;
    border: 2px solid #4f8fff;
}

QSlider::handle:horizontal:hover {
    background: #eff6ff;
    border-color: #93c5fd;
}

/* ---------- Terminal console ---------- */
QTextEdit#txtConsole {
    background-color: #080810;
    border: 1px solid #1a1a28;
    border-radius: 8px;
    font-family: "Cascadia Mono", "Cascadia Code", "Consolas", "SF Mono", monospace;
    font-size: 11px;
    color: #60a5fa;
    padding: 8px;
}

/* ---------- SpinBox styling ---------- */
QSpinBox {
    background-color: #0f0f18;
    border: 1px solid #252535;
    border-radius: 8px;
    padding: 6px 22px 6px 8px;
    color: #e2e8f0;
    min-height: 16px;
}

QSpinBox::up-button {
    subcontrol-origin: border;
    subcontrol-position: top right;
    width: 18px;
    border-left: 1px solid #252535;
    border-bottom: 1px solid #1a1a28;
    background-color: #161624;
    border-top-right-radius: 7px;
}

QSpinBox::down-button {
    subcontrol-origin: border;
    subcontrol-position: bottom right;
    width: 18px;
    border-left: 1px solid #252535;
    background-color: #161624;
    border-bottom-right-radius: 7px;
}

QSpinBox::up-button:hover, QSpinBox::down-button:hover {
    background-color: #252538;
}

/* ---------- Labels (utility classes via setProperty) ---------- */
QLabel#sectionHeader {
    font-weight: 700;
    font-size: 10px;
    color: #4f5566;
    letter-spacing: 1px;
    text-transform: uppercase;
}
"""

if _CHEVRON_CSS:
    DARK_STYLESHEET = DARK_STYLESHEET.replace(
        "QComboBox::down-arrow {\n    width: 10px;\n    height: 10px;\n}",
        f"QComboBox::down-arrow {{\n    {_CHEVRON_CSS}width: 10px;\n    height: 10px;\n}}",
    )
