"""macOS Human Interface Guidelines (HIG) Dark Mode design system and styles.

Provides refined dark mode colors, card panels, sliders, tables, and progress bars.
"""

from __future__ import annotations

# Color tokens
COLOR_BG = "#1e1e1e"
COLOR_CARD = "#252528"
COLOR_BLUE = "#0a84ff"
COLOR_GREEN = "#30d158"
COLOR_INDIGO = "#5e5ce6"
COLOR_RED = "#ff453a"
COLOR_ORANGE = "#ff9f0a"
COLOR_PURPLE = "#bf5af2"
COLOR_MUTED = "#8e8e93"
COLOR_BORDER = "#3a3a3c"

DARK_STYLESHEET = """
/* ==========================================================================
   macOS Human Interface Guidelines (HIG) Dark Mode Design System
   ========================================================================== */

QMainWindow, QWidget {
    background-color: #1e1e1e;
    color: #f5f5f7;
    font-family: -apple-system, BlinkMacSystemFont, "SF Pro Text", "Segoe UI", system-ui, sans-serif;
    font-size: 13px;
    selection-background-color: #0a84ff;
    selection-color: #ffffff;
}

/* Elevated card surfaces (macOS Finder / System Settings panel style) */
QGroupBox, QFrame#macPanel {
    background-color: #252528;
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 12px;
    margin-top: 10px;
    padding: 14px;
    font-weight: 600;
}

QGroupBox::title {
    subcontrol-origin: margin;
    subcontrol-position: top left;
    left: 14px;
    padding: 0 6px;
    color: #0a84ff;
    font-weight: 700;
}

/* Controls & Inputs (macOS dark textfields & select popups) */
QLineEdit, QComboBox, QSpinBox {
    background-color: #1c1c1e;
    border: 1px solid #3a3a3c;
    border-radius: 8px;
    padding: 7px 10px;
    color: #f5f5f7;
    selection-background-color: #0a84ff;
}

QLineEdit:focus, QComboBox:focus, QSpinBox:focus {
    border: 1px solid #0a84ff;
    background-color: #242426;
}

QComboBox::drop-down {
    border: none;
    padding-right: 8px;
}

QComboBox QAbstractItemView {
    background-color: #252528;
    border: 1px solid #3a3a3c;
    border-radius: 8px;
    selection-background-color: #0a84ff;
    selection-color: #ffffff;
    padding: 4px;
}

/* Buttons (Apple rounded buttons with subtle elevation) */
QPushButton {
    background-color: #2c2c2e;
    border: 1px solid #3a3a3c;
    border-radius: 8px;
    padding: 7px 16px;
    color: #f5f5f7;
    font-weight: 500;
}

QPushButton:hover {
    background-color: #3a3a3c;
    border-color: #48484a;
}

QPushButton:pressed {
    background-color: #1c1c1e;
}

QPushButton:disabled {
    background-color: #1a1a1c;
    border-color: #2c2c2e;
    color: #636366;
}

/* Primary Action Buttons (macOS Accent Blue Pill) */
QPushButton#btnStart {
    background-color: #0a84ff;
    border: 1px solid #0071e3;
    color: #ffffff;
    font-weight: 700;
    font-size: 13px;
    border-radius: 8px;
    padding: 8px 20px;
}

QPushButton#btnStart:hover {
    background-color: #0077ed;
}

QPushButton#btnStart:pressed {
    background-color: #0062c4;
}

/* Preview buttons */
QPushButton#btnPreview, QPushButton#btnPreviewClip, QPushButton#btnCompareFrame {
    background-color: #5e5ce6;
    border: 1px solid #4d4bc8;
    color: #ffffff;
    font-weight: 600;
    border-radius: 8px;
    padding: 7px 16px;
}

QPushButton#btnPreview:hover, QPushButton#btnPreviewClip:hover, QPushButton#btnCompareFrame:hover {
    background-color: #6e6cf0;
}

/* Destructive / Stop Action Button */
QPushButton#btnStop {
    background-color: #ff453a;
    border: 1px solid #d73a30;
    color: #ffffff;
    font-weight: 700;
    border-radius: 8px;
    padding: 8px 20px;
}

QPushButton#btnStop:hover {
    background-color: #e0382e;
}

QPushButton#btnStop:pressed {
    background-color: #b82820;
}

/* Data Table (macOS List / TableView) */
QTableWidget {
    background-color: #1c1c1e;
    alternate-background-color: #222225;
    border: 1px solid #2c2c2e;
    border-radius: 10px;
    gridline-color: transparent;
    selection-background-color: #0a84ff;
    selection-color: #ffffff;
}

QHeaderView::section {
    background-color: #252528;
    color: #8e8e93;
    padding: 8px;
    border: none;
    border-bottom: 1px solid #3a3a3c;
    font-weight: 600;
    font-size: 11px;
    text-transform: uppercase;
}

/* Progress Bar (macOS System Accent Linear Tracker) */
QProgressBar {
    background-color: #2c2c2e;
    border: none;
    border-radius: 5px;
    text-align: center;
    color: #ffffff;
    font-weight: 600;
    min-height: 10px;
}

QProgressBar::chunk {
    background-color: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #0a84ff, stop:1 #30d158);
    border-radius: 5px;
}

/* Checkbox (macOS Toggle indicator) */
QCheckBox {
    color: #aeaeb2;
    spacing: 8px;
}

QCheckBox::indicator {
    width: 16px;
    height: 16px;
    border: 1px solid #48484a;
    border-radius: 4px;
    background-color: #2c2c2e;
}

QCheckBox::indicator:checked {
    background-color: #0a84ff;
    border-color: #0a84ff;
}

/* Menus & Dialog Alerts (macOS Translucent Dark styling) */
QMenu, QMessageBox {
    background-color: #2c2c2e;
    border: 1px solid #3a3a3c;
    border-radius: 10px;
    padding: 6px;
}

QMenu::item {
    padding: 6px 20px 6px 12px;
    border-radius: 6px;
    color: #f5f5f7;
}

QMenu::item:selected {
    background-color: #0a84ff;
    color: #ffffff;
}

/* Scrollbars (macOS Overlay Scrollbar feel) */
QScrollBar:vertical {
    border: none;
    background: transparent;
    width: 8px;
    margin: 0px;
    border-radius: 4px;
}

QScrollBar::handle:vertical {
    background: #48484a;
    min-height: 24px;
    border-radius: 4px;
}

QScrollBar::handle:vertical:hover {
    background: #636366;
}

QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
    height: 0px;
}

QScrollBar:horizontal {
    border: none;
    background: transparent;
    height: 8px;
    margin: 0px;
    border-radius: 4px;
}

QScrollBar::handle:horizontal {
    background: #48484a;
    min-width: 24px;
    border-radius: 4px;
}

QScrollBar::handle:horizontal:hover {
    background: #636366;
}

QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {
    width: 0px;
}

/* Tooltips (macOS Cocoa HUD tooltips) */
QToolTip {
    background-color: #2c2c2e;
    color: #f5f5f7;
    border: 1px solid #3a3a3c;
    border-radius: 6px;
    padding: 6px 10px;
    font-size: 12px;
}

/* Sliders (macOS slider control) */
QSlider::groove:horizontal {
    height: 4px;
    background: #3a3a3c;
    border-radius: 2px;
}

QSlider::sub-page:horizontal {
    background: #0a84ff;
    border-radius: 2px;
}

QSlider::handle:horizontal {
    background: #ffffff;
    width: 16px;
    margin-top: -6px;
    margin-bottom: -6px;
    border-radius: 8px;
    border: 1px solid #48484a;
}

QSlider::handle:horizontal:hover {
    background: #f5f5f7;
    border-color: #0a84ff;
}

/* Expert Panel Toggle */
QPushButton#btnToggleExpert {
    background-color: #2c2c2e;
    border: 1px solid #3a3a3c;
    border-radius: 8px;
    padding: 5px 12px;
    color: #0a84ff;
    font-weight: 600;
    font-size: 12px;
}

QPushButton#btnToggleExpert:hover {
    background-color: #3a3a3c;
    border-color: #0a84ff;
}

/* Terminal Console (macOS Terminal dark style) */
QTextEdit#txtConsole {
    background-color: #141416;
    border: 1px solid #2c2c2e;
    border-radius: 8px;
    font-family: 'SF Mono', 'Cascadia Mono', 'Consolas', monospace;
    font-size: 11px;
    color: #0a84ff;
    padding: 8px;
}
"""
