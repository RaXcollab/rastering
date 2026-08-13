"""Dark instrument-console theme (2026-08-12 redesign).

One QSS stylesheet from six palette tokens + Fusion dark QPalette.
Approved visual spec: docs/superpowers/specs/2026-08-12-rastering-ui-redesign-design.md
(parent repo). Qt QSS has no text-transform/letter-spacing -- group titles
are written uppercase-free in the .ui and styled by color/weight only.
"""
from __future__ import annotations

PALETTE = {
    "graphite": "#161A20",   # window ground
    "panel":    "#1E242C",   # group boxes, dock
    "recess":   "#12151A",   # camera well, input wells, status bar
    "line":     "#313A44",
    "ink":      "#D9E0E7",
    "muted":    "#8794A1",
    "cyan":     "#3EB4C8",   # interactive emphasis + armed state ONLY
    "cyan_dim": "#2A7D8C",
    "amber":    "#E2A83D",   # annunciator warn
    "green":    "#52BE6E",   # annunciator good
    "red":      "#E15A4D",   # annunciator alert (REC)
}


def build_qss(mono: str) -> str:
    p = PALETTE
    return f"""
QMainWindow, QDialog {{ background: {p['graphite']}; }}
QWidget {{ color: {p['ink']}; }}
QGroupBox {{
    background: {p['panel']};
    border: 1px solid {p['line']};
    border-radius: 3px;
    margin-top: 9px;
    padding-top: 6px;
}}
QGroupBox::title {{
    subcontrol-origin: margin;
    left: 8px;
    color: {p['muted']};
    font-weight: 600;
    font-size: 11px;
}}
QTabWidget::pane {{ border: 1px solid {p['line']}; }}
QTabBar::tab {{
    background: {p['recess']};
    color: {p['muted']};
    padding: 5px 16px;
    border: 1px solid {p['line']};
    border-bottom: none;
}}
QTabBar::tab:selected {{
    background: {p['graphite']};
    color: {p['ink']};
    border-bottom: 2px solid {p['cyan']};
}}
QPushButton {{
    background: {p['panel']};
    border: 1px solid {p['line']};
    border-radius: 3px;
    padding: 4px 10px;
}}
QPushButton:hover {{ border-color: {p['muted']}; }}
QPushButton:pressed {{ background: {p['recess']}; }}
QPushButton:disabled {{ color: {p['muted']}; border-color: {p['line']}; }}
QPushButton[primary="true"] {{
    color: {p['cyan']};
    border-color: {p['cyan_dim']};
    font-weight: 600;
}}
QDoubleSpinBox, QSpinBox, QLineEdit, QComboBox {{
    background: {p['recess']};
    border: 1px solid {p['line']};
    border-radius: 3px;
    padding: 2px 5px;
    font-family: "{mono}";
}}
QTextEdit, QPlainTextEdit {{
    background: {p['recess']};
    border: 1px solid {p['line']};
    font-family: "{mono}";
    font-size: 11px;
}}
QStatusBar {{
    background: {p['recess']};
    border-top: 1px solid {p['line']};
}}
QStatusBar QLabel {{
    font-family: "{mono}";
    font-size: 11px;
    font-weight: 600;
    border: 1px solid {p['line']};
    border-radius: 3px;
    padding: 2px 8px;
    margin: 1px 2px;
}}
QLabel[chipState="idle"]  {{ color: {p['muted']}; }}
QLabel[chipState="good"]  {{ color: {p['green']}; border-color: {p['green']}; }}
QLabel[chipState="cyan"]  {{ color: {p['cyan']};  border-color: {p['cyan_dim']}; }}
QLabel[chipState="warn"]  {{ color: {p['amber']}; border-color: {p['amber']}; }}
QLabel[chipState="alert"] {{ color: {p['red']};   border-color: {p['red']}; }}
QMenuBar {{ background: {p['graphite']}; }}
QMenuBar::item:selected {{ background: {p['panel']}; }}
QMenu {{ background: {p['panel']}; border: 1px solid {p['line']}; }}
QMenu::item:selected {{ background: {p['recess']}; color: {p['cyan']}; }}
QProgressBar {{ background: {p['recess']}; border: 1px solid {p['line']}; }}
QDockWidget {{ background: {p['panel']}; }}
"""


def apply_theme(app) -> None:
    from PyQt5 import QtGui
    from PyQt5.QtGui import QFontDatabase

    app.setStyle("Fusion")
    p = PALETTE
    pal = QtGui.QPalette()
    c = QtGui.QColor
    pal.setColor(QtGui.QPalette.Window, c(p["graphite"]))
    pal.setColor(QtGui.QPalette.WindowText, c(p["ink"]))
    pal.setColor(QtGui.QPalette.Base, c(p["recess"]))
    pal.setColor(QtGui.QPalette.AlternateBase, c(p["panel"]))
    pal.setColor(QtGui.QPalette.Text, c(p["ink"]))
    pal.setColor(QtGui.QPalette.Button, c(p["panel"]))
    pal.setColor(QtGui.QPalette.ButtonText, c(p["ink"]))
    pal.setColor(QtGui.QPalette.ToolTipBase, c(p["panel"]))
    pal.setColor(QtGui.QPalette.ToolTipText, c(p["ink"]))
    pal.setColor(QtGui.QPalette.Highlight, c(p["cyan"]))
    pal.setColor(QtGui.QPalette.HighlightedText, c(p["recess"]))
    pal.setColor(QtGui.QPalette.Disabled, QtGui.QPalette.Text, c(p["muted"]))
    pal.setColor(QtGui.QPalette.Disabled, QtGui.QPalette.ButtonText, c(p["muted"]))
    app.setPalette(pal)

    mono = "Cascadia Code" if "Cascadia Code" in QFontDatabase().families() else "Consolas"
    app.setStyleSheet(build_qss(mono))
