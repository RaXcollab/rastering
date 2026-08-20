"""Instrument-console theme: dark (default) + high-contrast light variant.

One QSS stylesheet from palette tokens + Fusion QPalette. Applied at startup
in main_rastering.py; re-applied live from the View > Light theme toggle in
ui.py. Approved visual spec (dark):
docs/superpowers/specs/2026-08-12-rastering-ui-redesign-design.md (parent
repo); light variant + larger text added 2026-08-13 on operator feedback.
Qt QSS has no text-transform/letter-spacing -- group titles are written
uppercase-free in the .ui and styled by color/weight only.
"""
from __future__ import annotations

PALETTE = {
    "graphite": "#161A20",   # window ground
    "panel":    "#1E242C",   # group boxes, dock
    "recess":   "#12151A",   # camera well, input wells, status bar
    "line":     "#313A44",
    "ink":      "#E8EEF4",
    "muted":    "#A2AFBD",
    "cyan":     "#3EB4C8",   # interactive emphasis + armed state ONLY
    "cyan_dim": "#2A7D8C",
    "amber":    "#E2A83D",   # annunciator warn
    "green":    "#52BE6E",   # annunciator good
    "red":      "#E15A4D",   # annunciator alert (REC)
    # chip_idle sits BELOW every active chip color so annunciator prominence
    # tracks significance: idle must be the quietest thing on the strip.
    "chip_idle": "#7A8794",
    # press = pressed/selected well. Dark: same as recess (unchanged look).
    "press":    "#12151A",
}

# High-contrast light variant (2026-08-13 operator feedback): near-black ink
# on white wells; accent colors darkened so they stay readable on white.
PALETTE_LIGHT = {
    "graphite": "#EEF1F4",
    "panel":    "#F8FAFB",
    "recess":   "#FFFFFF",
    "line":     "#B6BFC8",
    "ink":      "#14181D",
    "muted":    "#45505B",
    "cyan":     "#0E6E7E",
    "cyan_dim": "#4899A8",
    "amber":    "#8F6508",
    "green":    "#1F7A3D",
    "red":      "#C13B2E",
    "chip_idle": "#8A96A2",  # deliberately quiet -- idle chips carry no news
    "press":    "#D8DFE6",   # darker than panel so press/selection is visible
}


def build_qss(mono: str, p: dict = PALETTE, base_pt: float = 9.0) -> str:
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
}}
QTabWidget::pane {{ border: 1px solid {p['line']}; }}
QTabBar::tab {{
    background: {p['press']};
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
QPushButton:pressed {{ background: {p['press']}; }}
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
}}
QStatusBar {{
    background: {p['recess']};
    border-top: 1px solid {p['line']};
}}
#status_strip {{
    background: {p['recess']};
    border-bottom: 1px solid {p['line']};
}}
#status_strip QLabel {{
    font-family: "{mono}";
    font-weight: 600;
    border: 1px solid {p['line']};
    border-radius: 3px;
    padding: 2px 8px;
    margin: 1px 2px;
}}
/* Current motor position is the readout operators look for -- render it
   larger, in full-strength ink instead of the quiet idle gray (ID selector
   outranks the [chipState] rules below). */
#chip_motor {{
    font-size: {base_pt * 1.5:.1f}pt;
    color: {p['ink']};
    border-color: {p['muted']};
}}
QLabel[hint="true"] {{ color: {p['muted']}; }}
QLabel[chipState="idle"]  {{ color: {p['chip_idle']}; }}
QLabel[chipState="good"]  {{ color: {p['green']}; border-color: {p['green']}; }}
QLabel[chipState="cyan"]  {{ color: {p['cyan']};  border-color: {p['cyan_dim']}; }}
QLabel[chipState="warn"]  {{ color: {p['amber']}; border-color: {p['amber']}; }}
QLabel[chipState="alert"] {{ color: {p['red']};   border-color: {p['red']}; }}
QMenuBar {{ background: {p['graphite']}; }}
QMenuBar::item:selected {{ background: {p['panel']}; }}
QMenu {{ background: {p['panel']}; border: 1px solid {p['line']}; }}
QMenu::item:selected {{ background: {p['press']}; color: {p['cyan']}; }}
QProgressBar {{ background: {p['recess']}; border: 1px solid {p['line']}; }}
QDockWidget {{ background: {p['panel']}; }}
"""


_BASE_PT = None      # system default font size, captured before the first override
_MONO = None         # resolved mono font family (never changes at runtime)
_APPLIED_LIGHT = False

# Theme is an operator preference, deliberately machine-wide (one registry key
# shared by all worktrees/instances) -- unlike calibration_data.json, which is
# per-worktree experiment state. Sole reader/writer of this key.
_SETTINGS_KEY = ("RaX", "rastering-gui")


def is_light() -> bool:
    """The theme actually applied (not the stored preference)."""
    return _APPLIED_LIGHT


def load_light_pref() -> bool:
    from PyQt5.QtCore import QSettings
    return QSettings(*_SETTINGS_KEY).value("light_theme", False, type=bool)


def save_light_pref(on: bool) -> None:
    from PyQt5.QtCore import QSettings
    QSettings(*_SETTINGS_KEY).setValue("light_theme", bool(on))


def apply_theme(app, light: bool = False) -> None:
    """Apply the dark (default) or light theme. Safe to re-call at runtime."""
    from PyQt5 import QtGui

    global _BASE_PT, _MONO, _APPLIED_LIGHT
    if _BASE_PT is None:
        # pointSizeF: Windows default is a fractional 8.25pt -- integer
        # pointSize() would truncate it and silently shrink the whole GUI.
        _BASE_PT = app.font().pointSizeF()
        if _BASE_PT <= 0:  # pixel-sized system font reports -1
            _BASE_PT = 9.0
        # First call only: setStyle + font scan repolish the whole widget
        # tree; on runtime toggles their answers never change.
        app.setStyle("Fusion")
        from PyQt5.QtGui import QFontDatabase
        _MONO = ("Cascadia Code"
                 if "Cascadia Code" in QFontDatabase().families() else "Consolas")
    _APPLIED_LIGHT = bool(light)
    p = PALETTE_LIGHT if light else PALETTE
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

    # Light mode reads one point larger (2026-08-13 operator feedback).
    pt = _BASE_PT + 1 if light else _BASE_PT
    f = app.font()
    f.setPointSizeF(pt)
    app.setFont(f)

    app.setStyleSheet(build_qss(_MONO, p, pt))
