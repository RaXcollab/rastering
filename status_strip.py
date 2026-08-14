"""Annunciator status strip for the rastering GUI (2026-08-12 redesign).

Pure chip-state functions first (unit-testable, no Qt needed), then the
StatusStrip widget wrapper. This module must never import ui, camera, or
pyueye -- its tests are camera-safe BECAUSE of that.
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

from PyQt5 import QtWidgets

ALWAYS_CHIPS = ("owner", "progress", "shots", "motor", "cal", "fps")
WARNING_CHIPS = ("pending", "bounds", "rec")
_WARNING_TEXT = {"pending": "PATH EDITED — RE-ARM", "bounds": "BOUNDS OFF", "rec": "● REC"}
_WARNING_STATE = {"pending": "warn", "bounds": "warn", "rec": "alert"}


def owner_state(active: bool, source) -> Tuple[str, str]:
    if active and source == "remote":
        return ("ARMED · BLACS", "cyan")
    if active:
        return ("LOCAL RUN", "good")
    return ("IDLE", "idle")


def progress_text(index: int, total: int) -> str:
    if total <= 0:
        return "pt — / —"
    return f"pt {index} / {total}"


def shots_text(n) -> str:
    return "— /pt" if n is None else f"×{int(n)} /pt"


def motor_text(mx, my) -> str:
    if mx is None or my is None:
        return "X — · Y — mm"
    return f"X {mx:.3f} · Y {my:.3f} mm"


def cal_state(has_cal: bool, collecting, from_file: bool, stale: bool) -> Tuple[str, str]:
    if collecting is not None:
        return (f"cal {int(collecting[0])}/{int(collecting[1])}", "warn")
    if not has_cal:
        return ("cal —", "idle")
    if stale:
        return ("cal stale", "warn")
    return ("cal ✓ file", "good") if from_file else ("cal ✓", "good")


def fps_text(fps, stalled: bool) -> Tuple[str, str]:
    if stalled or fps is None:
        return ("cam —", "warn")
    return (f"cam {fps:.1f} fps", "idle")


def geometry_stale(current: Optional[Dict[str, Any]],
                   bundled: Optional[Dict[str, Any]]) -> bool:
    """True when any key present in BOTH dicts disagrees (recursing into
    nested dicts, e.g. 'aoi'). Missing data is never stale -- an old
    calibration file without bundled settings must read 'cal ✓ file',
    not 'cal stale'."""
    if not current or not bundled:
        return False
    for key, b_val in bundled.items():
        if key not in current:
            continue
        c_val = current[key]
        if isinstance(b_val, dict) and isinstance(c_val, dict):
            if geometry_stale(c_val, b_val):
                return True
        elif c_val != b_val:
            return True
    return False


class StatusStrip:
    """Chips in an existing QStatusBar. Display-only: a consumer of state,
    never a raiser -- a raise in a status slot would yellow the operator
    GUI, so setters are guarded and idempotent."""

    def __init__(self, host) -> None:
        # host: a QStatusBar (chips as permanent widgets) or any widget with
        # a layout (chips appended -- the top-of-image strip row in ui.py).
        add = (host.addPermanentWidget if isinstance(host, QtWidgets.QStatusBar)
               else host.layout().addWidget)
        self._chips: Dict[str, QtWidgets.QLabel] = {}
        for key in ALWAYS_CHIPS + WARNING_CHIPS:
            lab = QtWidgets.QLabel()
            lab.setObjectName(f"chip_{key}")
            lab.setProperty("chipState", "idle")
            add(lab)
            self._chips[key] = lab
        for key in WARNING_CHIPS:
            self._chips[key].setText(_WARNING_TEXT[key])
            self._set_state(self._chips[key], _WARNING_STATE[key])
            self._chips[key].setVisible(False)

    @staticmethod
    def _set_state(lab: QtWidgets.QLabel, state: str) -> None:
        if lab.property("chipState") != state:
            lab.setProperty("chipState", state)
            lab.style().unpolish(lab)
            lab.style().polish(lab)

    def set_chip(self, key: str, text: str, state: str = "idle") -> None:
        lab = self._chips[key]
        if lab.text() != text:          # ~4 Hz telemetry feeds this; skip
            lab.setText(text)           # no-op repaints
        self._set_state(lab, state)

    def set_warning(self, key: str, on: bool) -> None:
        self._chips[key].setVisible(bool(on))
