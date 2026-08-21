"""Annunciator status strip for the rastering GUI (2026-08-12 redesign).

Pure chip-state functions first (unit-testable, no Qt needed), then the
StatusStrip widget wrapper. This module must never import ui, camera, or
pyueye -- its tests are camera-safe BECAUSE of that.
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

from PyQt5 import QtWidgets

# Priority order: the strip row clips from the right when width runs short,
# so the chips an operator can lose (shots config, cam diagnostics) sit last.
ALWAYS_CHIPS = ("owner", "motor", "progress", "cal", "shots", "fps")
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


class StripRow(QtWidgets.QWidget):
    """Chip host for the top-of-image strip. Degrades by WHOLE chips: when
    the row runs out of width, the lowest-priority always-chips hide
    entirely (no half-clipped text, no chips painting over each other) and
    return as soon as space does. Owner, position, and warning chips never
    drop."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._strip: Optional["StatusStrip"] = None

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        if self._strip is not None:
            self._strip.fit_to_width(self.width())


class StatusStrip:
    """Chips in an existing QStatusBar. Display-only: a consumer of state,
    never a raiser -- a raise in a status slot would yellow the operator
    GUI, so setters are guarded and idempotent."""

    _NEVER_DROP = ("owner", "motor")

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
        self._row = host if isinstance(host, StripRow) else None
        if self._row is not None:
            self._row._strip = self

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
            if key == "motor":
                # An explicit minimum is the only floor an over-constrained
                # QHBoxLayout respects (Fixed policy gets scaled away, tested
                # 2026-08-20): even if every droppable chip is already gone,
                # the position readout keeps its full text.
                lab.setMinimumWidth(lab.sizeHint().width())
            self._refit()
        self._set_state(lab, state)

    def set_warning(self, key: str, on: bool) -> None:
        self._chips[key].setVisible(bool(on))
        self._refit()

    def _refit(self) -> None:
        if self._row is not None:
            self.fit_to_width(self._row.width())

    def fit_to_width(self, width: int) -> None:
        """Prefix-visibility for the droppable chips: chips sit in priority
        order, so the first one that no longer fits hides together with
        everything after it. Never-drop chips and visible warning chips are
        reserved off the top."""
        if self._row is None or width <= 0:
            return
        lay = self._row.layout()
        margins = lay.contentsMargins()
        space = lay.spacing()
        avail = width - margins.left() - margins.right()
        for key in self._NEVER_DROP:
            avail -= self._chips[key].sizeHint().width() + space
        for key in WARNING_CHIPS:
            if not self._chips[key].isHidden():
                avail -= self._chips[key].sizeHint().width() + space
        fits = True
        for key in ALWAYS_CHIPS:
            if key in self._NEVER_DROP:
                continue
            lab = self._chips[key]
            need = lab.sizeHint().width() + space
            fits = fits and need <= avail
            if fits:
                avail -= need
            lab.setVisible(fits)
