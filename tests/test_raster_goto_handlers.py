"""UI-handler tests for the F2 "go to arbitrary site" controls (Stage 3).

(Named *_handlers.py, not *_ui.py: the repo .gitignore ignores `*_ui.py`,
which is reserved for pyuic-generated bindings.)

These invoke RasterMainWindow handler methods UNBOUND on a duck-typed stub --
no real QMainWindow, no camera. They import ui.py (PyQt5 + pyueye) and SKIP
cleanly if that import fails. RUN WITH THE LIVE GUI CLOSED (otherwise the ui
import blocks on the busy uEye camera).

Pins the select-then-confirm contract: spinbox / Ctrl+click only SELECT (never
move); only the "Move to selected" button commits, and only in step/stopped
mode (never continuous).

    conda activate rastering && pytest tests/test_raster_goto_handlers.py
"""

from __future__ import annotations

import os
import sys
import types

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class _Skip(Exception):
    pass


def _skip(msg: str) -> None:
    try:
        import pytest  # noqa: PLC0415
    except Exception:  # noqa: BLE001
        raise _Skip(msg)
    pytest.skip(msg)


def _win():
    """Import RasterMainWindow or skip cleanly."""
    try:
        from ui import RasterMainWindow  # noqa: PLC0415
    except Exception as e:  # noqa: BLE001
        _skip(f"ui.py not importable (needs PyQt5+pyueye / rastering env; close the GUI): {e!r}")
    return RasterMainWindow


class _Chk:
    def __init__(self, checked: bool) -> None:
        self._c = checked

    def isChecked(self) -> bool:
        return self._c


class _Ctl:
    def __init__(self) -> None:
        self.calls = []

    def select_path_index(self, n):
        self.calls.append(("select_index", int(n)))

    def select_nearest_path_point(self, x, y):
        self.calls.append(("select_nearest", float(x), float(y)))

    def request_go_to_path_index(self, n, **kw):
        self.calls.append(("goto", int(n), dict(kw)))
        return True

    def goto_selected_point(self, **kw):
        self.calls.append(("goto_selected", dict(kw)))
        return True


def test_goto_index_change_when_running_selects_no_move():
    W = _win()
    ctl = _Ctl()
    stub = types.SimpleNamespace(
        _raster_active_ui=True, controller=ctl,
        _raster_preview_pts=[(0.0, 0.0)],
        _apply_selection=lambda i, x, y: None,
    )
    W._on_goto_index_changed(stub, 3)
    assert ctl.calls == [("select_index", 3)]
    assert all(c[0] != "goto" for c in ctl.calls), "selection must NOT move motors"


def test_goto_index_change_when_idle_uses_preview():
    W = _win()
    applied = []
    stub = types.SimpleNamespace(
        _raster_active_ui=False, controller=_Ctl(),
        _raster_preview_pts=[(1.0, 2.0), (3.0, 4.0), (5.0, 6.0)],
        _apply_selection=lambda i, x, y: applied.append((i, x, y)),
    )
    W._on_goto_index_changed(stub, 1)
    assert applied == [(1, 3.0, 4.0)]


def test_goto_move_clicked_commits_by_coordinate_when_selected():
    """Move re-resolves the selected COORDINATE on the armed path (select-nearest)
    then commits via goto_selected_point -- never the stale preview index."""
    W = _win()
    ctl = _Ctl()
    stub = types.SimpleNamespace(
        raster_continuous_checkbox=_Chk(False),
        _selected_index=2, _selected_xy=(5.0, 6.0), _raster_active_ui=True,
        controller=ctl, _log=lambda m: None, _start_raster=lambda: None,
    )
    W._on_goto_move_clicked(stub)
    assert ("select_nearest", 5.0, 6.0) in ctl.calls, "must re-resolve by coordinate"
    assert ("goto_selected", {"source": "ui"}) in ctl.calls, "must commit via goto_selected_point"


def test_goto_move_clicked_rejected_in_continuous():
    W = _win()
    ctl = _Ctl()
    logs = []
    stub = types.SimpleNamespace(
        raster_continuous_checkbox=_Chk(True),
        _selected_index=2, _selected_xy=(1.0, 1.0), _raster_active_ui=True,
        controller=ctl, _log=lambda m: logs.append(m), _start_raster=lambda: None,
    )
    W._on_goto_move_clicked(stub)
    assert ctl.calls == [], "continuous -> no controller calls"
    assert logs, "should log a rejection"


def test_goto_move_clicked_no_selection_no_move():
    W = _win()
    ctl = _Ctl()
    logs = []
    stub = types.SimpleNamespace(
        raster_continuous_checkbox=_Chk(False),
        _selected_index=-1, _selected_xy=None, _raster_active_ui=True,
        controller=ctl, _log=lambda m: logs.append(m), _start_raster=lambda: None,
    )
    W._on_goto_move_clicked(stub)
    assert ctl.calls == [], "no selection -> no controller calls"
    assert logs


def test_select_on_path_when_running_uses_controller():
    W = _win()
    ctl = _Ctl()
    stub = types.SimpleNamespace(
        _raster_active_ui=True, controller=ctl,
        _raster_preview_pts=[], _apply_selection=lambda *a: None,
        _log=lambda m: None,
    )
    W._select_on_path(stub, 1.5, 2.5)
    assert ctl.calls == [("select_nearest", 1.5, 2.5)]


def test_select_on_path_when_idle_uses_preview_argmin():
    W = _win()
    applied = []
    stub = types.SimpleNamespace(
        _raster_active_ui=False, controller=_Ctl(),
        _raster_preview_pts=[(0.0, 0.0), (10.0, 10.0)],
        _apply_selection=lambda i, x, y: applied.append((i, x, y)),
        _log=lambda m: None,
    )
    W._select_on_path(stub, 9.0, 9.0)
    assert applied == [(1, 10.0, 10.0)]


def test_param_change_skips_when_active():
    """Live preview auto-refresh must NOT fire while a raster is armed/running."""
    W = _win()
    calls = []
    stub = types.SimpleNamespace(
        _raster_active_ui=True, _raster_preview_pts=[(0.0, 0.0)],
        _clear_raster_overlay=lambda: calls.append("clear"),
        _render_preview=lambda **k: calls.append("render"),
    )
    W._on_raster_param_changed(stub)
    assert calls == [], "no live refresh while a raster is armed/running"


def test_param_change_skips_when_no_preview():
    """Auto-refresh only keeps an EXISTING preview in sync; it never renders unprompted."""
    W = _win()
    calls = []
    stub = types.SimpleNamespace(
        _raster_active_ui=False, _raster_preview_pts=[],
        _clear_raster_overlay=lambda: calls.append("clear"),
        _render_preview=lambda **k: calls.append("render"),
    )
    W._on_raster_param_changed(stub)
    assert calls == [], "no auto-render until a preview exists"


def test_param_change_refreshes_existing_preview():
    """With a preview shown and not rastering, a param change clears the overlay
    (preserving hull/selection) then quietly re-renders."""
    W = _win()
    calls = []
    stub = types.SimpleNamespace(
        _raster_active_ui=False, _raster_preview_pts=[(0.0, 0.0), (1.0, 1.0)],
        _clear_raster_overlay=lambda: calls.append("clear"),
        _render_preview=lambda **k: calls.append(("render", k)),
    )
    W._on_raster_param_changed(stub)
    assert calls == ["clear", ("render", {"quiet": True})]


def test_param_change_resyncs_bounds_when_shown():
    """While the scan-bounds box is shown, a limit change must re-draw + re-enforce
    it (so enforcement never silently lags the displayed limits)."""
    W = _win()
    calls = []
    stub = types.SimpleNamespace(
        _bounds_item=object(), _raster_active_ui=False, _raster_preview_pts=[],
        _draw_and_enforce_bounds=lambda: calls.append("resync"),
        _clear_raster_overlay=lambda: calls.append("clear"),
        _render_preview=lambda **k: calls.append("render"),
    )
    W._on_raster_param_changed(stub)
    assert "resync" in calls, "bounds must re-sync when the box is shown"


def test_display_bounds_toggles_enforcement():
    """Enforce Bounds is a toggle: draw+enforce when off, clear when on."""
    W = _win()
    calls = []
    on = types.SimpleNamespace(
        _bounds_item=None, _current_bounds=lambda: (0.0, 1.0, 0.0, 1.0),
        _draw_and_enforce_bounds=lambda: calls.append("enforce"),
        _clear_bounds=lambda: calls.append("clear"),
        _log=lambda m: None,
    )
    W._display_bounds(on)
    assert calls == ["enforce"], "off -> on draws + enforces"

    calls.clear()
    off = types.SimpleNamespace(
        _bounds_item=object(), _current_bounds=lambda: (0.0, 1.0, 0.0, 1.0),
        _draw_and_enforce_bounds=lambda: calls.append("enforce"),
        _clear_bounds=lambda: calls.append("clear"),
        _log=lambda m: None,
    )
    W._display_bounds(off)
    assert calls == ["clear"], "on -> off clears + disables enforcement"


def test_step_mode_ui_gates_auto_raster_on_calibration():
    """Auto Raster Start + Step are disabled (with a 'Calibrate first' reason)
    until a calibration exists -- an uncalibrated raster runs in passthrough and
    drives the motors to nonsense positions."""
    W = _win()

    class _Btn:
        def __init__(self):
            self.enabled = None
            self.tip = None
        def setEnabled(self, v):
            self.enabled = v
        def setToolTip(self, t):
            self.tip = t

    class _Chk(_Btn):
        def __init__(self, checked):
            super().__init__()
            self._c = checked
        def isChecked(self):
            return self._c

    def _stub(cal, checked=False):
        return types.SimpleNamespace(
            _raster_active_ui=False,
            controller=types.SimpleNamespace(calibration=cal),
            start_button=_Btn(), raster_step_button=_Btn(),
            raster_continuous_checkbox=_Chk(checked),
            sleepTimer=_Btn(),
        )

    s = _stub(None)
    W._update_step_mode_ui(s)
    assert s.start_button.enabled is False, "uncalibrated -> Start disabled"
    assert s.raster_step_button.enabled is False, "uncalibrated -> Step disabled"
    assert "Calibrate first" in (s.start_button.tip or ""), "Start shows reason"

    s2 = _stub(object())
    W._update_step_mode_ui(s2)
    assert s2.start_button.enabled is True, "calibrated -> Start enabled"
    assert (s2.start_button.tip or "") == "", "calibrated -> no Start warning"


if __name__ == "__main__":
    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS {name}")
            except BaseException as e:  # noqa: BLE001
                if type(e).__name__ in ("_Skip", "Skipped"):
                    print(f"SKIP {name}: {e}")
                    continue
                failures += 1
                print(f"FAIL {name}: {type(e).__name__}: {e}")
    sys.exit(1 if failures else 0)
