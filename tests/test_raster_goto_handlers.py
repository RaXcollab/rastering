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
