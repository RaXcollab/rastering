"""Guards against the 2026-08-11 progressive-slowdown class of bugs.

Root cause then: the ~4 Hz telemetry poll fed _on_target_position, which
appended EVERY tick to self._history and redrew the full scatter each time
(position_history_20260810_211138.csv: 164,244 rows, 222 unique positions;
ScatterPlotItem.setData at 164k pts = ~300 ms on this machine).

CAMERA CAVEAT (same as test_command_queue.py): importing ui.py pulls in
PyQt5 + pyueye. Never run while the rastering GUI is running. Skips
cleanly where ui.py is not importable.

Standalone-runnable:
    conda activate rastering && python tests/test_ui_slowdown_guards.py
"""

from __future__ import annotations

import os
import sys
import types
from unittest import mock

# ui.py lives one level up from tests/.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import ui  # noqa: E402
    _UI_IMPORT_ERROR = None
except Exception as e:  # noqa: BLE001
    ui = None
    _UI_IMPORT_ERROR = e


def _require_ui():
    if ui is None:
        import pytest
        pytest.skip(f"ui.py not importable here: {_UI_IMPORT_ERROR}")


def _target_position_self(save_checked: bool = True) -> types.SimpleNamespace:
    """Duck-typed `self` carrying ONLY what _on_target_position touches.

    Same unbound-method pattern as test_command_queue.py: the REAL method
    body runs with zero hardware and no Qt event loop.
    """
    return types.SimpleNamespace(
        current_target_marker=mock.Mock(name="current_target_marker"),
        checkBox_2=types.SimpleNamespace(isChecked=lambda: save_checked),
        _history=[],
        _pos_history_file=None,
        _pos_history_write_warned=False,
        _refresh_manual_scatter=mock.Mock(name="_refresh_manual_scatter"),
        _log=mock.Mock(name="_log"),
    )


def test_idle_poll_repeats_are_not_recorded():
    """The telemetry poll repeats the same position ~4x/s while the motor
    is idle. Only the FIRST occurrence may be recorded."""
    _require_ui()
    fake = _target_position_self()
    for _ in range(3):
        ui.RasterMainWindow._on_target_position(fake, 1.0, 2.0)
    assert fake._history == [(1.0, 2.0)]
    assert fake._refresh_manual_scatter.call_count == 1


def test_position_changes_are_recorded():
    _require_ui()
    fake = _target_position_self()
    ui.RasterMainWindow._on_target_position(fake, 1.0, 2.0)
    ui.RasterMainWindow._on_target_position(fake, 1.0, 2.0)
    ui.RasterMainWindow._on_target_position(fake, 3.5, 4.5)
    assert fake._history == [(1.0, 2.0), (3.5, 4.5)]
    assert fake._refresh_manual_scatter.call_count == 2


def test_csv_written_only_on_change():
    _require_ui()
    fake = _target_position_self()
    fake._pos_history_file = mock.Mock(name="pos_history_file")
    for _ in range(3):
        ui.RasterMainWindow._on_target_position(fake, 1.0, 2.0)
    assert fake._pos_history_file.write.call_count == 1


def test_unchecked_records_nothing_but_marker_still_moves():
    _require_ui()
    fake = _target_position_self(save_checked=False)
    ui.RasterMainWindow._on_target_position(fake, 1.0, 2.0)
    assert fake._history == []
    fake.current_target_marker.setData.assert_called_once_with([1.0], [2.0])


def _frame_self() -> types.SimpleNamespace:
    return types.SimpleNamespace(
        _latest_frame=None,
        set_frame=mock.Mock(name="set_frame"),
    )


def test_frames_coalesce_to_latest():
    """Two frames arrive while the GUI is busy; one render tick must show
    only the NEWEST -- older frames are dropped, never queued."""
    _require_ui()
    fake = _frame_self()
    ui.RasterMainWindow._store_frame(fake, "frame1")
    ui.RasterMainWindow._store_frame(fake, "frame2")
    ui.RasterMainWindow._render_latest_frame(fake)
    fake.set_frame.assert_called_once_with("frame2")


def test_render_with_no_pending_frame_is_a_noop():
    _require_ui()
    fake = _frame_self()
    ui.RasterMainWindow._render_latest_frame(fake)
    fake.set_frame.assert_not_called()


def test_render_consumes_the_frame():
    """A frame renders exactly once -- the next tick must not re-render it."""
    _require_ui()
    fake = _frame_self()
    ui.RasterMainWindow._store_frame(fake, "frame1")
    ui.RasterMainWindow._render_latest_frame(fake)
    ui.RasterMainWindow._render_latest_frame(fake)
    assert fake.set_frame.call_count == 1


if __name__ == "__main__":
    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS {name}")
            except Exception as e:  # noqa: BLE001
                failures += 1
                print(f"FAIL {name}: {e}")
    sys.exit(1 if failures else 0)
