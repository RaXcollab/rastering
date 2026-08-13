"""Duck-typed wiring tests for the 2026-08-12 UI redesign glue in ui.py.

CAMERA CAVEAT (same as test_ui_slowdown_guards.py): importing ui.py pulls
in PyQt5 + pyueye. Never run while the rastering GUI is running.
Standalone-runnable:
    conda activate rastering && python -m pytest tests/test_ui_redesign_wiring.py
"""
from __future__ import annotations

import os
import sys
import types
from unittest import mock

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


def _spiral_self(text):
    return types.SimpleNamespace(
        alg_choice=types.SimpleNamespace(currentText=lambda: text),
        group_spiral=mock.Mock(name="group_spiral"),
    )


def test_spiral_group_shown_for_spiral():
    _require_ui()
    fake = _spiral_self("Spiral Raster")
    ui.RasterMainWindow._update_spiral_visibility(fake)
    fake.group_spiral.setVisible.assert_called_once_with(True)


def test_spiral_group_hidden_for_square():
    _require_ui()
    fake = _spiral_self("Square Raster X")
    ui.RasterMainWindow._update_spiral_visibility(fake)
    fake.group_spiral.setVisible.assert_called_once_with(False)
