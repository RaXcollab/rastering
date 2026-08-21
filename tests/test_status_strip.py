"""status_strip pure-logic + widget tests. Camera-safe: status_strip
imports PyQt5 only (no ui.py, no pyueye). Widget tests run offscreen.
Standalone-runnable:
    conda activate rastering && python -m pytest tests/test_status_strip.py
"""
from __future__ import annotations

import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import status_strip as ss  # noqa: E402


def test_owner_state():
    assert ss.owner_state(False, None) == ("IDLE", "idle")
    assert ss.owner_state(True, "local") == ("LOCAL RUN", "good")
    assert ss.owner_state(True, "remote") == ("ARMED · BLACS", "cyan")
    assert ss.owner_state(False, "remote") == ("IDLE", "idle")  # stale source, inactive


def test_progress_text():
    assert ss.progress_text(0, 0) == "pt — / —"
    assert ss.progress_text(37, 150) == "pt 37 / 150"


def test_shots_text():
    assert ss.shots_text(None) == "— /pt"
    assert ss.shots_text(3) == "×3 /pt"


def test_motor_text():
    assert ss.motor_text(None, None) == "X — · Y — mm"
    assert ss.motor_text(4.2134, 1.0) == "X 4.213 · Y 1.000 mm"


def test_cal_state_priority():
    assert ss.cal_state(False, None, False, False) == ("cal —", "idle")
    assert ss.cal_state(False, (2, 4), False, False) == ("cal 2/4", "warn")
    assert ss.cal_state(True, None, False, False) == ("cal ✓", "good")
    assert ss.cal_state(True, None, True, False) == ("cal ✓ file", "good")
    assert ss.cal_state(True, None, True, True) == ("cal stale", "warn")


def test_fps_text():
    assert ss.fps_text(13.24, False) == ("cam 13.2 fps", "idle")
    assert ss.fps_text(13.2, True) == ("cam —", "warn")
    assert ss.fps_text(None, False) == ("cam —", "warn")


def test_geometry_stale_intersection_only():
    bundled = {"rotation_k": -1, "flip_x": False,
               "aoi": {"width": 656, "height": 440}}
    same = {"rotation_k": -1, "flip_x": False, "flip_y": True,  # extra key ignored
            "aoi": {"width": 656, "height": 440, "start_x": 0}}
    assert ss.geometry_stale(same, bundled) is False
    rotated = dict(same, rotation_k=2)
    assert ss.geometry_stale(rotated, bundled) is True
    aoi_changed = dict(same, aoi={"width": 328, "height": 440})
    assert ss.geometry_stale(aoi_changed, bundled) is True
    assert ss.geometry_stale(None, bundled) is False   # unknown != stale
    assert ss.geometry_stale(same, None) is False


def _strip():
    from PyQt5 import QtWidgets
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    bar = QtWidgets.QStatusBar()
    return app, bar, ss.StatusStrip(bar)


def test_warning_chips_start_hidden_and_toggle():
    _app, _bar, strip = _strip()
    for key in ("pending", "bounds", "rec"):
        assert not strip._chips[key].isVisibleTo(_bar)
    strip.set_warning("rec", True)
    assert strip._chips["rec"].isVisibleTo(_bar)
    strip.set_warning("rec", False)
    assert not strip._chips["rec"].isVisibleTo(_bar)


def _strip_row():
    from PyQt5 import QtWidgets
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    row = ss.StripRow()
    lay = QtWidgets.QHBoxLayout(row)
    lay.setContentsMargins(6, 3, 6, 3)
    strip = ss.StatusStrip(row)
    lay.addStretch(1)
    strip.set_chip("owner", *ss.owner_state(True, "remote"))
    strip.set_chip("motor", ss.motor_text(12.345, 9.876))
    strip.set_chip("progress", ss.progress_text(37, 400))
    strip.set_chip("cal", "cal ✓ file", "good")
    strip.set_chip("shots", ss.shots_text(3))
    strip.set_chip("fps", "cam 27.3 fps")
    return app, row, strip


def test_strip_row_drops_whole_chips_lowest_priority_first():
    app, row, strip = _strip_row()
    row.setFixedSize(2000, 36)
    row.show()  # offscreen platform: never on a real screen
    app.processEvents()
    assert all(not strip._chips[k].isHidden() for k in ss.ALWAYS_CHIPS)
    full = sum(strip._chips[k].sizeHint().width() for k in ss.ALWAYS_CHIPS)
    row.setFixedWidth(full - 40)
    app.processEvents()
    hidden = [k for k in ss.ALWAYS_CHIPS if strip._chips[k].isHidden()]
    # something dropped, whole chips only, from the low-priority end
    assert hidden
    assert hidden == list(ss.ALWAYS_CHIPS[-len(hidden):])
    assert "owner" not in hidden and "motor" not in hidden
    row.setFixedWidth(2000)  # space returns -> chips return
    app.processEvents()
    assert all(not strip._chips[k].isHidden() for k in ss.ALWAYS_CHIPS)


def test_strip_row_motor_never_squeezed_below_text():
    app, row, strip = _strip_row()
    motor = strip._chips["motor"]
    row.setFixedSize(220, 36)  # narrower than owner+motor alone
    row.show()
    app.processEvents()
    assert motor.minimumWidth() == motor.sizeHint().width()
    assert motor.width() >= motor.sizeHint().width()


def test_set_chip_updates_text_and_state_property():
    _app, _bar, strip = _strip()
    strip.set_chip("owner", "ARMED · BLACS", "cyan")
    lab = strip._chips["owner"]
    assert lab.text() == "ARMED · BLACS"
    assert lab.property("chipState") == "cyan"


def test_theme_qss_covers_every_chip_state():
    import theme
    qss = theme.build_qss("Consolas")
    for state in ("idle", "good", "cyan", "warn", "alert"):
        assert f'chipState="{state}"' in qss, f"QSS missing chip state {state}"
    for token in ("#161A20", "#12151A", "#3EB4C8", "#E2A83D", "#52BE6E", "#E15A4D"):
        assert token in qss, f"QSS missing palette token {token}"


def test_motor_chip_prominent_and_scales_with_base_pt():
    # The position readout gets its own larger rule; font-size must follow
    # the applied base point size (light theme runs one point larger).
    import theme
    assert "#chip_motor" in theme.build_qss("Consolas")
    assert "font-size: 15.0pt" in theme.build_qss("Consolas", theme.PALETTE, 10.0)


def test_palettes_stay_in_lockstep():
    # The palettes are independent hand-maintained literals; a key added to
    # one but not the other is a KeyError inside a Qt slot when the operator
    # clicks View > Light theme -- PyQt5 aborts the process on that.
    import theme
    assert set(theme.PALETTE) == set(theme.PALETTE_LIGHT)
    for pal in (theme.PALETTE, theme.PALETTE_LIGHT):
        qss = theme.build_qss("Consolas", pal)
        for state in ("idle", "good", "cyan", "warn", "alert"):
            assert f'chipState="{state}"' in qss
