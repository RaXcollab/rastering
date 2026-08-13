"""Structural assertions on raster_gui.ui (pure XML — camera-safe, no Qt).

Encodes the 2026-08-12 redesign: Run / Pattern / Setup tabs, deduplicated
controls, always-on status strip (strip itself is code, not .ui).
Standalone-runnable: conda activate rastering && python -m pytest tests/test_raster_gui_ui_structure.py
"""
from __future__ import annotations

import os
import xml.etree.ElementTree as ET

UI_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "raster_gui.ui")


def _root():
    return ET.parse(UI_PATH).getroot()


def _tab_widget(root):
    for w in root.iter("widget"):
        if w.get("class") == "QTabWidget":
            return w
    raise AssertionError("no QTabWidget in raster_gui.ui")


def _tab_titles(root):
    titles = []
    for tab in _tab_widget(root).findall("widget"):
        for attr in tab.findall("attribute"):
            if attr.get("name") == "title":
                titles.append(attr.find("string").text)
    return titles


def _names_under(el):
    return {w.get("name") for w in el.iter("widget")}


def _tab_by_title(root, title):
    for tab in _tab_widget(root).findall("widget"):
        for attr in tab.findall("attribute"):
            if attr.get("name") == "title" and attr.find("string").text == title:
                return tab
    raise AssertionError(f"no tab titled {title!r}")


def test_three_tabs_in_order():
    assert _tab_titles(_root()) == ["Run", "Pattern", "Setup"]


def test_run_tab_contents():
    names = _names_under(_tab_by_title(_root(), "Run"))
    for expected in ("group_raster", "start_button", "stop_button",
                     "raster_step_button", "raster_continuous_checkbox",
                     "sleepTimer", "checkBox_2", "group_jog", "group_move"):
        assert expected in names, f"{expected} missing from Run tab"


def test_pattern_tab_contents():
    names = _names_under(_tab_by_title(_root(), "Pattern"))
    for expected in ("group_pattern", "alg_choice", "group_steps", "xstep",
                     "ystep", "group_spiral", "group_bounds",
                     "enforce_bounds_checkbox", "path_button", "clearAll",
                     "save_button"):
        assert expected in names, f"{expected} missing from Pattern tab"


def test_setup_tab_contents():
    names = _names_under(_tab_by_title(_root(), "Setup"))
    for expected in ("calibrateButton", "group_calmat", "group_device_home",
                     "group_user_home", "group_backlash",
                     "group_display_options"):
        assert expected in names, f"{expected} missing from Setup tab"


def test_no_widget_name_lost_or_duplicated():
    # Every object name in the file must be unique (uic requires it).
    names = [w.get("name") for w in _root().iter("widget") if w.get("name")]
    dupes = {n for n in names if names.count(n) > 1}
    assert not dupes, f"duplicate object names: {dupes}"
