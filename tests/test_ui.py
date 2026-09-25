"""Smoke tests for the window and tray, run offscreen with a fake engine."""

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication  # noqa: E402

from test_app import make_controller  # noqa: E402
from timbrel.core.params import ParamSpec  # noqa: E402
from timbrel.ui.main_window import MainWindow, Tray  # noqa: E402
from timbrel.ui.widgets import LevelMeter, ParamSlider  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def window(qapp, tmp_path):
    controller = make_controller(tmp_path)
    controller.start()
    win = MainWindow(controller)
    win.poll_timer.stop()  # the fake engine never produces audio blocks
    win.tray = Tray(win)
    yield win
    win.timer.stop()
    win.poll_timer.stop()
    win.tray.hide()
    win.deleteLater()


def sliders(window):
    return window.effects_area.widget().findChildren(ParamSlider)


def test_window_shows_devices_presets_and_state(window):
    assert window.input_combo.currentText() == "Microphone (USB Mic)"
    assert window.output_combo.currentText() == "CABLE Input (VB-Audio Virtual Cable)"
    assert window.preset_list.count() == 8
    assert window.preset_list.currentItem().text() == "Clean"
    assert "EFFECTS ON" in window.bypass_button.text()
    assert not window.cable_banner.isVisibleTo(window)


def test_bypass_button_toggles_everywhere(window):
    window.bypass_button.click()
    assert window.controller.bypassed
    assert "BYPASSED" in window.bypass_button.text()
    assert "BYPASSED" in window.windowTitle()
    assert window.tray.bypass_action.isChecked()
    assert "BYPASSED" in window.tray.toolTip()


def test_tray_bypass_and_preset_menu(window):
    window.tray.bypass_action.trigger()
    assert window.controller.bypassed
    robot = next(a for a in window.tray.presets_menu.actions() if a.text() == "Robot")
    robot.trigger()
    assert window.controller.preset.name == "Robot"
    assert window.preset_list.currentItem().text() == "Robot"


def test_selecting_preset_rebuilds_sliders(window):
    window.preset_list.setCurrentRow(4)  # Robot
    assert window.controller.preset.name == "Robot"
    names = [s.name for s in sliders(window)]
    assert names[:2] == ["threshold_db", "release_ms"]  # gate first
    assert "freq_hz" in names
    assert "mix" in names


def test_slider_changes_live_param(window):
    window.controller.select_preset("Deep")
    slider = next(s for s in sliders(window) if s.name == "semitones")
    slider.slider.setValue(0)  # far left = -12
    assert window.controller.effects[0].params["semitones"] == -12
    assert "modified" in window.preset_title.text()


def test_rename_delete_disabled_for_builtins(window):
    assert not window.rename_button.isEnabled()
    window.controller.save_preset_as("Mine")
    assert window.rename_button.isEnabled()
    assert window.preset_list.currentItem().text() == "Mine"


def test_close_hides_to_tray(window):
    window.show()
    window.close()
    assert not window.isVisible()
    assert window.controller.engine is not None  # audio keeps running


def test_meter_tracks_peaks(qapp):
    meter = LevelMeter()
    meter.set_peak(1.0)
    assert meter.level_db == 0.0
    meter.set_peak(0.0)
    assert -2.0 < meter.level_db < 0.0  # falls gradually, not instantly


def test_param_slider_maps_range(qapp):
    slider = ParamSlider("semitones", ParamSpec(0, -12, 12, "st"), 5)
    assert slider.value() == pytest.approx(5, abs=0.03)
    assert slider.readout.text() == "5.0 st"
