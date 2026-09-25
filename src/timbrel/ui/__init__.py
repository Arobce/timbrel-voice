"""PySide6 user interface."""

from __future__ import annotations

import sys

from PySide6.QtWidgets import QApplication, QSystemTrayIcon

from timbrel.app import Controller
from timbrel.platform import windows
from timbrel.presets import PresetLibrary
from timbrel.settings import Settings


def run_gui(minimized: bool = False, preset: str | None = None) -> int:
    from timbrel.ui.first_run import FirstRunDialog
    from timbrel.ui.main_window import MainWindow, Tray

    app = QApplication.instance() or QApplication(sys.argv)
    app.setApplicationName("Timbrel")
    app.setQuitOnLastWindowClosed(False)

    data_dir = windows.app_data_dir()
    settings_path = data_dir / "settings.json"
    settings = Settings.load(settings_path)
    if preset:
        settings.preset = preset
    controller = Controller(settings, settings_path, PresetLibrary(data_dir / "presets"))
    controller.start()

    window = MainWindow(controller)
    from timbrel.ui.hotkeys import HotkeyBridge

    window.hotkeys = HotkeyBridge(controller)
    if QSystemTrayIcon.isSystemTrayAvailable():
        window.tray = Tray(window)
        window.tray.show()
    else:
        app.setQuitOnLastWindowClosed(True)

    if not minimized or window.tray is None:
        window.show()
    if not settings.first_run_done:
        window.show_normal()
        FirstRunDialog(window).exec()
        controller.mark_first_run_done()
    return app.exec()
