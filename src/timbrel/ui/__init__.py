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
    from timbrel.ui.single_instance import InstanceServer, notify_running_instance, server_name

    app = QApplication.instance() or QApplication(sys.argv)
    app.setApplicationName("Timbrel")
    app.setQuitOnLastWindowClosed(False)

    # One copy per user. The lock is claimed before anything slow (opening
    # devices can take seconds on a fresh machine), so a second launch in
    # that window can't slip through.
    lock = windows.InstanceLock(server_name())
    if not lock.acquired:
        notify_running_instance(retry_seconds=5.0)  # it may still be starting up
        return 0
    # Listen for "show" requests right away; show once the window exists.
    shown = {"window": None, "pending": False}

    def show_requested() -> None:
        if shown["window"] is not None:
            shown["window"].show_normal()
        else:
            shown["pending"] = True

    instance_server = InstanceServer(show_requested)
    app.aboutToQuit.connect(instance_server.close)
    app.aboutToQuit.connect(lock.release)

    data_dir = windows.app_data_dir()
    settings_path = data_dir / "settings.json"
    settings = Settings.load(settings_path)
    if preset:
        settings.preset = preset
    controller = Controller(settings, settings_path, PresetLibrary(data_dir / "presets"))
    controller.start()

    window = MainWindow(controller)
    shown["window"] = window
    if shown["pending"]:
        minimized = False  # someone launched Timbrel again while it was starting
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
