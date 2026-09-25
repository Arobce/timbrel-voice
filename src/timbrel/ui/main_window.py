"""Main window (FR10, FR13) and tray icon (FR11)."""

from __future__ import annotations

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QAction, QActionGroup, QCloseEvent
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMenu,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSystemTrayIcon,
    QVBoxLayout,
    QWidget,
)

from timbrel.app import VOICE_TEST_SECONDS, Controller
from timbrel.core.effects import Effect
from timbrel.platform.windows import VB_CABLE_URL
from timbrel.presets import PresetError
from timbrel.ui.hotkeys import ACTION_LABELS, HotkeyBridge, HotkeysDialog
from timbrel.ui.widgets import BYPASS_COLOR, ON_COLOR, LevelMeter, ParamSlider, state_icon

BLOCK_SIZES = [128, 256, 512, 1024]
EFFECT_TITLES = {
    "gate": "Noise gate",
    "pitch": "Pitch shift",
    "robot": "Robot",
    "radio": "Radio",
    "echo": "Echo",
    "eq": "EQ",
    "compressor": "Compressor",
}


class MainWindow(QMainWindow):
    def __init__(self, controller: Controller) -> None:
        super().__init__()
        self.controller = controller
        self.tray: Tray | None = None
        self.hotkeys: HotkeyBridge | None = None
        self._devices_version = -1
        self._was_paused = False
        self._quitting = False
        self._told_about_tray = False
        self.setWindowTitle("Timbrel")
        self.setMinimumSize(760, 560)

        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)

        self.cable_banner = QLabel(
            "VB-Audio Virtual Cable isn't installed, so apps can't hear Timbrel yet. "
            f'Install it from <a href="{VB_CABLE_URL}">{VB_CABLE_URL}</a> and restart. '
            "Until then you can still listen through the monitor."
        )
        self.cable_banner.setOpenExternalLinks(True)
        self.cable_banner.setWordWrap(True)
        self.cable_banner.setStyleSheet(
            "background:#fff4ce; color:#5c4400; padding:8px; border-radius:4px;"
        )
        root.addWidget(self.cable_banner)
        self.error_banner = QLabel()
        self.error_banner.setWordWrap(True)
        self.error_banner.setStyleSheet(
            "background:#fde7e9; color:#8a1020; padding:8px; border-radius:4px;"
        )
        root.addWidget(self.error_banner)

        root.addWidget(self._build_devices())

        middle = QHBoxLayout()
        middle.addWidget(self._build_presets(), 1)
        middle.addWidget(self._build_effects(), 2)
        root.addLayout(middle, 1)

        root.addWidget(self._build_voice_test())
        root.addLayout(self._build_bottom())
        root.addWidget(self._build_settings())

        self.refresh_devices()
        self.refresh()
        controller.listeners.append(self.refresh)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.tick)
        self.timer.start(33)
        # Unplug detection and auto-resume.
        self.poll_timer = QTimer(self)
        self.poll_timer.timeout.connect(controller.poll)
        self.poll_timer.start(500)

    # --- construction --------------------------------------------------------

    def _build_devices(self) -> QGroupBox:
        box = QGroupBox("Devices")
        grid = QGridLayout(box)
        self.input_combo = QComboBox()
        self.output_combo = QComboBox()
        self.monitor_combo = QComboBox()
        self.monitor_check = QCheckBox("Monitor (hear yourself)")
        grid.addWidget(QLabel("Microphone"), 0, 0)
        grid.addWidget(self.input_combo, 0, 1)
        grid.addWidget(QLabel("Output (apps hear this)"), 1, 0)
        grid.addWidget(self.output_combo, 1, 1)
        grid.addWidget(self.monitor_check, 2, 0)
        grid.addWidget(self.monitor_combo, 2, 1)
        grid.setColumnStretch(1, 1)
        for combo in (self.input_combo, self.output_combo, self.monitor_combo):
            combo.activated.connect(self._devices_changed)
        self.monitor_check.toggled.connect(self.controller.set_monitor_enabled)
        return box

    def _build_presets(self) -> QGroupBox:
        box = QGroupBox("Presets")
        layout = QVBoxLayout(box)
        self.preset_list = QListWidget()
        self.preset_list.itemClicked.connect(self._preset_clicked)
        self.preset_list.currentItemChanged.connect(self._preset_current_changed)
        layout.addWidget(self.preset_list, 1)
        row = QHBoxLayout()
        self.save_button = QPushButton("Save as…")
        self.rename_button = QPushButton("Rename…")
        self.delete_button = QPushButton("Delete")
        self.save_button.clicked.connect(self._save_preset)
        self.rename_button.clicked.connect(self._rename_preset)
        self.delete_button.clicked.connect(self._delete_preset)
        for button in (self.save_button, self.rename_button, self.delete_button):
            row.addWidget(button)
        layout.addLayout(row)
        return box

    def _build_effects(self) -> QGroupBox:
        box = QGroupBox("Effects")
        layout = QVBoxLayout(box)
        self.preset_title = QLabel()
        self.preset_title.setStyleSheet("font-weight:600;")
        self.preset_description = QLabel()
        self.preset_description.setWordWrap(True)
        layout.addWidget(self.preset_title)
        layout.addWidget(self.preset_description)
        self.effects_area = QScrollArea()
        self.effects_area.setWidgetResizable(True)
        layout.addWidget(self.effects_area, 1)
        self._built_for: tuple[int, ...] = ()
        return box

    def _build_voice_test(self) -> QGroupBox:
        box = QGroupBox("Voice test: hear what you sound like")
        row = QHBoxLayout(box)
        self.record_button = QPushButton(f"\u25cf  Record {VOICE_TEST_SECONDS:.0f} s")
        self.record_button.clicked.connect(self._record_test)
        self.record_progress = QProgressBar()
        self.record_progress.setRange(0, 100)
        self.record_progress.setTextVisible(False)
        self.record_progress.setMaximumWidth(120)
        self.play_fx_button = QPushButton("\u25b6  Play with effects")
        self.play_fx_button.clicked.connect(lambda: self._play_test(processed=True))
        self.play_raw_button = QPushButton("\u25b6  Play original")
        self.play_raw_button.clicked.connect(lambda: self._play_test(processed=False))
        self.stop_test_button = QPushButton("\u25a0  Stop")
        self.stop_test_button.clicked.connect(self.controller.stop_test_playback)
        self.test_hint = QLabel()
        self.test_hint.setWordWrap(True)
        for widget in (
            self.record_button,
            self.record_progress,
            self.play_fx_button,
            self.play_raw_button,
            self.stop_test_button,
        ):
            row.addWidget(widget)
        row.addWidget(self.test_hint, 1)
        self._update_voice_test()
        return box

    def _record_test(self) -> None:
        if not self.controller.start_voice_test():
            QMessageBox.information(
                self, "Audio isn't running", "Start audio (check the devices) and try again."
            )
        self._update_voice_test()

    def _play_test(self, processed: bool) -> None:
        self.controller.play_voice_test(processed=processed)
        self._update_voice_test()

    def _update_voice_test(self) -> None:
        state, progress = self.controller.voice_test_state()
        running = self.controller.engine is not None
        self.record_button.setEnabled(running and state != "recording")
        self.record_progress.setValue(int(progress * 100))
        self.play_fx_button.setEnabled(state in ("ready", "playing"))
        self.play_raw_button.setEnabled(state in ("ready", "playing"))
        self.stop_test_button.setEnabled(state == "playing")
        hints = {
            "idle": "Record a few seconds, then play it back with any preset. Nothing is saved.",
            "recording": "Recording\u2026 talk normally.",
            "ready": f"Pick a preset and press play to hear it with {self.controller.preset.name}"
            ". Plays on your monitor device (or speakers).",
            "playing": "Playing\u2026 apps hear silence from Timbrel until it finishes.",
        }
        self.test_hint.setText(hints[state])

    def _build_bottom(self) -> QHBoxLayout:
        row = QHBoxLayout()
        self.bypass_button = QPushButton()
        self.bypass_button.setCheckable(True)
        self.bypass_button.setMinimumSize(200, 64)
        self.bypass_button.clicked.connect(self.controller.set_bypass)
        row.addWidget(self.bypass_button)

        meters = QFormLayout()
        self.input_meter = LevelMeter()
        self.output_meter = LevelMeter()
        meters.addRow("In", self.input_meter)
        meters.addRow("Out", self.output_meter)
        self.status_label = QLabel()
        self.status_label.setToolTip(
            "Latency is Windows' own estimate for the audio streams plus effect delay; "
            "real delay measured through VB-Cable is usually 10-15 ms lower. "
            "Dropouts counts audio glitches on the mic-to-apps path (should stay 0)."
        )
        meters.addRow("", self.status_label)
        row.addLayout(meters, 1)
        return row

    def _build_settings(self) -> QGroupBox:
        box = QGroupBox("Settings")
        row = QHBoxLayout(box)
        s = self.controller.settings
        self.exclusive_check = QCheckBox("Exclusive mic (lower latency)")
        self.exclusive_check.setToolTip(
            "Opens the microphone exclusively: about 18 ms less delay, but other apps "
            "can't use the mic directly while Timbrel runs. Falls back automatically "
            "if the mic is busy."
        )
        self.exclusive_check.setChecked(s.exclusive_mic)
        self.exclusive_check.toggled.connect(self.controller.set_exclusive_mic)
        self.block_combo = QComboBox()
        for size in BLOCK_SIZES:
            self.block_combo.addItem(f"{size} samples ({size / 48:.1f} ms)", size)
        self.block_combo.setCurrentIndex(max(0, self.block_combo.findData(s.block_size)))
        self.block_combo.activated.connect(
            lambda: self.controller.set_block_size(self.block_combo.currentData())
        )
        self.autostart_check = QCheckBox("Start with Windows (minimized, Clean preset)")
        self.autostart_check.setChecked(s.start_with_windows)
        self.autostart_check.toggled.connect(self._autostart_toggled)
        self.hotkeys_button = QPushButton("Hotkeys…")
        self.hotkeys_button.clicked.connect(self._edit_hotkeys)
        row.addWidget(self.exclusive_check)
        row.addWidget(QLabel("Block size"))
        row.addWidget(self.block_combo)
        row.addStretch(1)
        row.addWidget(self.hotkeys_button)
        row.addWidget(self.autostart_check)
        return box

    # --- refresh -------------------------------------------------------------

    def refresh_devices(self) -> None:
        c, s = self.controller, self.controller.settings
        self._devices_version = c.devices_version

        def fill(
            combo: QComboBox,
            devices: list,
            selected: str | None,
            extra: tuple[str, str | None] | None,
        ) -> None:
            combo.blockSignals(True)
            combo.clear()
            if extra:
                combo.addItem(*extra)
            for device in devices:
                combo.addItem(device.name, device.name)
            index = combo.findData(selected) if selected is not None else -1
            combo.setCurrentIndex(index if index >= 0 else 0)
            combo.blockSignals(False)

        status = c.status()
        fill(self.input_combo, c.input_devices(), status.input_name or s.input_device, None)
        cable = c.cable
        output = s.output_device if s.output_device is not None else (cable.name if cable else "")
        fill(self.output_combo, c.output_devices(), output, ("None (monitor only)", ""))
        fill(self.monitor_combo, c.monitor_devices(), s.monitor_device, ("Default speakers", None))
        self.monitor_check.blockSignals(True)
        self.monitor_check.setChecked(s.monitor_enabled)
        self.monitor_check.blockSignals(False)

    def refresh(self) -> None:
        c = self.controller
        status = c.status()
        self.cable_banner.setVisible(not status.cable_found)
        self.error_banner.setVisible(bool(status.error))
        self.error_banner.setText(
            f"Audio paused: {status.error}. Timbrel keeps checking and resumes by itself "
            "when the device is back."
            if status.error
            else ""
        )
        if c.devices_version != self._devices_version:
            self.refresh_devices()
        self._notify_pause_change(bool(status.error), status.error)

        names = c.library.names()
        self.preset_list.blockSignals(True)
        if [self.preset_list.item(i).text() for i in range(self.preset_list.count())] != names:
            self.preset_list.clear()
            for preset in c.library.all():
                item = QListWidgetItem(preset.name)
                item.setToolTip(preset.description or ("Built-in" if preset.builtin else "Yours"))
                self.preset_list.addItem(item)
        for i in range(self.preset_list.count()):
            if self.preset_list.item(i).text() == c.preset.name:
                self.preset_list.setCurrentRow(i)
        self.preset_list.blockSignals(False)
        self.rename_button.setEnabled(not c.preset.builtin)
        self.delete_button.setEnabled(not c.preset.builtin)

        self._refresh_title()
        self.preset_description.setText(c.preset.description)
        self._rebuild_sliders()

        on = not c.bypassed
        self.bypass_button.setChecked(not on)
        self.bypass_button.setText(
            "EFFECTS ON\n(click to bypass)" if on else "BYPASSED\n(real voice)"
        )
        color = (ON_COLOR if on else BYPASS_COLOR).name()
        self.bypass_button.setStyleSheet(
            f"QPushButton {{ background:{color}; color:white; font-size:16px; "
            "font-weight:700; border-radius:6px; }"
        )
        self.setWindowIcon(state_icon(on))
        self._update_hotkey_hint()
        self.setWindowTitle(f"Timbrel — {'ON' if on else 'BYPASSED'} — {c.preset.name}")
        if self.tray is not None:
            self.tray.refresh()

    def _notify_pause_change(self, paused: bool, reason: str | None) -> None:
        # The window is often hidden (in a game), so say it in the tray too.
        if paused == self._was_paused:
            return
        self._was_paused = paused
        if self.tray is None:
            return
        if paused:
            self.tray.showMessage("Timbrel: audio paused", f"{reason}. Resumes by itself.")
        else:
            self.tray.showMessage("Timbrel: audio resumed", "Your voice effects are back on.")

    def _refresh_title(self) -> None:
        c = self.controller
        suffix = " (modified — Save as… to keep)" if c.modified else ""
        self.preset_title.setText(f"{c.preset.name}{suffix}")

    def _rebuild_sliders(self) -> None:
        c = self.controller
        key = (id(c.gate), *(id(fx) for fx in c.effects))
        if key == self._built_for:
            return
        self._built_for = key
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.addWidget(self._effect_box(None, c.gate))
        for index, effect in enumerate(c.effects):
            layout.addWidget(self._effect_box(index, effect))
        layout.addStretch(1)
        self.effects_area.setWidget(container)

    def _effect_box(self, index: int | None, effect: Effect) -> QGroupBox:
        box = QGroupBox(EFFECT_TITLES.get(effect.name, effect.name))
        layout = QVBoxLayout(box)
        for name, spec in effect.PARAMS.items():
            slider = ParamSlider(name, spec, effect.params[name])
            slider.changed.connect(
                lambda pname, value, target=index: self._param_changed(target, pname, value)
            )
            layout.addWidget(slider)
        return box

    def tick(self) -> None:
        self._update_voice_test()
        status = self.controller.status()
        self.input_meter.set_peak(status.input_peak)
        self.output_meter.set_peak(status.output_peak)
        if status.running:
            mic = "exclusive mic" if status.mic_exclusive else "shared mic"
            latency = f"~{status.latency_ms:.0f} ms (estimate)" if status.latency_ms else "–"
            self.status_label.setText(f"Latency {latency} · Dropouts {status.xruns} · {mic}")
        else:
            self.status_label.setText("Audio stopped")

    # --- actions -------------------------------------------------------------

    def _devices_changed(self) -> None:
        self.controller.set_devices(
            self.input_combo.currentData(),
            self.output_combo.currentData(),
            self.monitor_combo.currentData(),
        )

    def _preset_clicked(self, item: QListWidgetItem) -> None:
        self.controller.select_preset(item.text())

    def _preset_current_changed(self, current: QListWidgetItem | None, _: object) -> None:
        if current is not None and current.text() != self.controller.preset.name:
            self.controller.select_preset(current.text())

    def _param_changed(self, target: int | None, name: str, value: float) -> None:
        self.controller.set_param(target, name, value)
        self._refresh_title()

    def _ask_name(self, title: str, initial: str) -> str | None:
        name, ok = QInputDialog.getText(self, title, "Preset name:", text=initial)
        return name if ok and name.strip() else None

    def _save_preset(self) -> None:
        c = self.controller
        initial = c.preset.name if not c.preset.builtin else f"My {c.preset.name}"
        name = self._ask_name("Save preset", initial)
        if name is None:
            return
        existing = c.library.get(name)
        overwrite = False
        if existing is not None and not existing.builtin:
            answer = QMessageBox.question(self, "Replace preset?", f"Replace “{existing.name}”?")
            if answer != QMessageBox.StandardButton.Yes:
                return
            overwrite = True
        try:
            c.save_preset_as(name, overwrite=overwrite)
        except PresetError as exc:
            QMessageBox.warning(self, "Can't save preset", str(exc))

    def _rename_preset(self) -> None:
        c = self.controller
        name = self._ask_name("Rename preset", c.preset.name)
        if name is None:
            return
        try:
            c.rename_preset(c.preset.name, name)
        except PresetError as exc:
            QMessageBox.warning(self, "Can't rename preset", str(exc))

    def _delete_preset(self) -> None:
        c = self.controller
        answer = QMessageBox.question(self, "Delete preset?", f"Delete “{c.preset.name}”?")
        if answer == QMessageBox.StandardButton.Yes:
            try:
                c.delete_preset(c.preset.name)
            except PresetError as exc:
                QMessageBox.warning(self, "Can't delete preset", str(exc))

    def _edit_hotkeys(self) -> None:
        dialog = HotkeysDialog(self.controller.settings.hotkeys, self)
        if not dialog.exec():
            return
        self.controller.set_hotkeys(dialog.result_hotkeys)
        if self.hotkeys is not None and self.hotkeys.bind():
            QMessageBox.warning(
                self, "Some hotkeys didn't register", "\n".join(self.hotkeys.errors)
            )
        self._update_hotkey_hint()

    def _update_hotkey_hint(self) -> None:
        keys = self.controller.settings.hotkeys
        hint = ", ".join(
            f"{ACTION_LABELS[a]}: {keys[a].upper()}" for a in ACTION_LABELS if keys.get(a)
        )
        self.hotkeys_button.setToolTip(hint or "No hotkeys set")
        bypass_key = keys.get("bypass", "")
        self.bypass_button.setToolTip(f"Hotkey: {bypass_key.upper()}" if bypass_key else "")

    def _autostart_toggled(self, enabled: bool) -> None:
        try:
            self.controller.set_start_with_windows(enabled)
        except OSError as exc:
            QMessageBox.warning(self, "Couldn't change startup setting", str(exc))
            self.autostart_check.blockSignals(True)
            self.autostart_check.setChecked(not enabled)
            self.autostart_check.blockSignals(False)

    def show_normal(self) -> None:
        self.show()
        self.setWindowState(self.windowState() & ~Qt.WindowState.WindowMinimized)
        self.raise_()
        self.activateWindow()

    def quit(self) -> None:
        self._quitting = True
        self.controller.stop_test_playback()
        if self.hotkeys is not None:
            self.hotkeys.close()
        self.poll_timer.stop()
        self.controller.stop()
        if self.tray is not None:
            self.tray.hide()
        QApplication.quit()

    def closeEvent(self, event: QCloseEvent) -> None:  # noqa: N802
        # Close hides to the tray (FR11); Quit is in the tray menu.
        if self._quitting or self.tray is None:
            self.controller.stop()
            event.accept()
            return
        event.ignore()
        self.hide()
        if not self._told_about_tray:
            self._told_about_tray = True
            self.tray.showMessage(
                "Timbrel is still running",
                "Your voice effects stay on. Right-click the tray icon to bypass or quit.",
                state_icon(not self.controller.bypassed),
                4000,
            )


class Tray(QSystemTrayIcon):
    """Tray icon with Bypass, Presets and Quit (FR11); icon shows ON/BYPASSED."""

    def __init__(self, window: MainWindow) -> None:
        super().__init__(window)
        self.window = window
        self.controller = window.controller
        menu = QMenu()
        show = menu.addAction("Show Timbrel")
        show.triggered.connect(window.show_normal)
        self.bypass_action = QAction("Bypass (real voice)", menu, checkable=True)
        self.bypass_action.triggered.connect(self.controller.set_bypass)
        menu.addAction(self.bypass_action)
        self.presets_menu = menu.addMenu("Presets")
        self._preset_group = QActionGroup(menu)
        menu.addSeparator()
        quit_action = menu.addAction("Quit")
        quit_action.triggered.connect(window.quit)
        self._menu = menu
        self.setContextMenu(menu)
        self.activated.connect(self._activated)
        self._preset_names: list[str] = []
        self.refresh()

    def _activated(self, reason: QSystemTrayIcon.ActivationReason) -> None:
        if reason in (
            QSystemTrayIcon.ActivationReason.Trigger,
            QSystemTrayIcon.ActivationReason.DoubleClick,
        ):
            self.window.show_normal()

    def refresh(self) -> None:
        c = self.controller
        on = not c.bypassed
        self.setIcon(state_icon(on))
        state = "Effects ON" if on else "BYPASSED"
        if c.error:
            state += ", audio paused"
        self.setToolTip(f"Timbrel — {state} ({c.preset.name})")
        self.bypass_action.setChecked(not on)
        names = c.library.names()
        if names != self._preset_names:
            self._preset_names = names
            self.presets_menu.clear()
            for action in self._preset_group.actions():
                self._preset_group.removeAction(action)
            for name in names:
                action = QAction(name, self.presets_menu, checkable=True)
                action.triggered.connect(lambda _=False, n=name: c.select_preset(n))
                self._preset_group.addAction(action)
                self.presets_menu.addAction(action)
        for action in self._preset_group.actions():
            action.setChecked(action.text() == c.preset.name)
