"""The "AI Voice (experimental)" section of the main window."""

from __future__ import annotations

import threading

from PySide6.QtCore import QUrl
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QGridLayout,
    QGroupBox,
    QLabel,
    QPushButton,
    QWidget,
)

from timbrel.ai import voices_dir
from timbrel.ai.voice_effect import AiVoice
from timbrel.app import Controller, Status
from timbrel.ui.widgets import ParamSlider

QUIET_MIC_BOOST_DB = 29.0  # the input boost is maxed out: the mic is too quiet


class AiPanel(QGroupBox):
    def __init__(self, controller: Controller, parent: QWidget | None = None) -> None:
        super().__init__("AI Voice (experimental)", parent)
        self.controller = controller
        # None = still checking; then (ok, message). Checked off the UI thread
        # because importing onnxruntime takes a moment.
        self._availability: tuple[bool, str] | None = None
        threading.Thread(target=self._check, name="timbrel-ai-check", daemon=True).start()

        grid = QGridLayout(self)
        s = controller.settings
        self.enable = QCheckBox("Use AI voice")
        self.enable.setToolTip(
            "Converts your voice into the chosen voice on your NVIDIA GPU. "
            "Adds about 0.4 s of delay. Use only voices you have the right to use."
        )
        self.enable.toggled.connect(self._enable_toggled)
        self.voice_combo = QComboBox()
        self.voice_combo.setMinimumWidth(180)
        self.voice_combo.activated.connect(self._voice_chosen)
        self.folder_button = QPushButton("Open voices folder")
        self.folder_button.clicked.connect(self._open_folder)
        self.pitch = ParamSlider(
            "semitones", AiVoice.PARAMS["semitones"], s.ai_semitones, label="pitch"
        )
        self.pitch.setToolTip("Shift your pitch toward the voice: about +12 for male to female.")
        self.pitch.changed.connect(self.controller.set_ai_param)
        self.strength = ParamSlider(
            "index_rate", AiVoice.PARAMS["index_rate"], s.ai_index_rate, label="strength"
        )
        self.strength.setToolTip(
            "How strongly to sound like the voice. High values can garble words; "
            "lower it if you hear syllables you didn't say."
        )
        self.strength.changed.connect(self.controller.set_ai_param)
        self.status = QLabel("Checking for a GPU…")
        self.status.setWordWrap(True)

        grid.addWidget(self.enable, 0, 0)
        grid.addWidget(self.voice_combo, 0, 1)
        grid.addWidget(self.folder_button, 0, 2)
        grid.addWidget(self.pitch, 1, 0, 1, 3)
        grid.addWidget(self.strength, 2, 0, 1, 3)
        grid.addWidget(self.status, 3, 0, 1, 3)
        grid.setColumnStretch(1, 1)
        self.refresh()

    def _check(self) -> None:
        try:
            self._availability = (True, self.controller.check_ai())
        except Exception as exc:  # noqa: BLE001 - shown to the user
            self._availability = (False, str(exc) or type(exc).__name__)

    # --- refresh ---------------------------------------------------------------

    def refresh(self) -> None:
        c = self.controller
        names = c.ai_voice_names()
        current = c.settings.ai_voice or c.ai_loading or self.voice_combo.currentText()
        self.voice_combo.blockSignals(True)
        if [self.voice_combo.itemText(i) for i in range(self.voice_combo.count())] != names:
            self.voice_combo.clear()
            self.voice_combo.addItems(names)
        if current in names:
            self.voice_combo.setCurrentText(current)
        self.voice_combo.blockSignals(False)

        on = c.ai_active or c.ai_loading is not None
        self.enable.blockSignals(True)
        self.enable.setChecked(on)
        self.enable.blockSignals(False)
        usable = bool(self._availability and self._availability[0]) and bool(names)
        self.enable.setEnabled(usable or on)
        self.voice_combo.setEnabled(bool(names))

    def tick(self, status: Status) -> None:
        if self._availability is not None and not self.enable.isEnabled():
            self.refresh()  # the GPU check has finished
        self.status.setText(self._status_text(status))

    def _status_text(self, status: Status) -> str:
        if status.ai_loading:
            return f"Loading {status.ai_loading} on the GPU…"
        if status.ai_error:
            return f"⚠ Couldn't start the AI voice: {status.ai_error}"
        if status.ai_voice:
            delay = self.controller.chain.latency_samples / 48
            text = (
                f"On: {status.ai_voice} · GPU {status.ai_compute_ms:.0f} ms per 100 ms step"
                f" · adds about {delay / 1000:.1f} s of delay"
            )
            if status.ai_boost_db >= QUIET_MIC_BOOST_DB:
                text += (
                    "\n⚠ Your mic is very quiet (boosted +30 dB): move closer or raise its "
                    "level in Windows for clearer results."
                )
            return text
        if self._availability is None:
            return "Checking for a GPU…"
        ok, message = self._availability
        if not ok:
            return f"Not available: {message}"
        if not self.controller.ai_voice_names():
            return (
                f"No voices yet. Put .onnx voice files (and optional .index files) in "
                f"{voices_dir()}. Use only voices you have the right to use."
            )
        return f"Ready ({message}). Pick a voice and tick “Use AI voice”."

    # --- actions -----------------------------------------------------------------

    def _enable_toggled(self, on: bool) -> None:
        name = self.voice_combo.currentText()
        self.controller.set_ai_voice(name if on and name else None)

    def _voice_chosen(self) -> None:
        if self.enable.isChecked():
            self.controller.set_ai_voice(self.voice_combo.currentText())

    def _open_folder(self) -> None:
        folder = voices_dir()
        folder.mkdir(parents=True, exist_ok=True)
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(folder)))
