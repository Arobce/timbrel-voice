"""Global hotkeys (FR8, FR9): bridge from the keyboard hook to the UI thread,
plus the dialog for rebinding them."""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping

from PySide6.QtCore import QObject, Qt, QTimer, Signal
from PySide6.QtGui import QKeySequence
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QHBoxLayout,
    QKeySequenceEdit,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from timbrel.app import Controller
from timbrel.platform.windows import DEFAULT_HOTKEYS, GlobalHotkeys, normalize_hotkey

ACTION_LABELS = {
    "bypass": "Bypass on/off",
    "prev_preset": "Previous preset",
    "next_preset": "Next preset",
}
DEBOUNCE_SECONDS = 0.25  # key auto-repeat must not flip bypass back and forth
HOOK_REINSTALL_MS = 30_000

# Qt's names for keys -> ours (platform.windows.KEY_CODES).
_QT_TO_KEYBOARD = {
    "pgup": "page up",
    "pgdown": "page down",
    "del": "delete",
    "ins": "insert",
    "return": "enter",
    "escape": "esc",
    "meta": "windows",
    "scrolllock": "scroll lock",
}


def qt_to_hotkey(sequence: QKeySequence) -> str:
    """Convert the first chord of a Qt key sequence, e.g. Ctrl+Shift+F9."""
    if sequence.isEmpty():
        return ""
    text = sequence.toString(QKeySequence.SequenceFormat.PortableText).split(",")[0]
    parts = [_QT_TO_KEYBOARD.get(p.strip().lower(), p.strip().lower()) for p in text.split("+")]
    return normalize_hotkey("+".join(parts))


def hotkey_to_qt(hotkey: str) -> QKeySequence:
    reverse = {v: k for k, v in _QT_TO_KEYBOARD.items()}
    parts = [reverse.get(p, p) for p in hotkey.split("+")] if hotkey else []
    return QKeySequence("+".join(p.capitalize() for p in parts))


class HotkeyBridge(QObject):
    """Receives hotkeys on the keyboard hook's thread and runs the actions on
    the UI thread (Qt queues signals across threads)."""

    triggered = Signal(str)

    def __init__(self, controller: Controller, hotkeys: GlobalHotkeys | None = None) -> None:
        super().__init__()
        self.controller = controller
        self.hotkeys = hotkeys or GlobalHotkeys()
        self.errors: list[str] = []
        self._last: dict[str, float] = {}
        self._actions: Mapping[str, Callable[[], None]] = {
            "bypass": controller.toggle_bypass,
            "prev_preset": lambda: controller.step_preset(-1),
            "next_preset": lambda: controller.step_preset(1),
        }
        self.triggered.connect(self._run, Qt.ConnectionType.QueuedConnection)
        self.bind()
        # Windows drops a slow low-level hook without telling anyone;
        # re-installing it regularly means a dropped hook heals itself.
        self._heal_timer = QTimer(self)
        self._heal_timer.timeout.connect(self.hotkeys.reinstall)
        self._heal_timer.start(HOOK_REINSTALL_MS)

    def bind(self) -> list[str]:
        bindings = {}
        for action, hotkey in self.controller.settings.hotkeys.items():
            if hotkey and action in self._actions:
                bindings[hotkey] = lambda a=action: self.triggered.emit(a)
        self.errors = self.hotkeys.bind(bindings)
        return self.errors

    def _run(self, action: str) -> None:
        now = time.monotonic()
        if now - self._last.get(action, 0.0) < DEBOUNCE_SECONDS:
            return
        self._last[action] = now
        self._actions[action]()

    def close(self) -> None:
        self._heal_timer.stop()
        self.hotkeys.clear()


class HotkeysDialog(QDialog):
    def __init__(self, current: Mapping[str, str], parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Hotkeys")
        layout = QVBoxLayout(self)
        note = QLabel(
            "Hotkeys work everywhere, including in full-screen games, and the key still "
            "reaches the game. They can't see keys while a game runs as administrator "
            "unless Timbrel does too."
        )
        note.setWordWrap(True)
        layout.addWidget(note)
        form = QFormLayout()
        self.edits: dict[str, QKeySequenceEdit] = {}
        for action, label in ACTION_LABELS.items():
            edit = QKeySequenceEdit(hotkey_to_qt(current.get(action, "")))
            edit.setMaximumSequenceLength(1)
            clear = QPushButton("Clear")
            clear.clicked.connect(edit.clear)
            row = QHBoxLayout()
            row.addWidget(edit, 1)
            row.addWidget(clear)
            form.addRow(label, row)
            self.edits[action] = edit
        layout.addLayout(form)
        self.error = QLabel()
        self.error.setStyleSheet("color:#c42b1c;")
        layout.addWidget(self.error)
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.RestoreDefaults
            | QDialogButtonBox.StandardButton.Ok
            | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self._accept)
        buttons.rejected.connect(self.reject)
        buttons.button(QDialogButtonBox.StandardButton.RestoreDefaults).clicked.connect(
            self._defaults
        )
        layout.addWidget(buttons)
        self.result_hotkeys: dict[str, str] = dict(current)

    def _defaults(self) -> None:
        for action, edit in self.edits.items():
            edit.setKeySequence(hotkey_to_qt(DEFAULT_HOTKEYS[action]))

    def _accept(self) -> None:
        chosen = {}
        try:
            for action, edit in self.edits.items():
                chosen[action] = qt_to_hotkey(edit.keySequence())
        except ValueError as exc:
            self.error.setText(str(exc))
            return
        used = [k for k in chosen.values() if k]
        if len(used) != len(set(used)):
            self.error.setText("Each hotkey can only be used for one action.")
            return
        self.result_hotkeys = chosen
        self.accept()
