"""Hotkey conversion, thread bridging and the rebind dialog (offscreen Qt)."""

import os
import threading

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtGui import QKeySequence  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from test_app import make_controller  # noqa: E402
from timbrel.platform.windows import DEFAULT_HOTKEYS  # noqa: E402
from timbrel.ui import hotkeys as hk  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


class FakeGlobalHotkeys:
    def __init__(self):
        self.bound = {}

    def bind(self, bindings):
        self.bound = dict(bindings)
        return []

    def clear(self):
        self.bound = {}

    def press(self, hotkey, from_thread=True):
        callback = self.bound[hotkey]
        if from_thread:
            t = threading.Thread(target=callback)
            t.start()
            t.join()
        else:
            callback()


@pytest.mark.parametrize(
    ("qt", "expected"),
    [("F9", "f9"), ("Ctrl+Shift+F9", "ctrl+shift+f9"), ("Alt+PgUp", "alt+page up")],
)
def test_qt_to_hotkey(qapp, qt, expected):
    assert hk.qt_to_hotkey(QKeySequence(qt)) == expected


def test_hotkey_round_trip(qapp):
    for hotkey in ("f9", "ctrl+shift+f10", "alt+page up"):
        assert hk.qt_to_hotkey(hk.hotkey_to_qt(hotkey)) == hotkey
    assert hk.qt_to_hotkey(QKeySequence()) == ""


def test_defaults_bound(qapp, tmp_path):
    fake = FakeGlobalHotkeys()
    hk.HotkeyBridge(make_controller(tmp_path), fake)
    assert set(fake.bound) == set(DEFAULT_HOTKEYS.values())


def test_hotkey_from_hook_thread_runs_on_ui_thread(qapp, tmp_path):
    controller = make_controller(tmp_path)
    fake = FakeGlobalHotkeys()
    bridge = hk.HotkeyBridge(controller, fake)  # keep a reference, as the window does
    ran_on = []
    controller.listeners.append(lambda: ran_on.append(threading.current_thread()))
    fake.press("f9")
    assert not controller.bypassed  # queued, not run on the hook thread
    qapp.processEvents()
    assert controller.bypassed
    assert ran_on == [threading.main_thread()]
    bridge.close()


def test_preset_hotkeys_step_through_presets(qapp, tmp_path):
    controller = make_controller(tmp_path)
    fake = FakeGlobalHotkeys()
    bridge = hk.HotkeyBridge(controller, fake)
    fake.press("f11", from_thread=False)
    qapp.processEvents()
    assert controller.preset.name == "Deep"
    bridge._last.clear()
    fake.press("f10", from_thread=False)
    qapp.processEvents()
    assert controller.preset.name == "Clean"


def test_key_repeat_is_debounced(qapp, tmp_path):
    controller = make_controller(tmp_path)
    fake = FakeGlobalHotkeys()
    bridge = hk.HotkeyBridge(controller, fake)  # noqa: F841 (kept alive)
    for _ in range(5):  # a held key auto-repeats
        fake.press("f9", from_thread=False)
    qapp.processEvents()
    assert controller.bypassed  # toggled once, not five times


def test_rebinding_uses_new_keys(qapp, tmp_path):
    controller = make_controller(tmp_path)
    fake = FakeGlobalHotkeys()
    bridge = hk.HotkeyBridge(controller, fake)
    controller.set_hotkeys({"bypass": "ctrl+b", "prev_preset": "", "next_preset": "f11"})
    bridge.bind()
    assert set(fake.bound) == {"ctrl+b", "f11"}


def test_dialog_rejects_duplicate_keys(qapp):
    dialog = hk.HotkeysDialog(DEFAULT_HOTKEYS)
    dialog.edits["next_preset"].setKeySequence(QKeySequence("F9"))
    dialog._accept()
    assert "only be used for one action" in dialog.error.text()
    dialog.edits["next_preset"].setKeySequence(QKeySequence("Ctrl+F11"))
    dialog._accept()
    assert dialog.result_hotkeys == {
        "bypass": "f9",
        "prev_preset": "f10",
        "next_preset": "ctrl+f11",
    }


def test_window_shows_paused_banner_and_resumes(qapp, tmp_path):
    from timbrel.ui.main_window import MainWindow

    controller = make_controller(tmp_path, settings=None)
    controller.start()
    win = MainWindow(controller)
    win.poll_timer.stop()
    try:
        for _ in range(4):  # no audio callbacks arrive: treated as unplugged
            controller.poll()
        assert win.error_banner.isVisibleTo(win)
        assert "resumes by itself" in win.error_banner.text()
        controller.start()
        assert not win.error_banner.isVisibleTo(win)
    finally:
        win.timer.stop()
        win.deleteLater()


# --- low-level matcher (no Windows hook needed) ----------------------------------

from timbrel.platform.windows import (  # noqa: E402
    HotkeyMatcher,
    normalize_hotkey,
    parse_hotkey,
)

VK_F9, VK_LCTRL, VK_RSHIFT, VK_A = 0x78, 0xA2, 0xA1, 0x41


def matcher_with(*hotkeys):
    m = HotkeyMatcher()
    hits = []
    m.set_bindings({h: (lambda h=h: hits.append(h)) for h in hotkeys})
    return m, hits


def press(m, *vks):
    for vk in vks:
        cb = m.feed(vk, True)
        if cb:
            cb()


def release(m, *vks):
    for vk in vks:
        m.feed(vk, False)


@pytest.mark.parametrize(
    ("text", "expected"),
    [("F9", "f9"), ("Shift + Ctrl+F10", "ctrl+shift+f10"), ("alt+Page Up", "alt+page up")],
)
def test_normalize_hotkey(text, expected):
    assert normalize_hotkey(text) == expected


@pytest.mark.parametrize("text", ["", "ctrl+", "ctrl", "f9+f10", "hyper+f9", "notakey"])
def test_invalid_hotkeys(text):
    with pytest.raises(ValueError):
        parse_hotkey(text)


def test_plain_key_fires_once_per_press():
    m, hits = matcher_with("f9")
    press(m, VK_F9, VK_F9, VK_F9)  # auto-repeat while held
    release(m, VK_F9)
    press(m, VK_F9)
    assert hits == ["f9", "f9"]


def test_modifiers_must_match_exactly():
    m, hits = matcher_with("f9", "ctrl+shift+f9")
    press(m, VK_LCTRL, VK_F9)  # ctrl+f9 isn't bound
    release(m, VK_F9, VK_LCTRL)
    press(m, VK_LCTRL, VK_RSHIFT, VK_F9)
    release(m, VK_F9, VK_RSHIFT, VK_LCTRL)
    press(m, VK_F9)
    assert hits == ["ctrl+shift+f9", "f9"]


def test_unbound_keys_do_nothing():
    m, hits = matcher_with("f9")
    press(m, VK_A)
    assert hits == []
