"""Windows audio device detection (WASAPI) and VB-Audio Virtual Cable lookup.

The pure functions take device/host-API lists in the shape returned by
``sounddevice.query_devices()`` / ``query_hostapis()`` so they can be tested
without audio hardware.
"""

from __future__ import annotations

import ctypes
import os
import sys
import threading
import winreg
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import sounddevice as sd

WASAPI_HOSTAPI_NAME = "Windows WASAPI"
# VB-Cable's playback endpoint. Apps then pick "CABLE Output" as their mic.
CABLE_PLAYBACK_NAME = "CABLE Input"
VB_CABLE_URL = "https://vb-audio.com/Cable/"

Direction = Literal["input", "output"]
DeviceList = Sequence[Mapping[str, Any]]


class DeviceError(Exception):
    """A device could not be found or chosen."""


@dataclass(frozen=True)
class Device:
    index: int
    name: str
    max_input_channels: int
    max_output_channels: int
    default_samplerate: float

    def supports(self, direction: Direction) -> bool:
        if direction == "input":
            return self.max_input_channels > 0
        return self.max_output_channels > 0


@dataclass(frozen=True)
class WasapiDevices:
    devices: list[Device]
    default_input: int | None
    default_output: int | None

    def of(self, direction: Direction) -> list[Device]:
        return [d for d in self.devices if d.supports(direction)]

    def by_index(self, index: int) -> Device | None:
        return next((d for d in self.devices if d.index == index), None)

    def default(self, direction: Direction) -> int | None:
        return self.default_input if direction == "input" else self.default_output


def wasapi_devices(devices: DeviceList, hostapis: DeviceList) -> WasapiDevices:
    """Return the WASAPI devices from full PortAudio device/host-API lists."""
    api_index = next(
        (i for i, api in enumerate(hostapis) if api["name"] == WASAPI_HOSTAPI_NAME), None
    )
    if api_index is None:
        raise DeviceError("WASAPI host API not found; Timbrel requires Windows 10/11.")
    api = hostapis[api_index]

    found = [
        Device(
            index=i,
            name=d["name"],
            max_input_channels=d["max_input_channels"],
            max_output_channels=d["max_output_channels"],
            default_samplerate=d["default_samplerate"],
        )
        for i, d in enumerate(devices)
        if d["hostapi"] == api_index
    ]
    default_in = api.get("default_input_device", -1)
    default_out = api.get("default_output_device", -1)
    return WasapiDevices(
        devices=found,
        default_input=default_in if default_in >= 0 else None,
        default_output=default_out if default_out >= 0 else None,
    )


def is_virtual_cable(device: Device) -> bool:
    """True for any VB-Audio Virtual Cable endpoint, input or output."""
    return "vb-audio" in device.name.lower()


def find_cable(wasapi: WasapiDevices) -> Device | None:
    """Return VB-Cable's playback endpoint ("CABLE Input ..."), if installed."""
    for device in wasapi.of("output"):
        if device.name.lower().startswith(CABLE_PLAYBACK_NAME.lower()):
            return device
    return None


def check_route(mic: Device, out: Device) -> None:
    """Refuse cable -> cable, which would feed our output back into our input."""
    if is_virtual_cable(mic) and is_virtual_cable(out):
        raise DeviceError(
            f"Input {mic.name!r} is the virtual cable itself, which would create a "
            "feedback loop. Choose your real microphone with --input."
        )


def default_device(direction: Direction, wasapi: WasapiDevices) -> Device:
    """The output defaults to VB-Cable and the input to the system default mic.

    VB-Cable's installer often makes "CABLE Output" the default mic; in that
    case the first real microphone is used instead.
    """
    if direction == "output":
        cable = find_cable(wasapi)
        if cable is None:
            raise DeviceError(
                "VB-Audio Virtual Cable not found. Install it from "
                f"{VB_CABLE_URL} or choose an output with --output."
            )
        return cable
    default = wasapi.default("input")
    device = wasapi.by_index(default) if default is not None else None
    if device is not None and device.supports("input") and not is_virtual_cable(device):
        return device
    real_mics = [d for d in wasapi.of("input") if not is_virtual_cable(d)]
    if not real_mics:
        raise DeviceError("No microphone found; choose one with --input.")
    return real_mics[0]


def resolve_device(spec: str | None, direction: Direction, wasapi: WasapiDevices) -> Device:
    """Pick a WASAPI device from a CLI spec: an index, a name fragment, or None
    (see ``default_device``)."""
    if spec is None:
        return default_device(direction, wasapi)

    candidates = wasapi.of(direction)
    spec = spec.strip()
    if spec.isdigit():
        device = wasapi.by_index(int(spec))
        if device is None or not device.supports(direction):
            raise DeviceError(
                f"Device {spec} is not a WASAPI {direction} device. "
                "Run with --list-devices to see valid choices."
            )
        return device

    needle = spec.lower()
    matches = [d for d in candidates if needle in d.name.lower()]
    exact = [d for d in matches if d.name.lower() == needle]
    if len(exact) == 1:
        return exact[0]
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise DeviceError(
            f"No WASAPI {direction} device matches {spec!r}. "
            "Run with --list-devices to see valid choices."
        )
    names = ", ".join(f"[{d.index}] {d.name}" for d in matches)
    raise DeviceError(f"{spec!r} matches several {direction} devices: {names}. Use the index.")


def find_by_name(name: str | None, direction: Direction, wasapi: WasapiDevices) -> Device | None:
    """The device with exactly this name (as saved in settings), if present."""
    if not name:
        return None
    return next((d for d in wasapi.of(direction) if d.name == name), None)


def query_wasapi_devices() -> WasapiDevices:
    return wasapi_devices(sd.query_devices(), sd.query_hostapis())


def stream_settings(device: Device | None = None, exclusive: bool = False) -> sd.WasapiSettings:
    """WASAPI settings for a device.

    VB-Cable's playback endpoint is always opened in exclusive mode: only
    Timbrel writes to it, and exclusive mode measures ~30 ms lower round-trip
    latency than shared mode. Other devices use shared mode, so other apps can
    keep using them, unless ``exclusive`` is set (the --exclusive-mic option:
    ~18 ms lower latency, but no other app can open the mic meanwhile).
    """
    cable_playback = device is not None and is_virtual_cable(device) and device.supports("output")
    if exclusive or cable_playback:
        return sd.WasapiSettings(exclusive=True)
    return shared_settings()


def shared_settings() -> sd.WasapiSettings:
    return sd.WasapiSettings(auto_convert=True)


# --- app data and start with Windows ------------------------------------------

APP_NAME = "Timbrel"
RUN_KEY = r"Software\Microsoft\Windows\CurrentVersion\Run"


def app_data_dir() -> Path:
    r"""%APPDATA%\Timbrel (settings.json and presets\)."""
    base = os.environ.get("APPDATA") or str(Path.home() / "AppData" / "Roaming")
    return Path(base) / APP_NAME


def autostart_command(preset: str = "Clean") -> str:
    """Command line that launches Timbrel minimized to the tray."""
    args = f'--minimized --preset "{preset}"'
    if getattr(sys, "frozen", False):  # PyInstaller build
        return f'"{sys.executable}" {args}'
    exe = Path(sys.executable)
    pythonw = exe.with_name("pythonw.exe")  # no console window
    return f'"{pythonw if pythonw.exists() else exe}" -m timbrel {args}'


def set_autostart(enabled: bool, command: str | None = None) -> None:
    """Add or remove Timbrel from the current user's startup programs."""
    with winreg.OpenKey(winreg.HKEY_CURRENT_USER, RUN_KEY, 0, winreg.KEY_SET_VALUE) as key:
        if enabled:
            winreg.SetValueEx(key, APP_NAME, 0, winreg.REG_SZ, command or autostart_command())
        else:
            try:
                winreg.DeleteValue(key, APP_NAME)
            except FileNotFoundError:
                pass


def autostart_enabled() -> bool:
    try:
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, RUN_KEY) as key:
            winreg.QueryValueEx(key, APP_NAME)
            return True
    except FileNotFoundError:
        return False


def rescan_devices() -> WasapiDevices:
    """Re-read the device list. PortAudio caches it at startup, so plugging a
    device back in is only seen after re-initialising. No streams may be open."""
    sd._terminate()
    sd._initialize()
    return query_wasapi_devices()


# --- global hotkeys -----------------------------------------------------------

DEFAULT_HOTKEYS = {"bypass": "f9", "prev_preset": "f10", "next_preset": "f11"}


_MODIFIER_VKS = {
    0x10: "shift", 0xA0: "shift", 0xA1: "shift",
    0x11: "ctrl", 0xA2: "ctrl", 0xA3: "ctrl",
    0x12: "alt", 0xA4: "alt", 0xA5: "alt",
    0x5B: "windows", 0x5C: "windows",
}  # fmt: skip
MODIFIERS = ("ctrl", "alt", "shift", "windows")
KEY_CODES: dict[str, int] = {
    **{f"f{i}": 0x6F + i for i in range(1, 25)},
    **{chr(c): c for c in range(ord("A"), ord("Z") + 1)},
    **{str(d): 0x30 + d for d in range(10)},
    **{f"num {d}": 0x60 + d for d in range(10)},
    "space": 0x20, "enter": 0x0D, "tab": 0x09, "esc": 0x1B, "backspace": 0x08,
    "pause": 0x13, "scroll lock": 0x91, "insert": 0x2D, "delete": 0x2E,
    "home": 0x24, "end": 0x23, "page up": 0x21, "page down": 0x22,
    "left": 0x25, "up": 0x26, "right": 0x27, "down": 0x28,
}  # fmt: skip
KEY_CODES = {name.lower(): vk for name, vk in KEY_CODES.items()}


def parse_hotkey(text: str) -> tuple[frozenset[str], int]:
    """ "ctrl+shift+f9" -> ({"ctrl", "shift"}, VK_F9). Exactly one non-modifier key."""
    parts = [p.strip().lower() for p in text.split("+")]
    if not parts or any(not p for p in parts):
        raise ValueError(f"{text!r} is not a valid hotkey")
    mods = frozenset(p for p in parts if p in MODIFIERS)
    keys = [p for p in parts if p not in MODIFIERS]
    if len(keys) != 1 or keys[0] not in KEY_CODES or len(mods) != len(parts) - 1:
        raise ValueError(f"{text!r} is not a valid hotkey")
    return mods, KEY_CODES[keys[0]]


def normalize_hotkey(text: str) -> str:
    """Validate a hotkey like "Ctrl + Shift+F9"; returns "ctrl+shift+f9"."""
    mods, vk = parse_hotkey(text)
    key = next(name for name, code in KEY_CODES.items() if code == vk)
    return "+".join([*(m for m in MODIFIERS if m in mods), key])


class HotkeyMatcher:
    """Turns raw key-down/up events into hotkey hits.

    Tracks which modifiers are held, fires on the key-down of the main key
    only when the held modifiers match exactly, and ignores auto-repeat.
    """

    def __init__(self) -> None:
        self.bindings: dict[tuple[frozenset[str], int], Callable[[], None]] = {}
        self._held: set[int] = set()

    def set_bindings(self, bindings: Mapping[str, Callable[[], None]]) -> None:
        self.bindings = {parse_hotkey(k): cb for k, cb in bindings.items()}

    def feed(self, vk: int, down: bool) -> Callable[[], None] | None:
        """Process one key event; returns the callback to run, if any."""
        if not down:
            self._held.discard(vk)
            return None
        repeat = vk in self._held
        self._held.add(vk)
        if repeat or vk in _MODIFIER_VKS:
            return None
        mods = frozenset(_MODIFIER_VKS[k] for k in self._held if k in _MODIFIER_VKS)
        return self.bindings.get((mods, vk))


class GlobalHotkeys:
    """System-wide hotkeys through a low-level keyboard hook (WH_KEYBOARD_LL).

    They fire while a full-screen game has focus, and keys are never
    swallowed: the game still gets them. They can't see keys sent to a program
    running as administrator unless Timbrel runs as administrator too.
    Callbacks run on the hook's thread and must return quickly (Windows drops
    hooks that stall): hand work to the UI thread before touching Qt.
    """

    WH_KEYBOARD_LL = 13
    WM_KEYDOWN, WM_KEYUP, WM_SYSKEYDOWN, WM_SYSKEYUP = 0x100, 0x101, 0x104, 0x105
    WM_QUIT = 0x0012

    def __init__(self) -> None:
        self.matcher = HotkeyMatcher()
        self._thread: threading.Thread | None = None
        self._thread_id = 0
        self._ready = threading.Event()
        self._error: str | None = None

    def bind(self, bindings: Mapping[str, Callable[[], None]]) -> list[str]:
        """Replace all hotkeys; returns error messages (empty on success)."""
        valid, errors = {}, []
        for hotkey, callback in bindings.items():
            try:
                parse_hotkey(hotkey)
                valid[hotkey] = callback
            except ValueError as exc:
                errors.append(str(exc))
        self.matcher.set_bindings(valid)
        if valid and self._thread is None:
            self._start()
        if self._error:
            errors.append(self._error)
        return errors

    def clear(self) -> None:
        self.matcher.set_bindings({})
        if self._thread is not None:
            ctypes.windll.user32.PostThreadMessageW(self._thread_id, self.WM_QUIT, 0, 0)
            self._thread.join(timeout=2)
            self._thread = None

    def _start(self) -> None:
        self._ready.clear()
        self._error = None
        self._thread = threading.Thread(target=self._run, name="timbrel-hotkeys", daemon=True)
        self._thread.start()
        self._ready.wait(timeout=2)

    def _run(self) -> None:
        from ctypes import wintypes as wt

        user32 = ctypes.WinDLL("user32", use_last_error=True)
        lresult = ctypes.c_ssize_t
        hook_proc_type = ctypes.WINFUNCTYPE(lresult, ctypes.c_int, wt.WPARAM, wt.LPARAM)
        user32.SetWindowsHookExW.argtypes = [ctypes.c_int, hook_proc_type, wt.HINSTANCE, wt.DWORD]
        user32.SetWindowsHookExW.restype = wt.HHOOK
        user32.CallNextHookEx.argtypes = [wt.HHOOK, ctypes.c_int, wt.WPARAM, wt.LPARAM]
        user32.CallNextHookEx.restype = lresult

        class KbdLlHookStruct(ctypes.Structure):
            _fields_ = [
                ("vkCode", wt.DWORD),
                ("scanCode", wt.DWORD),
                ("flags", wt.DWORD),
                ("time", wt.DWORD),
                ("dwExtraInfo", ctypes.c_size_t),
            ]

        down_msgs = (self.WM_KEYDOWN, self.WM_SYSKEYDOWN)
        up_msgs = (self.WM_KEYUP, self.WM_SYSKEYUP)

        def proc(code: int, wparam: int, lparam: int) -> int:
            if code >= 0 and wparam in down_msgs + up_msgs:
                vk = ctypes.cast(lparam, ctypes.POINTER(KbdLlHookStruct)).contents.vkCode
                callback = self.matcher.feed(vk, wparam in down_msgs)
                if callback is not None:
                    try:
                        callback()
                    except Exception:  # noqa: BLE001 - never let a hotkey kill the hook
                        pass
            return user32.CallNextHookEx(None, code, wparam, lparam)

        callback_ref = hook_proc_type(proc)  # keep alive for the hook's lifetime
        self._thread_id = ctypes.windll.kernel32.GetCurrentThreadId()
        hook = user32.SetWindowsHookExW(self.WH_KEYBOARD_LL, callback_ref, None, 0)
        if not hook:
            self._error = f"couldn't install keyboard hook (error {ctypes.get_last_error()})"
            self._ready.set()
            self._thread = None
            return
        self._ready.set()
        msg = wt.MSG()
        while user32.GetMessageW(ctypes.byref(msg), None, 0, 0) > 0:
            user32.TranslateMessage(ctypes.byref(msg))
            user32.DispatchMessageW(ctypes.byref(msg))
        user32.UnhookWindowsHookEx(hook)
