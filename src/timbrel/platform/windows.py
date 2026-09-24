"""Windows audio device detection (WASAPI) and VB-Audio Virtual Cable lookup.

The pure functions take device/host-API lists in the shape returned by
``sounddevice.query_devices()`` / ``query_hostapis()`` so they can be tested
without audio hardware.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
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


def find_cable(wasapi: WasapiDevices) -> Device | None:
    """Return VB-Cable's playback endpoint ("CABLE Input ..."), if installed."""
    for device in wasapi.of("output"):
        if device.name.lower().startswith(CABLE_PLAYBACK_NAME.lower()):
            return device
    return None


def resolve_device(spec: str | None, direction: Direction, wasapi: WasapiDevices) -> Device:
    """Pick a WASAPI device from a CLI spec: an index, a name fragment, or None.

    With no spec, the output defaults to VB-Cable and the input to the system
    default microphone.
    """
    candidates = wasapi.of(direction)

    if spec is None:
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
        if device is None or not device.supports("input"):
            raise DeviceError("No default input device; choose one with --input.")
        return device

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


def query_wasapi_devices() -> WasapiDevices:
    return wasapi_devices(sd.query_devices(), sd.query_hostapis())


def stream_settings() -> sd.WasapiSettings:
    """Shared-mode WASAPI, letting Windows convert rate/channels when the
    device's mix format differs from our 48 kHz mono/stereo stream."""
    return sd.WasapiSettings(auto_convert=True)
