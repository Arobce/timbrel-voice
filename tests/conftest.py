import pytest

from timbrel.platform.windows import WasapiDevices, wasapi_devices

HOSTAPIS = [
    {"name": "MME", "default_input_device": 0, "default_output_device": 1},
    {"name": "Windows WASAPI", "default_input_device": 3, "default_output_device": 4},
]


def _dev(name: str, hostapi: int, ins: int, outs: int) -> dict:
    return {
        "name": name,
        "hostapi": hostapi,
        "max_input_channels": ins,
        "max_output_channels": outs,
        "default_samplerate": 48000.0,
    }


DEVICES = [
    _dev("Microphone (USB Mic)", 0, 2, 0),  # 0: MME, ignored
    _dev("CABLE Input (VB-Audio Virtual Cable)", 0, 0, 2),  # 1: MME, ignored
    _dev("Microphone (Webcam)", 1, 2, 0),  # 2
    _dev("Microphone (USB Mic)", 1, 2, 0),  # 3: default input
    _dev("Speakers (Realtek)", 1, 0, 2),  # 4: default output
    _dev("CABLE Input (VB-Audio Virtual Cable)", 1, 0, 8),  # 5
    _dev("CABLE Output (VB-Audio Virtual Cable)", 1, 2, 0),  # 6
    _dev("Headphones (USB Mic)", 1, 0, 2),  # 7
]

DEVICES_WITHOUT_CABLE = [d for d in DEVICES if "CABLE" not in d["name"]]


@pytest.fixture
def wasapi() -> WasapiDevices:
    return wasapi_devices(DEVICES, HOSTAPIS)


@pytest.fixture
def wasapi_no_cable() -> WasapiDevices:
    return wasapi_devices(DEVICES_WITHOUT_CABLE, HOSTAPIS)
