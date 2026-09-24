import pytest
import sounddevice as sd

from conftest import DEVICES, HOSTAPIS
from timbrel.platform.windows import (
    DeviceError,
    check_route,
    find_cable,
    resolve_device,
    stream_settings,
    wasapi_devices,
)


def test_only_wasapi_devices_are_listed(wasapi):
    assert [d.index for d in wasapi.devices] == [2, 3, 4, 5, 6, 7]
    assert [d.index for d in wasapi.of("input")] == [2, 3, 6]
    assert [d.index for d in wasapi.of("output")] == [4, 5, 7]
    assert wasapi.default_input == 3
    assert wasapi.default_output == 4


def test_missing_wasapi_raises():
    with pytest.raises(DeviceError, match="WASAPI"):
        wasapi_devices(DEVICES, HOSTAPIS[:1])


def test_find_cable_picks_wasapi_playback_endpoint(wasapi):
    cable = find_cable(wasapi)
    assert cable is not None
    assert cable.index == 5


def test_default_output_is_cable(wasapi):
    assert resolve_device(None, "output", wasapi).index == 5


def test_default_output_without_cable_explains_install(wasapi_no_cable):
    with pytest.raises(DeviceError, match="vb-audio.com/Cable"):
        resolve_device(None, "output", wasapi_no_cable)


def test_default_input_is_system_default(wasapi):
    assert resolve_device(None, "input", wasapi).index == 3


def test_default_input_skips_cable_when_it_is_system_default():
    hostapis = [HOSTAPIS[0], {**HOSTAPIS[1], "default_input_device": 6}]
    wasapi = wasapi_devices(DEVICES, hostapis)
    assert resolve_device(None, "input", wasapi).index == 2


def test_cable_to_cable_route_is_refused(wasapi):
    mic = resolve_device("cable output", "input", wasapi)
    out = resolve_device(None, "output", wasapi)
    with pytest.raises(DeviceError, match="feedback loop"):
        check_route(mic, out)


def test_mic_to_cable_route_is_allowed(wasapi):
    check_route(resolve_device(None, "input", wasapi), resolve_device(None, "output", wasapi))


def test_resolve_by_index(wasapi):
    assert resolve_device("2", "input", wasapi).index == 2
    assert resolve_device(" 7 ", "output", wasapi).index == 7


@pytest.mark.parametrize(
    ("spec", "direction"),
    [("0", "input"), ("4", "input"), ("3", "output"), ("99", "output")],
)
def test_resolve_by_index_rejects_wrong_device(wasapi, spec, direction):
    with pytest.raises(DeviceError, match="--list-devices"):
        resolve_device(spec, direction, wasapi)


def test_resolve_by_name_fragment_is_case_insensitive(wasapi):
    assert resolve_device("webcam", "input", wasapi).index == 2
    assert resolve_device("REALTEK", "output", wasapi).index == 4


def test_resolve_by_name_filters_by_direction(wasapi):
    # "USB Mic" names both an input and an output device.
    assert resolve_device("usb mic", "input", wasapi).index == 3
    assert resolve_device("usb mic", "output", wasapi).index == 7


def test_ambiguous_name_lists_matches(wasapi):
    with pytest.raises(DeviceError, match=r"\[2\].*\[3\]"):
        resolve_device("microphone", "input", wasapi)


def test_exact_name_wins_over_fragment(wasapi):
    assert resolve_device("microphone (usb mic)", "input", wasapi).index == 3


def test_unknown_name_raises(wasapi):
    with pytest.raises(DeviceError, match="No WASAPI output"):
        resolve_device("nonexistent", "output", wasapi)


def _is_exclusive(settings) -> bool:
    return bool(settings._streaminfo.flags & sd._lib.paWinWasapiExclusive)


def test_only_cable_playback_is_exclusive(wasapi):
    cable = resolve_device(None, "output", wasapi)
    speakers = resolve_device("realtek", "output", wasapi)
    mic = resolve_device(None, "input", wasapi)
    cable_capture = resolve_device("cable output", "input", wasapi)
    assert _is_exclusive(stream_settings(cable))
    assert not _is_exclusive(stream_settings(speakers))
    assert not _is_exclusive(stream_settings(mic))
    assert not _is_exclusive(stream_settings(cable_capture))
