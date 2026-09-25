"""Application controller: ties settings, presets, the effect chain and the
audio engine together. Has no Qt dependency, so the UI stays thin and the
logic is testable without audio hardware.

All methods run on the UI (main) thread. The audio threads only ever see the
engine and chain, which publish changes by reference swap.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import sounddevice as sd

from timbrel.core.chain import EffectChain
from timbrel.core.effects import Effect, NoiseGate
from timbrel.core.engine import DEFAULT_SAMPLE_RATE, Engine, EngineConfig
from timbrel.platform import windows
from timbrel.presets import Preset, PresetLibrary, preset_from_effects
from timbrel.settings import Settings


def _is_exclusive(settings: sd.WasapiSettings | None) -> bool:
    return settings is not None and bool(settings._streaminfo.flags & sd._lib.paWinWasapiExclusive)


def start_engine(
    config: EngineConfig, chain: EffectChain, factory: Callable[..., Engine] = Engine
) -> tuple[Engine, EngineConfig]:
    """Start the engine, falling back to shared mode on each side that refuses
    exclusive access (device busy or format unsupported)."""
    shared = windows.shared_settings()
    attempts = [
        config,
        dataclasses.replace(config, input_settings=shared),
        dataclasses.replace(config, input_settings=shared, output_settings=shared),
    ]
    for attempt in attempts[:-1]:
        engine = factory(attempt, chain)
        try:
            engine.start()
            return engine, attempt
        except sd.PortAudioError:
            pass
    engine = factory(attempts[-1], chain)
    engine.start()
    return engine, attempts[-1]


@dataclass(frozen=True)
class Status:
    running: bool
    error: str | None
    cable_found: bool
    input_name: str | None
    output_name: str | None
    mic_exclusive: bool
    latency_ms: float | None
    xruns: int
    input_peak: float
    output_peak: float


class Controller:
    def __init__(
        self,
        settings: Settings,
        settings_path: Path | None,
        library: PresetLibrary,
        query_devices: Callable[[], windows.WasapiDevices] = windows.query_wasapi_devices,
        engine_factory: Callable[..., Engine] = Engine,
        autostart: Callable[[bool], None] | None = None,
    ) -> None:
        self.settings = settings
        self._settings_path = settings_path
        self.library = library
        self._query_devices = query_devices
        self._engine_factory = engine_factory
        self._autostart = autostart or (lambda enabled: windows.set_autostart(enabled))
        self.engine: Engine | None = None
        self.error: str | None = None
        self.devices = query_devices()
        self._mic: windows.Device | None = None
        self._out: windows.Device | None = None
        self._mic_exclusive = False
        self.modified = False

        preset = library.get(settings.preset) or library.all()[0]
        self.preset: Preset = preset
        self.settings.preset = preset.name
        self.chain = self._new_chain(preset)
        self.chain.set_bypass(settings.bypass)
        self.listeners: list[Callable[[], None]] = []

    # --- helpers ------------------------------------------------------------

    def _new_chain(self, preset: Preset) -> EffectChain:
        bs = self.settings.block_size
        chain = EffectChain(DEFAULT_SAMPLE_RATE, bs, preset.build_effects(DEFAULT_SAMPLE_RATE, bs))
        chain.gate.set_params(**preset.gate_params())
        return chain

    def _save(self) -> None:
        if self._settings_path is not None:
            self.settings.save(self._settings_path)

    def _notify(self) -> None:
        for listener in list(self.listeners):
            listener()

    # --- devices and engine -----------------------------------------------

    def input_devices(self) -> list[windows.Device]:
        return [d for d in self.devices.of("input") if not windows.is_virtual_cable(d)]

    def output_devices(self) -> list[windows.Device]:
        return self.devices.of("output")

    def monitor_devices(self) -> list[windows.Device]:
        return [d for d in self.devices.of("output") if not windows.is_virtual_cable(d)]

    @property
    def cable(self) -> windows.Device | None:
        return windows.find_cable(self.devices)

    def refresh_devices(self) -> None:
        self.devices = self._query_devices()

    def _resolve(self) -> tuple[windows.Device, windows.Device | None, windows.Device | None]:
        s = self.settings
        mic = windows.find_by_name(s.input_device, "input", self.devices)
        if mic is None or windows.is_virtual_cable(mic):
            mic = windows.default_device("input", self.devices)
        out = windows.find_by_name(s.output_device, "output", self.devices) or self.cable
        monitor = None
        if s.monitor_device:
            monitor = windows.find_by_name(s.monitor_device, "output", self.devices)
        if monitor is None and out is None:
            # Monitor-only use without VB-Cable: fall back to the default speakers.
            default = self.devices.default("output")
            monitor = self.devices.by_index(default) if default is not None else None
        if out is not None:
            windows.check_route(mic, out)
        return mic, out, monitor

    def start(self) -> bool:
        """(Re)start audio with the current settings. Returns success; on
        failure ``error`` says why and the app keeps running without audio."""
        self.stop()
        try:
            mic, out, monitor = self._resolve()
            cfg = EngineConfig(
                input_device=mic.index,
                output_device=out.index if out else None,
                block_size=self.settings.block_size,
                output_channels=min(2, out.max_output_channels) if out else 2,
                input_settings=windows.stream_settings(mic, exclusive=self.settings.exclusive_mic),
                output_settings=windows.stream_settings(out) if out else None,
                monitor_device=monitor.index if monitor else None,
                monitor_channels=min(2, monitor.max_output_channels) if monitor else 2,
                monitor_settings=windows.shared_settings() if monitor else None,
            )
            self.engine, used = start_engine(cfg, self.chain, self._engine_factory)
        except (windows.DeviceError, sd.PortAudioError, ValueError) as exc:
            self.engine = None
            self.error = str(exc)
            self._notify()
            return False
        self.engine.set_monitor(self.settings.monitor_enabled or out is None)
        self._mic, self._out = mic, out
        self._mic_exclusive = _is_exclusive(used.input_settings)
        self.error = None
        self._notify()
        return True

    def stop(self) -> None:
        if self.engine is not None:
            self.engine.stop()
            self.engine = None

    def set_devices(
        self, input_name: str | None, output_name: str | None, monitor_name: str | None
    ) -> None:
        s = self.settings
        s.input_device, s.output_device, s.monitor_device = input_name, output_name, monitor_name
        self._save()
        self.start()

    def set_monitor_enabled(self, enabled: bool) -> None:
        self.settings.monitor_enabled = enabled
        self._save()
        if self.engine is not None:
            if enabled and not self.engine.has_monitor:
                self.start()  # the monitor stream is opened at start
            else:
                self.engine.set_monitor(enabled or self._out is None)
        self._notify()

    def set_exclusive_mic(self, enabled: bool) -> None:
        self.settings.exclusive_mic = enabled
        self._save()
        self.start()

    def set_block_size(self, block_size: int) -> None:
        self.settings.block_size = block_size
        self._save()
        # Effects preallocate per block size, so rebuild the chain.
        live = self.snapshot("live")
        bypass = self.chain.bypassed
        self.chain = self._new_chain(live)
        self.chain.set_bypass(bypass)
        self.start()

    def status(self) -> Status:
        engine = self.engine
        running = engine is not None and engine.running
        stats = engine.stats if engine is not None else None
        return Status(
            running=running,
            error=self.error,
            cable_found=self.cable is not None,
            input_name=self._mic.name if self._mic and running else None,
            output_name=self._out.name if self._out and running else None,
            mic_exclusive=self._mic_exclusive,
            latency_ms=engine.latency_ms if running and engine is not None else None,
            xruns=stats.xruns if stats else 0,
            input_peak=stats.input_peak if stats and running else 0.0,
            output_peak=stats.output_peak if stats and running else 0.0,
        )

    # --- presets and effects -------------------------------------------------

    @property
    def gate(self) -> NoiseGate:
        return self.chain.gate

    @property
    def effects(self) -> tuple[Effect, ...]:
        return self.chain.effects

    def select_preset(self, name: str) -> None:
        preset = self.library.get(name)
        if preset is None:
            return
        bs = self.settings.block_size
        self.chain.set_effects(preset.build_effects(DEFAULT_SAMPLE_RATE, bs))
        self.chain.gate.set_params(**preset.gate_params())
        self.preset = preset
        self.modified = False
        self.settings.preset = preset.name
        self._save()
        self._notify()

    def step_preset(self, step: int) -> None:
        names = self.library.names()
        i = names.index(self.preset.name) if self.preset.name in names else 0
        self.select_preset(names[(i + step) % len(names)])

    def set_param(self, target: int | None, name: str, value: float) -> None:
        """Change a live parameter: target None = gate, else effect index."""
        effect = self.gate if target is None else self.effects[target]
        effect.set_params(**{name: value})
        self.modified = True

    def snapshot(self, name: str) -> Preset:
        return preset_from_effects(name, self.gate, self.effects, self.preset.description)

    def save_preset_as(self, name: str, overwrite: bool = False) -> Preset:
        saved = self.library.save(self.snapshot(name), overwrite=overwrite)
        self.preset = saved
        self.modified = False
        self.settings.preset = saved.name
        self._save()
        self._notify()
        return saved

    def rename_preset(self, old: str, new: str) -> None:
        renamed = self.library.rename(old, new)
        if self.preset.name.lower() == old.lower():
            self.preset = renamed
            self.settings.preset = renamed.name
            self._save()
        self._notify()

    def delete_preset(self, name: str) -> None:
        self.library.delete(name)
        if self.preset.name.lower() == name.lower():
            self.select_preset(self.library.all()[0].name)
        else:
            self._notify()

    # --- bypass and other settings ---------------------------------------------

    @property
    def bypassed(self) -> bool:
        return self.chain.bypassed

    def set_bypass(self, bypass: bool) -> None:
        self.chain.set_bypass(bypass)
        self.settings.bypass = bypass
        self._save()
        self._notify()

    def toggle_bypass(self) -> None:
        self.set_bypass(not self.bypassed)

    def set_start_with_windows(self, enabled: bool) -> None:
        self._autostart(enabled)
        self.settings.start_with_windows = enabled
        self._save()

    def mark_first_run_done(self) -> None:
        self.settings.first_run_done = True
        self._save()
