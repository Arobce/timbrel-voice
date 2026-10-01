"""Small custom widgets: level meter, parameter slider, state icons."""

from __future__ import annotations

import math

from PySide6.QtCore import QRectF, Qt, Signal
from PySide6.QtGui import QColor, QIcon, QPainter, QPainterPath, QPen, QPixmap
from PySide6.QtWidgets import QHBoxLayout, QLabel, QSizePolicy, QSlider, QWidget

from timbrel.core.params import ParamSpec

ON_COLOR = QColor("#1f9d55")
BYPASS_COLOR = QColor("#8a8f98")
METER_FLOOR_DB = -60.0


class LevelMeter(QWidget):
    """Horizontal peak meter in dB with a falling bar and a short peak hold."""

    DECAY_DB_PER_TICK = 1.5

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._db = METER_FLOOR_DB
        self._hold_db = METER_FLOOR_DB
        self._hold_ticks = 0
        self.setMinimumSize(160, 14)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

    @property
    def level_db(self) -> float:
        return self._db

    def set_peak(self, peak: float) -> None:
        db = 20 * math.log10(peak) if peak > 1e-6 else METER_FLOOR_DB
        db = max(METER_FLOOR_DB, min(0.0, db))
        self._db = db if db > self._db else max(db, self._db - self.DECAY_DB_PER_TICK)
        if db >= self._hold_db:
            self._hold_db, self._hold_ticks = db, 30
        elif self._hold_ticks > 0:
            self._hold_ticks -= 1
        else:
            self._hold_db = max(METER_FLOOR_DB, self._hold_db - self.DECAY_DB_PER_TICK)
        self.update()

    def paintEvent(self, event: object) -> None:  # noqa: N802
        p = QPainter(self)
        r = self.rect()
        p.fillRect(r, self.palette().base())

        def x_for(db: float) -> float:
            return r.width() * (db - METER_FLOOR_DB) / -METER_FLOOR_DB

        width = x_for(self._db)
        for start_db, end_db, color in (
            (METER_FLOOR_DB, -12.0, "#2fb36d"),
            (-12.0, -3.0, "#e0b021"),
            (-3.0, 0.0, "#e0483b"),
        ):
            x0, x1 = x_for(start_db), min(x_for(end_db), width)
            if x1 > x0:
                p.fillRect(QRectF(x0, 0, x1 - x0, r.height()), QColor(color))
        if self._hold_db > METER_FLOOR_DB:
            p.fillRect(QRectF(x_for(self._hold_db) - 1, 0, 2, r.height()), self.palette().text())
        p.setPen(self.palette().mid().color())
        p.drawRect(r.adjusted(0, 0, -1, -1))


class ParamSlider(QWidget):
    """Label + slider + value readout for one effect parameter."""

    STEPS = 1000
    changed = Signal(str, float)

    def __init__(
        self,
        name: str,
        spec: ParamSpec,
        value: float,
        parent: QWidget | None = None,
        label: str | None = None,
    ) -> None:
        super().__init__(parent)
        self.name = name
        self.spec = spec
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        words = name.split("_")
        if words[-1] in ("db", "ms", "hz"):  # the unit is shown in the readout
            words = words[:-1]
        label = QLabel(label or " ".join(words).replace("freq", "frequency"))
        label.setMinimumWidth(96)
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setRange(0, self.STEPS)
        self.slider.setAccessibleName(name)
        self.readout = QLabel()
        self.readout.setMinimumWidth(72)
        self.readout.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        layout.addWidget(label)
        layout.addWidget(self.slider, 1)
        layout.addWidget(self.readout)
        self.set_value(value)
        self.slider.valueChanged.connect(self._on_slider)

    def value(self) -> float:
        span = self.spec.max - self.spec.min
        return self.spec.min + span * self.slider.value() / self.STEPS

    def set_value(self, value: float) -> None:
        span = self.spec.max - self.spec.min
        pos = round((self.spec.clamp(value) - self.spec.min) / span * self.STEPS) if span else 0
        self.slider.blockSignals(True)
        self.slider.setValue(pos)
        self.slider.blockSignals(False)
        self._update_readout()

    def _update_readout(self) -> None:
        value = self.value()
        span = abs(self.spec.max - self.spec.min)
        digits = 0 if span >= 100 else 1 if span >= 5 else 2
        unit = self.spec.unit
        sep = "" if unit.startswith(":") or not unit else " "
        self.readout.setText(f"{value:.{digits}f}{sep}{unit}")

    def _on_slider(self) -> None:
        self._update_readout()
        self.changed.emit(self.name, self.value())


def state_icon(on: bool, size: int = 64) -> QIcon:
    """Tray/window icon: green sound wave when effects are ON, grey with a
    slash when BYPASSED, so the state is visible at a glance (FR13)."""
    pixmap = QPixmap(size, size)
    pixmap.fill(Qt.GlobalColor.transparent)
    p = QPainter(pixmap)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)
    color = ON_COLOR if on else BYPASS_COLOR
    p.setBrush(color)
    p.setPen(Qt.PenStyle.NoPen)
    p.drawEllipse(QRectF(2, 2, size - 4, size - 4))
    pen = QPen(QColor("white"), size / 12)
    pen.setCapStyle(Qt.PenCapStyle.RoundCap)
    p.setPen(pen)
    p.setBrush(Qt.BrushStyle.NoBrush)
    wave = QPainterPath()
    wave.moveTo(size * 0.2, size * 0.5)
    for i in range(1, 61):
        x = 0.2 + 0.6 * i / 60
        y = 0.5 - 0.18 * math.sin(2 * math.pi * 2 * (i / 60))
        wave.lineTo(size * x, size * y)
    p.drawPath(wave)
    if not on:
        p.drawLine(int(size * 0.22), int(size * 0.78), int(size * 0.78), int(size * 0.22))
    p.end()
    return QIcon(pixmap)
