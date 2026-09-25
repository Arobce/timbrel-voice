"""Built-in effects. Chain order: noise gate -> effects -> limiter."""

from timbrel.core.effects.base import Effect
from timbrel.core.effects.dynamics import Compressor, Limiter, NoiseGate
from timbrel.core.effects.echo import Echo
from timbrel.core.effects.eq import Eq
from timbrel.core.effects.pitch import PitchShift
from timbrel.core.effects.radio import Radio
from timbrel.core.effects.robot import Robot

# Effects that can be placed between the gate and the limiter.
EFFECTS: dict[str, type[Effect]] = {
    cls.name: cls for cls in (PitchShift, Robot, Radio, Echo, Eq, Compressor)
}

__all__ = [
    "EFFECTS",
    "Compressor",
    "Echo",
    "Effect",
    "Eq",
    "Limiter",
    "NoiseGate",
    "PitchShift",
    "Radio",
    "Robot",
]
