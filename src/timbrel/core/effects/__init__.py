"""Built-in effects. Chain order: noise gate -> selected effects -> limiter."""

from timbrel.core.effects.base import Effect
from timbrel.core.effects.dynamics import Limiter, NoiseGate
from timbrel.core.effects.pitch import PitchShift
from timbrel.core.effects.radio import Radio
from timbrel.core.effects.robot import Robot

# Effects that can be placed between the gate and the limiter.
EFFECTS: dict[str, type[Effect]] = {cls.name: cls for cls in (PitchShift, Robot, Radio)}

__all__ = ["EFFECTS", "Effect", "Limiter", "NoiseGate", "PitchShift", "Radio", "Robot"]
