import time
from dataclasses import dataclass
from enum import Enum

from motion import Motion


class Axis(str, Enum):
    VX = "vx"
    VY = "vy"


class Sign(str, Enum):
    POSITIVE = "positive"
    NEGATIVE = "negative"


@dataclass(frozen=True)
class GestureRule:
    axis: Axis
    sign: Sign
    min_speed: float
    cooldown_ms: int
    hold_frames: int


class GestureEngine:
    """
    Declarative gesture engine that classifies Motion into discrete gesture events.

    Design principles:
    - Rules are data-driven (GESTURES dict).
    - Motion is treated as a continuous signal (provided by MotionEstimator).
    - A lightweight temporal mechanism ("streak") is used to reduce jitter and
      ensure gesture stability without a full state machine.

    Gesture detection pipeline:
        1. Match motion against gesture rules (axis, sign, thresholds).
        2. Track short-lived streaks per (hand, gesture) to ensure persistence.
        3. Emit gesture only if:
            - streak >= required frames
            - cooldown has elapsed

    Notes:
    - This engine is stateless with respect to long gesture lifecycles.
    - It does NOT model "gesture start/continue/end", only emits discrete events.
    - Consumers are responsible for interpreting gestures (e.g. scroll, navigate).

    Attributes:
        GESTURES (dict): Declarative gesture definitions.
        state (dict): Last fired timestamps per (hand, gesture).
        streaks (dict): Short-lived match counters per (hand, gesture).
    """

    GESTURES: dict[str, GestureRule] = {
        "swipe_up": GestureRule(
            axis=Axis.VY, sign=Sign.NEGATIVE, min_speed=0.02, cooldown_ms=120, hold_frames=2
        ),
        "swipe_down": GestureRule(
            axis=Axis.VY, sign=Sign.POSITIVE, min_speed=0.02, cooldown_ms=120, hold_frames=2
        ),
        "swipe_left": GestureRule(
            axis=Axis.VX, sign=Sign.NEGATIVE, min_speed=0.02, cooldown_ms=120, hold_frames=2
        ),
        "swipe_right": GestureRule(
            axis=Axis.VX, sign=Sign.POSITIVE, min_speed=0.02, cooldown_ms=120, hold_frames=2
        ),
    }

    def __init__(self):
        self.state: dict[tuple, float] = {}  # (hand, gesture) -> last_fire_time
        self.streaks: dict[tuple, int] = {}  # (hand, gesture) -> consecutive matches

    def process(self, motion: Motion | None, hand_label: str) -> str | None:
        if motion is None:
            return None

        now = time.time() * 1000

        best = None
        best_score = 0.0

        for name, rule in self.GESTURES.items():
            key = (hand_label, name)

            if self._match(motion, rule):
                self.streaks[key] = self.streaks.get(key, 0) + 1

                axis_value = abs(motion.vy if rule.axis == Axis.VY else motion.vx)

                if (
                    self.streaks[key] >= rule.hold_frames
                    and axis_value > best_score
                    and self._can_fire(key, rule, now)
                ):
                    best = name
                    best_score = axis_value
            else:
                self.streaks[key] = 0

        if best:
            self._mark_fired((hand_label, best), now)
            return best

        return None

    def _match(self, motion: Motion, rule: GestureRule) -> bool:
        axis_value = motion.vy if rule.axis == Axis.VY else motion.vx
        other_axis = motion.vx if rule.axis == Axis.VY else motion.vy

        if rule.sign == Sign.POSITIVE and axis_value <= 0:
            return False
        if rule.sign == Sign.NEGATIVE and axis_value >= 0:
            return False

        # dominance check (prevent diagonal noise)
        if abs(axis_value) < abs(other_axis):
            return False

        if motion.speed < rule.min_speed:
            return False

        return True

    def _can_fire(self, key: tuple, rule: GestureRule, now: float) -> bool:
        return (now - self.state.get(key, 0)) > rule.cooldown_ms

    def _mark_fired(self, key: tuple, now: float) -> None:
        self.state[key] = now
        self.streaks[key] = 0
