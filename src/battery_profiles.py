"""Battery capability profiles used by the WTA optimizer."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BatteryProfile:
    profile_id: str
    max_range_km: float
    prob_kill: float


_PROFILES: dict[str, BatteryProfile] = {
    "PIL-ALFA": BatteryProfile("PIL-ALFA", max_range_km=15.0, prob_kill=0.90),
    "PIL-BETA": BatteryProfile("PIL-BETA", max_range_km=12.0, prob_kill=0.86),
    "PIL-GAMMA": BatteryProfile("PIL-GAMMA", max_range_km=7.0, prob_kill=0.80),
    "PIL-DELTA": BatteryProfile("PIL-DELTA", max_range_km=5.0, prob_kill=0.76),
}

_DEFAULT_PROFILE = BatteryProfile("PIL-VARSAYILAN", max_range_km=5.0, prob_kill=0.75)


def normalize_battery_name(name: str) -> str:
    """Normalize Turkish display text such as PİL-ALFA to ASCII profile keys."""
    return (
        str(name)
        .strip()
        .upper()
        .replace("İ", "I")
        .replace("ı", "I")
    )


def profile_for_battery(name: str) -> BatteryProfile:
    return _PROFILES.get(normalize_battery_name(name), _DEFAULT_PROFILE)
