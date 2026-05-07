"""Coordinate helpers for Celik Kubbe radar display conversions."""

from __future__ import annotations

import math

DISPLAY_RADIUS_MAX = 0.95


def clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))


def km_to_display_radius(
    range_km: float,
    max_range_km: float,
    display_radius_max: float = DISPLAY_RADIUS_MAX,
) -> float:
    """Convert a physical radar range in km to the normalized radar scope radius."""
    if max_range_km <= 0:
        return 0.0
    return clamp((range_km / max_range_km) * display_radius_max, 0.0, display_radius_max)


def display_radius_to_km(
    display_radius: float,
    max_range_km: float,
    display_radius_max: float = DISPLAY_RADIUS_MAX,
) -> float:
    """Convert a normalized radar scope radius back to physical kilometers."""
    if max_range_km <= 0 or display_radius_max <= 0:
        return 0.0
    radius = clamp(display_radius, 0.0, display_radius_max)
    return (radius / display_radius_max) * max_range_km


def polar_to_display_xy(range_km: float, bearing_deg: float, max_range_km: float) -> tuple[float, float]:
    """Return normalized x/y radar display coordinates for a physical polar position."""
    radius = km_to_display_radius(range_km, max_range_km)
    bearing_rad = math.radians(bearing_deg)
    return radius * math.cos(bearing_rad), radius * math.sin(bearing_rad)
