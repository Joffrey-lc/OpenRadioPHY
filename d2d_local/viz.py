"""Plotting helpers for PHY diagnostics."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image, ImageDraw

from .qam import ordered_constellation


def _map_points_to_pixels(points: np.ndarray, axis_limit: float, size: int, margin: int) -> np.ndarray:
    scale = (size - 2 * margin - 1) / (2.0 * axis_limit)
    x = margin + (points.real + axis_limit) * scale
    y = margin + (axis_limit - points.imag) * scale
    return np.stack([x, y], axis=1)


def save_constellation_plot(
    symbols: np.ndarray,
    path: str | Path,
    *,
    bits_per_symbol: int,
    title: str | None = None,
    max_points: int = 6000,
) -> Path:
    """Save a constellation plot."""
    path = Path(path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)

    samples = np.asarray(symbols, dtype=np.complex128).reshape(-1)
    if samples.size == 0:
        raise ValueError("symbols is empty")
    if max_points > 0 and samples.size > max_points:
        step = int(np.ceil(samples.size / max_points))
        samples = samples[::step]

    ideal_points, _ = ordered_constellation(bits_per_symbol)
    ideal_peak = float(np.max(np.abs(ideal_points))) if ideal_points.size else 1.0
    sample_peak = float(np.max(np.abs(samples))) if samples.size else 1.0
    axis_limit = 1.2 * max(ideal_peak, sample_peak, 1.0)

    size = 900
    margin = 70
    image = Image.new("RGBA", (size, size), (255, 255, 255, 255))
    draw = ImageDraw.Draw(image, "RGBA")

    grid_color = (220, 220, 220, 255)
    axis_color = (140, 140, 140, 255)
    sample_color = (31, 119, 180, 90)
    ideal_color = (214, 39, 40, 255)
    text_color = (40, 40, 40, 255)

    for frac in (-1.0, -0.5, 0.5, 1.0):
        x = margin + (frac + 1.0) * (size - 2 * margin - 1) / 2.0
        y = margin + (1.0 - (frac + 1.0) / 2.0) * (size - 2 * margin - 1)
        draw.line((x, margin, x, size - margin), fill=grid_color, width=1)
        draw.line((margin, y, size - margin, y), fill=grid_color, width=1)

    center = _map_points_to_pixels(np.asarray([0.0 + 0.0j]), axis_limit, size, margin)[0]
    draw.line((margin, center[1], size - margin, center[1]), fill=axis_color, width=2)
    draw.line((center[0], margin, center[0], size - margin), fill=axis_color, width=2)

    sample_pixels = _map_points_to_pixels(samples, axis_limit, size, margin)
    for x, y in sample_pixels:
        draw.ellipse((x - 2, y - 2, x + 2, y + 2), fill=sample_color)

    ideal_pixels = _map_points_to_pixels(ideal_points, axis_limit, size, margin)
    for x, y in ideal_pixels:
        draw.line((x - 7, y - 7, x + 7, y + 7), fill=ideal_color, width=2)
        draw.line((x - 7, y + 7, x + 7, y - 7), fill=ideal_color, width=2)

    label = title or f"RX Constellation ({bits_per_symbol} bits/symbol)"
    draw.text((margin, 18), label, fill=text_color)
    draw.text((margin, size - 32), "I", fill=text_color)
    draw.text((18, margin), "Q", fill=text_color)

    image.convert("RGB").save(path)
    return path


def save_correlation_plot(
    metric: np.ndarray,
    path: str | Path,
    *,
    threshold: float | None = None,
    starts: Iterable[int] = (),
    title: str | None = None,
    max_points: int = 4000,
) -> Path:
    """Save a matched-filter plot."""
    path = Path(path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)

    values = np.asarray(metric, dtype=np.float64).reshape(-1)
    if values.size == 0:
        raise ValueError("metric is empty")

    sample_indices = np.arange(values.size, dtype=np.int64)
    if max_points > 0 and values.size > max_points:
        step = int(np.ceil(values.size / max_points))
        values = values[::step]
        sample_indices = sample_indices[::step]
    else:
        step = 1

    finite = np.isfinite(values)
    if not np.any(finite):
        raise ValueError("metric contains no finite values")
    values = np.where(finite, values, 0.0)

    y_max = float(np.max(values))
    y_min = float(np.min(values))
    if threshold is not None:
        y_max = max(y_max, float(threshold))
        y_min = min(y_min, 0.0)
    if y_max <= y_min:
        y_max = y_min + 1.0

    width = 1200
    height = 520
    margin_left = 72
    margin_right = 28
    margin_top = 44
    margin_bottom = 54
    plot_w = width - margin_left - margin_right
    plot_h = height - margin_top - margin_bottom

    image = Image.new("RGBA", (width, height), (255, 255, 255, 255))
    draw = ImageDraw.Draw(image, "RGBA")

    grid_color = (225, 225, 225, 255)
    axis_color = (120, 120, 120, 255)
    line_color = (31, 119, 180, 255)
    threshold_color = (214, 39, 40, 255)
    start_color = (44, 160, 44, 180)
    text_color = (40, 40, 40, 255)

    for frac in (0.0, 0.25, 0.5, 0.75, 1.0):
        y = margin_top + (1.0 - frac) * plot_h
        draw.line((margin_left, y, width - margin_right, y), fill=grid_color, width=1)

    draw.line(
        (margin_left, margin_top, margin_left, height - margin_bottom),
        fill=axis_color,
        width=2,
    )
    draw.line(
        (margin_left, height - margin_bottom, width - margin_right, height - margin_bottom),
        fill=axis_color,
        width=2,
    )

    x_norm = np.linspace(0.0, 1.0, values.size)
    x_pix = margin_left + x_norm * plot_w
    y_pix = margin_top + (1.0 - (values - y_min) / (y_max - y_min)) * plot_h
    polyline = [(float(x), float(y)) for x, y in np.stack([x_pix, y_pix], axis=1)]
    if len(polyline) >= 2:
        draw.line(polyline, fill=line_color, width=2)

    if threshold is not None:
        threshold_y = margin_top + (1.0 - (float(threshold) - y_min) / (y_max - y_min)) * plot_h
        draw.line(
            (margin_left, threshold_y, width - margin_right, threshold_y),
            fill=threshold_color,
            width=2,
        )

    starts_arr = np.asarray(tuple(starts), dtype=np.int64).reshape(-1)
    if starts_arr.size:
        starts_arr = starts_arr[(starts_arr >= 0) & (starts_arr < sample_indices[-1] + step)]
        for start in starts_arr:
            x = margin_left + (float(start) / max(float(sample_indices[-1]), 1.0)) * plot_w
            draw.line((x, margin_top, x, height - margin_bottom), fill=start_color, width=1)

    label = title or "Matched Filter Metric"
    draw.text((margin_left, 12), label, fill=text_color)
    draw.text((margin_left, height - 28), f"index (downsample step={step})", fill=text_color)
    draw.text((12, margin_top), "metric", fill=text_color)
    draw.text((margin_left, margin_top - 18), f"max={y_max:.3g}", fill=text_color)
    draw.text((margin_left, height - margin_bottom + 8), f"min={y_min:.3g}", fill=text_color)
    if threshold is not None:
        draw.text((width - 220, margin_top + 8), f"threshold={threshold:.3g}", fill=threshold_color)
    if starts_arr.size:
        draw.text((width - 220, margin_top + 28), f"starts={starts_arr.size}", fill=start_color)

    image.convert("RGB").save(path)
    return path

def save_papr_ccdf_plot(
    power_ratio_db: np.ndarray,
    path: str | Path,
    *,
    title: str | None = None,
    max_points: int = 4000,
) -> Path:
    """Save a PAPR CCDF plot."""
    path = Path(path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)

    values = np.asarray(power_ratio_db, dtype=np.float64).reshape(-1)
    values = values[np.isfinite(values)]
    if values.size == 0:
        raise ValueError("power_ratio_db contains no finite values")
    values = np.sort(values)
    n = int(values.size)
    ccdf = 1.0 - (np.arange(n, dtype=np.float64) + 1.0) / float(n)
    ccdf = np.maximum(ccdf, 1.0 / float(n))

    if max_points > 0 and n > max_points:
        idx = np.unique(np.linspace(0, n - 1, max_points).astype(np.int64))
        values = values[idx]
        ccdf = ccdf[idx]

    x_min = min(0.0, float(np.min(values)))
    x_max = max(1.0, float(np.max(values)))
    if x_max <= x_min:
        x_max = x_min + 1.0
    y_log = np.log10(ccdf)
    y_min = min(-4.0, float(np.floor(np.min(y_log))))
    y_max = 0.0

    width = 1200
    height = 520
    margin_left = 82
    margin_right = 28
    margin_top = 44
    margin_bottom = 58
    plot_w = width - margin_left - margin_right
    plot_h = height - margin_top - margin_bottom

    image = Image.new("RGBA", (width, height), (255, 255, 255, 255))
    draw = ImageDraw.Draw(image, "RGBA")

    grid_color = (225, 225, 225, 255)
    axis_color = (120, 120, 120, 255)
    line_color = (31, 119, 180, 255)
    text_color = (40, 40, 40, 255)

    for frac in (0.0, 0.25, 0.5, 0.75, 1.0):
        y = margin_top + frac * plot_h
        draw.line((margin_left, y, width - margin_right, y), fill=grid_color, width=1)
    for frac in (0.0, 0.25, 0.5, 0.75, 1.0):
        x = margin_left + frac * plot_w
        draw.line((x, margin_top, x, height - margin_bottom), fill=grid_color, width=1)

    draw.line((margin_left, margin_top, margin_left, height - margin_bottom), fill=axis_color, width=2)
    draw.line((margin_left, height - margin_bottom, width - margin_right, height - margin_bottom), fill=axis_color, width=2)

    x_pix = margin_left + (values - x_min) / (x_max - x_min) * plot_w
    y_pix = margin_top + (1.0 - (y_log - y_min) / (y_max - y_min)) * plot_h
    polyline = [(float(x), float(y)) for x, y in np.stack([x_pix, y_pix], axis=1)]
    if len(polyline) >= 2:
        draw.line(polyline, fill=line_color, width=2)

    label = title or "TX PAPR CCDF"
    draw.text((margin_left, 12), label, fill=text_color)
    draw.text((margin_left, height - 28), "power / average power (dB)", fill=text_color)
    draw.text((12, margin_top), "CCDF", fill=text_color)
    draw.text((margin_left, margin_top - 18), f"x=[{x_min:.1f},{x_max:.1f}] dB", fill=text_color)
    draw.text((width - 230, margin_top + 8), f"samples={n}", fill=text_color)
    for exp in range(0, int(abs(y_min)) + 1):
        y_val = -float(exp)
        y = margin_top + (1.0 - (y_val - y_min) / (y_max - y_min)) * plot_h
        if margin_top <= y <= height - margin_bottom:
            draw.text((margin_left - 58, y - 7), f"1e-{exp}", fill=text_color)

    image.convert("RGB").save(path)
    return path
