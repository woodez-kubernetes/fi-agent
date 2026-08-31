"""Inline SVG sparklines.

Drawn as raw SVG rather than with a charting library so the report stays a single
self-contained file with no scripts and no external assets.
"""

from __future__ import annotations

from xml.sax.saxutils import escape

WIDTH = 160
HEIGHT = 40
PAD = 3


def sparkline(values: list[float], up: bool | None = None, label: str = "") -> str:
    """Render a trailing-price sparkline. Returns '' when there is nothing to draw."""
    points = [v for v in values if v is not None]
    if len(points) < 2:
        return ""

    low = min(points)
    high = max(points)
    span = high - low
    # A perfectly flat series would divide by zero; draw it down the middle instead.
    if span <= 0:
        span = 1.0
        normalise = lambda _v: HEIGHT / 2  # noqa: E731
    else:
        def normalise(value: float) -> float:
            fraction = (value - low) / span
            return HEIGHT - PAD - fraction * (HEIGHT - 2 * PAD)

    step = (WIDTH - 2 * PAD) / (len(points) - 1)
    coords = [(PAD + i * step, normalise(v)) for i, v in enumerate(points)]
    path = " ".join(f"{x:.1f},{y:.1f}" for x, y in coords)

    rising = points[-1] >= points[0] if up is None else up
    stroke = "var(--up)" if rising else "var(--down)"
    fill_id = f"g{'u' if rising else 'd'}"

    area = f"{PAD},{HEIGHT} {path} {coords[-1][0]:.1f},{HEIGHT}"
    last_x, last_y = coords[-1]
    title = f"<title>{escape(label)}</title>" if label else ""

    return (
        f'<svg class="spark" viewBox="0 0 {WIDTH} {HEIGHT}" width="{WIDTH}" '
        f'height="{HEIGHT}" role="img" aria-label="{escape(label or "price trend")}">'
        f"{title}"
        f'<defs><linearGradient id="{fill_id}" x1="0" x2="0" y1="0" y2="1">'
        f'<stop offset="0%" stop-color="{stroke}" stop-opacity="0.28"/>'
        f'<stop offset="100%" stop-color="{stroke}" stop-opacity="0"/>'
        f"</linearGradient></defs>"
        f'<polygon points="{area}" fill="url(#{fill_id})"/>'
        f'<polyline points="{path}" fill="none" stroke="{stroke}" stroke-width="1.6" '
        f'stroke-linejoin="round" stroke-linecap="round"/>'
        f'<circle cx="{last_x:.1f}" cy="{last_y:.1f}" r="2.4" fill="{stroke}"/>'
        f"</svg>"
    )
