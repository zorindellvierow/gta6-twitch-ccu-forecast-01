"""gta6-twitch-ccu-forecast-01 entry (Python 3.13, standard library only).

Forecasts each hour 2..24 of a big console game's Twitch launch day from the
case's own comparables: for every past launch in the case, take its release-day
hourly viewers and its mean viewers in the 24 hours before release day, then
fit, separately for each forecast hour h, a two-regressor log-linear model

    log viewers(h) = a_h + b1_h * log(h1) + b2_h * log(pre)

where h1 is that launch's hour-1 print and pre its pre-release mean. The
median is blended 50/50 in log space with the plain median day-shape ratio
(median viewers(h)/viewers(1) across the training launches), which hedges the
regression against a few huge launches dominating the evening hours. The
quantiles are that blended median times exp(z_tau * sd_h), sd_h the residual
standard deviation of that hour's fit (floored at 0.25 in log units).

Why this shape: across past launches the evening peak is nearly independent
of the hour-1 print (a launch whose hour 1 is small because most timezones
had not unlocked yet still peaks at a similar absolute level), so a plain
ratio-to-hour-1 prior overstates the evening for big hour-1 prints and
understates it for small ones. The hour-1 print does carry information early
(b1 is near 1 at h=2 and falls toward ~0.4-0.5 by the evening), and the
pre-release mean carries the game's own hype level (b2 grows to ~0.2 in the
evening). Launches with a tiny hour-1 print (below 10,000 viewers) follow a
different regime -- their ratios explode by 100-2000x -- so when the case's
own hour-1 print is at least 10,000 the fit trains only on comparables with
hour-1 prints of at least 10,000; otherwise it trains on all comparables.
Log hour-1 and log pre-release inputs are clamped to the training range so an
unprecedented launch cannot extrapolate the fit off a cliff.

Deterministic: comparables are processed in name order, no randomness, no
network, no files besides the case given. Runs in well under a second.

Input: every <id>.in in the input directory (the grader runs one case at a
time): JSON {"schema": "makefun/twitch-ccu-case/v1", ..., "known", "forecast",
"history_hourly", "comparables", ...}. Output: <id>.out with
{"hours": [{"hour": h, "q": [q05, q25, q50, q75, q95]}, ...]}, exactly 23
rows, hours 2..24, each q five finite numbers above 0 and at most 1e9,
non-decreasing.

Outside the grader's container, MAKEFUN_INPUT_DIR and MAKEFUN_OUTPUT_DIR
stand in for /input and /output:
    mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
"""

import json
import math
import os
import sys
from pathlib import Path

# The grader's -P leaves /bot off sys.path: put it back first, so an import of
# a module shipped beside this file would find it.
sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

# The quantile levels the contest scores, in the order each row lists them.
LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
# The standard normal's quantile at each level.
Z = [-1.645, -0.674, 0, 0.674, 1.645]
# Launches below this hour-1 print follow a different (late-unlock) regime.
MIN_TRAIN_H1 = 10000.0
# Floor on an hour's residual log-spread, so no hour forecasts with a
# spuriously narrow band.
MIN_SD = 0.25
MAX_VIEWERS = 1e9


def round2(x: float) -> float | int:
    """x rounded to 2 decimals, half up, as JavaScript's Math.round(x * 100) / 100 rounds it."""
    scaled = x * 100
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 100
    return int(value) if value.is_integer() else value


def norm_ppf(p: float) -> float:
    """Inverse standard normal CDF (Acklam's rational approximation)."""
    if p <= 0.0 or p >= 1.0:
        raise ValueError(f"level {p} outside (0, 1)")
    a = [-3.969683028665376e01, 2.209460984245205e02, -2.759285984469534e02,
         1.383577518672690e02, -3.066479806614716e01, 2.506628277459239e00]
    b = [-5.447609879822406e01, 1.615858368580409e02, -1.556989798598866e02,
         6.680131188771972e01, -1.328068155288572e01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e00,
         -2.549732539343734e00, 4.374664141464968e00, 2.938163982698783e00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e00,
         3.754408661907416e00]
    plow = 0.02425
    phigh = 1 - plow
    if p < plow:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
            ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    if p > phigh:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
            ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    q = p - 0.5
    r = q * q
    return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / \
        (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1)


def z_values(levels: list[float]) -> list[float]:
    """Standard-normal quantiles for the case's levels (exact for the usual five)."""
    out = []
    for level, z in zip(LEVELS, Z):
        out.append(z)
    if list(levels) == LEVELS:
        return out
    return [norm_ppf(level) for level in levels]


def solve3(matrix: list[list[float]], rhs: list[float]) -> list[float]:
    """Solve a 3x3 linear system by Gaussian elimination with partial pivoting."""
    m = [row[:] + [rhs[i]] for i, row in enumerate(matrix)]
    for col in range(3):
        piv = max(range(col, 3), key=lambda r: abs(m[r][col]))
        if abs(m[piv][col]) < 1e-12:
            raise ValueError("singular normal equations")
        m[col], m[piv] = m[piv], m[col]
        for r in range(3):
            if r != col:
                f = m[r][col] / m[col][col]
                for k in range(col, 4):
                    m[r][k] -= f * m[col][k]
    return [m[i][3] / m[i][i] for i in range(3)]


def comparable_series(comp: dict) -> tuple[list[float], float] | None:
    """A comparable's release-day hourly viewers (24 values) and its mean
    viewers over the 24 hours before release day. None when unusable."""
    hourly = comp.get("hourly") or []
    day_start = comp.get("day_start")
    if not day_start or not hourly:
        return None
    idx = None
    for i, row in enumerate(hourly):
        if row.get("from") == day_start:
            idx = i
            break
    if idx is None or len(hourly) < idx + 24:
        return None
    day = hourly[idx:idx + 24]
    viewers = []
    for row in day:
        v = row.get("viewers")
        if v is None:
            return None
        viewers.append(float(v))
    if viewers[0] <= 0:
        return None
    pre_rows = hourly[max(0, idx - 24):idx]
    pre_vals = [float(r["viewers"]) for r in pre_rows
                if r.get("viewers") is not None]
    if not pre_vals:
        return None
    pre = sum(pre_vals) / len(pre_vals)
    if pre <= 0:
        return None
    return viewers, pre


def case_pre_release(case: dict) -> float | None:
    """The case game's own mean viewers over the 24 hourly points before its
    release day, from history_hourly. None when unavailable."""
    hourly = case.get("history_hourly") or []
    day_start = case.get("day_start")
    if not hourly or not day_start:
        return None
    before = [r for r in hourly
              if r.get("from") and r["from"] < day_start
              and r.get("viewers") is not None]
    if not before:
        return None
    tail = before[-24:]
    return sum(float(r["viewers"]) for r in tail) / len(tail)


def fit_hour(train: list[tuple[float, float, list[float]]], hour: int):
    """Two-regressor OLS for one forecast hour. train rows are
    (h1, pre, day_viewers). Returns (a, b1, b2, sd, lo1, hi1, lo2, hi2)."""
    xs1 = [math.log(h1) for h1, _, _ in train]
    xs2 = [math.log(pre) for _, pre, _ in train]
    ys = [math.log(max(day[hour - 1], 1.0)) for _, _, day in train]
    n = len(train)
    s1 = sum(xs1)
    s2 = sum(xs2)
    sy = sum(ys)
    s11 = sum(x * x for x in xs1)
    s12 = sum(a * b for a, b in zip(xs1, xs2))
    s22 = sum(x * x for x in xs2)
    s1y = sum(a * b for a, b in zip(xs1, ys))
    s2y = sum(a * b for a, b in zip(xs2, ys))
    a, b1, b2 = solve3(
        [[float(n), s1, s2], [s1, s11, s12], [s2, s12, s22]],
        [sy, s1y, s2y])
    resid = [y - (a + b1 * x + b2 * z) for x, z, y in zip(xs1, xs2, ys)]
    mean_r = sum(resid) / n
    var = sum((r - mean_r) ** 2 for r in resid) / n
    sd = math.sqrt(var)
    return (a, b1, b2, sd, min(xs1), max(xs1), min(xs2), max(xs2))


def median_ratio_medians(train: list[tuple[float, float, list[float]]]) -> list[float]:
    """Per-hour median ratio to hour 1 across the training launches (the
    plain day-shape prior). Used only for its median, blended 50/50 in log
    space with the regression median: it hedges the regression against a
    training set whose evening hours are dominated by a few huge launches."""
    medians = []
    for h in range(2, 25):
        ratios = sorted(day[h - 1] / day[0] for _, _, day in train)
        n = len(ratios)
        medians.append(ratios[n // 2] if n % 2 else
                       0.5 * (ratios[n // 2 - 1] + ratios[n // 2]))
    return medians


def forecast_case(case: dict, name: str) -> dict:
    levels = list(case.get("levels") or LEVELS)
    zs = z_values(levels)
    known = case.get("known") or []
    first = next((k for k in known if k.get("hour") == 1), None)
    if first is None:
        raise ValueError(f"{name} has no hour-1 print")
    p1 = max(1.0, float(first["viewers"]))

    # Training rows from the case's comparables, in name order (deterministic).
    rows: list[tuple[float, float, list[float]]] = []
    for comp in sorted(case.get("comparables") or [],
                       key=lambda c: str(c.get("name", ""))):
        parsed = comparable_series(comp)
        if parsed is not None:
            viewers, pre = parsed
            rows.append((viewers[0], pre, viewers))
    if not rows:
        raise ValueError(f"{name} has no usable comparables")

    # Big hour-1 prints train on big hour-1 launches only; a small hour-1
    # print trains on everything (the late-unlock regime).
    train = [r for r in rows if r[0] >= MIN_TRAIN_H1] if p1 >= MIN_TRAIN_H1 else rows
    if len(train) < 8:
        train = rows

    pre = case_pre_release(case)
    if pre is None or pre <= 0:
        pres = sorted(r[1] for r in train)
        pre = pres[len(pres) // 2]

    fits = {h: fit_hour(train, h) for h in range(2, 25)}
    shape_medians = median_ratio_medians(train)
    log_h1 = math.log(p1)
    log_pre = math.log(pre)

    hours = []
    for row in case["forecast"]:
        h = int(row["hour"])
        a, b1, b2, sd, lo1, hi1, lo2, hi2 = fits[h]
        lh = min(max(log_h1, lo1), hi1)
        lp = min(max(log_pre, lo2), hi2)
        mu = a + b1 * lh + b2 * lp
        # Blend the regression median with the plain day-shape median in log
        # space, and spread the regression's residual spread around it.
        med = math.exp(0.5 * mu + 0.5 * math.log(p1 * shape_medians[h - 2]))
        spread = max(sd, MIN_SD)
        qs: list[float | int] = []
        prev = 0.0
        for z in zs:
            q = med * math.exp(z * spread)
            q = min(max(q, 1.0), MAX_VIEWERS)
            q = max(q, prev)  # keep the row non-decreasing
            qs.append(round2(q))
            prev = float(qs[-1])
        hours.append({"hour": h, "q": qs})
    return {"hours": hours}


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    # Every case file, in name order, so two runs write the same bytes.
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case, path.name)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
