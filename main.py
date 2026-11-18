"""gta6-twitch-ccu-forecast-01's starter in Python 3.13 (standard library only): persistence.

The same program as starter/main.ts (the house baseline), number for number and byte for byte: it forecasts that every
hour of release day draws as many concurrent Twitch viewers in the Grand Theft Auto VI category as hour 1 did, with a
spread that widens like a random walk's in log space:

    q_tau(h) = p1 * exp(z_tau * 0.15 * sqrt(h - 1)), rounded to 2 decimals (half up)

for the hours h = 2..24 of the UTC day, where p1 is hour 1's print (the case's `known` row for hour 1; a print below 1
counts as 1) and z_tau is the standard normal's quantile at each level (0.05, 0.25, 0.5, 0.75, 0.95):
-1.645, -0.674, 0, 0.674, 1.645.

How the grader runs it: `python3 -P -s /bot/main.py` (PYTHONHASHSEED=0) in python:3.13-slim, standard library only: no
network, no pip packages. -P keeps this file's directory OFF sys.path, so a module you ship beside main.py (say
/bot/model.py) imports only after the sys.path line below, which puts the directory first; -s drops user
site-packages.

What an entry must do:
  - read every <id>.in in the input directory (the grader runs one case at a time, with only that case's file there):
    JSON {"schema": "makefun/twitch-ccu-case/v1", "id", "game", "day_start", "checkpoint", "levels", "known",
    "forecast", "history", "history_hourly", "streams", "categories_hourly", "comparables", "notes"}, where `known`
    holds hour 1's print {hour: 1, from, until, viewers, channels, reads} and `forecast` lists the hours to forecast,
    2..24;
  - write <id>.out to the output directory: {"hours": [{"hour": 2, "q": [q05, q25, q50, q75, q95]}, ...]}, exactly 23
    rows, hours 2..24 each once, no other key, each q five finite numbers above 0 and at most 1e9 (viewers; fractions
    allowed), non-decreasing; at most 65,536 bytes. Anything else is an invalid case, scored as if every quantile
    equalled hour 1's print, and an entry with an invalid case takes no prize.

Outside the grader's container, MAKEFUN_INPUT_DIR and MAKEFUN_OUTPUT_DIR stand in for /input and /output:
    mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
"""

import json
import math
import os
import sys
from pathlib import Path

# The grader's -P leaves /bot off sys.path: put it back first, so `import model` finds a model.py shipped beside this file.
sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

# The quantile levels the contest scores, in the order each row lists them.
LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
# The standard normal's quantile at each level.
Z = [-1.645, -0.674, 0, 0.674, 1.645]
# The spread of an hour's move, in natural-log units per square-root hour.
SIGMA = 0.15


def round2(x: float) -> float | int:
    """x rounded to 2 decimals, half up, as JavaScript's Math.round(x * 100) / 100 rounds it.

    A whole number comes back as an int, so json.dumps writes 1240 where JavaScript writes 1240 (not 1240.0).
    """
    scaled = x * 100
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 100
    return int(value) if value.is_integer() else value


def hour_quantiles(p1: float, h: int) -> list[float | int]:
    """The persistence forecast of hour h: the five quantiles around hour 1's print."""
    return [round2(p1 * math.exp(z * SIGMA * math.sqrt(h - 1))) for z in Z]


def forecast_case(case: dict, name: str) -> dict:
    if case["levels"] != LEVELS:
        raise ValueError(f"{name} asks for levels {case['levels']}, not {LEVELS}")
    first = next((k for k in case["known"] if k["hour"] == 1), None)
    if first is None:
        raise ValueError(f"{name} has no hour-1 print")
    p1 = max(1, first["viewers"])
    return {"hours": [{"hour": row["hour"], "q": hour_quantiles(p1, row["hour"])} for row in case["forecast"]]}


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    # Every case file, in name order, so two runs write the same bytes.
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case, path.name)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
