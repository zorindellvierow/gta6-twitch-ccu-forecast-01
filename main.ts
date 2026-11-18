// gta6-twitch-ccu-forecast-01's house baseline (Node 22, standard library only): persistence. It forecasts that every
// hour of release day draws as many concurrent Twitch viewers in the Grand Theft Auto VI category as hour 1 did, with a
// spread that widens like a random walk's in log space:
//
//   q_tau(h) = p1 * exp(z_tau * 0.15 * sqrt(h - 1)), rounded to 2 decimals
//
// for the hours h = 2..24 of the UTC day, where p1 is hour 1's print (the case's `known` row for hour 1; a print below 1
// counts as 1), 0.15 is the spread in natural-log units per square-root hour, and z_tau is the standard normal's
// quantile at each level tau (0.05, 0.25, 0.5, 0.75, 0.95): -1.645, -0.674, 0, 0.674, 1.645. Its numbers are fixed in
// advance, never tuned on any data. A prize needs a loss strictly below the best house entry's: this one's, or the house
// model one-shot's.
//
// The kit's starter/main.ts is this program byte for byte (starter/main.py the same numbers in Python): start your entry
// from it. What an entry must do:
//   - read every <id>.in in the input directory (the grader runs one case at a time, with only that case's file there):
//     JSON {"schema":"makefun/twitch-ccu-case/v1","id","game","day_start","checkpoint","levels","known","forecast",
//     "history","history_hourly","streams","categories_hourly","comparables","notes"}, where `known` holds hour 1's
//     print {hour: 1, from, until, viewers, channels, reads} and `forecast` lists the hours to forecast, 2..24;
//   - write <id>.out to the output directory: {"hours":[{"hour":2,"q":[q05,q25,q50,q75,q95]}, ...]}, exactly 23 rows,
//     hours 2..24 each once, no other key, each q five finite numbers above 0 and at most 1e9 (viewers; fractions
//     allowed), non-decreasing; at most 65,536 bytes. Anything else is an invalid case, scored as if every quantile
//     equalled hour 1's print, and an entry with an invalid case takes no prize.
// It runs offline, with no network, under the case's time cap: everything it knows is in the case file.
//
// Outside the grader's container, MAKEFUN_INPUT_DIR and MAKEFUN_OUTPUT_DIR stand in for /input and /output:
//   mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out node main.ts

import { mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const INPUT_DIR = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const OUTPUT_DIR = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';

/** The quantile levels the contest scores, in the order each row lists them. */
const LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95];
/** The standard normal's quantile at each level. */
const Z = [-1.645, -0.674, 0, 0.674, 1.645];
/** The spread of an hour's move, in natural-log units per square-root hour. */
const SIGMA = 0.15;

interface Case {
  levels: number[];
  known: Array<{ hour: number; viewers: number }>;
  forecast: Array<{ hour: number }>;
}

/** x rounded to 2 decimals (half up). */
const round2 = (x: number): number => Math.round(x * 100) / 100;

/** The persistence forecast of one hour h: the five quantiles around hour 1's print. */
const hourQuantiles = (p1: number, h: number): number[] => Z.map((z) => round2(p1 * Math.exp(z * SIGMA * Math.sqrt(h - 1))));

mkdirSync(OUTPUT_DIR, { recursive: true });
// Every case file, in name order, so two runs write the same bytes.
const inputs = readdirSync(INPUT_DIR).filter((f) => f.endsWith('.in')).sort();
for (const file of inputs) {
  const id = file.slice(0, -'.in'.length);
  const c = JSON.parse(readFileSync(join(INPUT_DIR, file), 'utf8')) as Case;
  if (c.levels.length !== LEVELS.length || c.levels.some((t, j) => t !== LEVELS[j])) throw new Error(`${file} asks for levels ${JSON.stringify(c.levels)}, not ${JSON.stringify(LEVELS)}`);
  const first = c.known.find((k) => k.hour === 1);
  if (first === undefined) throw new Error(`${file} has no hour-1 print`);
  const p1 = Math.max(1, first.viewers);
  const hours = c.forecast.map(({ hour }) => ({ hour, q: hourQuantiles(p1, hour) }));
  writeFileSync(join(OUTPUT_DIR, `${id}.out`), `${JSON.stringify({ hours })}\n`);
}
