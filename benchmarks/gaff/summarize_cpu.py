#!/usr/bin/env python3
"""Summarise a run_cpu_matrix.sh output directory.

For every ``np<ranks>_nt<threads>`` configuration: median steps/s of the
repeats for Atomiverse and LAMMPS, the ATV/LAMMPS wall ratio (below 1 means
Atomiverse is faster), and the per-section ratio of median average seconds
(LAMMPS ``timer full`` sections; Atomiverse prints the same labels).

Atomiverse wall is ``sim.run`` of the timed segment (includes that run's
setup force evaluation); LAMMPS is ``Loop time`` of the timed ``run``.
Atomiverse timing tables are cumulative per driver, so its timed sections
are the last table minus the warmup table.

Usage: python summarize_cpu.py OUT [--json OUT/summary.json]
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
from collections import defaultdict
from pathlib import Path

SECTIONS = ("Pair", "Bond", "Kspace", "Neigh", "Comm", "Output", "Modify",
            "Other")
TAG = re.compile(r"(atv|lmp)_np(\d+)_nt(\d+)_r(\d+)\.log$")
NUM = r"([-+0-9.eE]+)"
# Timing rows carry min/avg/max/%varavg/%total (≥ 4 numbers); the memory
# table that follows in Atomiverse logs reuses section names with 2 numbers.
SEP = r"[\s|│]+"
ROW = re.compile(r"^\W*(" + "|".join(SECTIONS) + r")\b" + SEP + NUM + SEP
                 + NUM + SEP + NUM + SEP + NUM)


def timing_tables(text: str) -> list[dict[str, float]]:
    """Average seconds per section of every timing table in a log."""
    tables: list[dict[str, float]] = []
    for line in text.splitlines():
        match = ROW.match(line)
        if match:
            if match.group(1) == "Pair":
                tables.append({})
            if tables:
                tables[-1][match.group(1)] = float(match.group(3))
    return tables


def timed_sections(text: str, cumulative: bool) -> dict[str, float]:
    """Sections of the timed run.

    LAMMPS prints one table per ``run``; the last is the timed run.
    Atomiverse prints one per ``run()`` that is *cumulative* over the
    driver's lifetime, so the timed run is the last table minus the one
    before it (the warmup).
    """
    if not cumulative:
        # The timed run's table follows its "Loop time" line; with OPENMP
        # LAMMPS prints a per-thread table after it, which must not win.
        loop = text.rfind("Loop time of")
        tables = timing_tables(text[loop:]) if loop >= 0 else []
        return tables[0] if tables else {}
    tables = timing_tables(text)
    if not tables:
        return {}
    last = tables[-1]
    if len(tables) >= 2:
        prev = tables[-2]
        return {k: v - prev.get(k, 0.0) for k, v in last.items()}
    return last


def atv_steps_per_s(text: str) -> float | None:
    match = re.findall(r"PERFORMANCE\s+wall=\S+ s\s+steps/s=([0-9.]+)", text)
    return float(match[-1]) if match else None


def lmp_steps_per_s(text: str) -> float | None:
    match = re.findall(r"Loop time of ([0-9.eE+-]+) on \d+ procs for (\d+) "
                       r"steps", text)
    if not match:
        return None
    seconds, steps = match[-1]
    return int(steps) / float(seconds)


def summarise(out: Path) -> dict:
    runs: dict = defaultdict(lambda: {"atv": [], "lmp": []})
    for log in sorted(out.glob("*.log")):
        match = TAG.search(log.name)
        if not match:
            continue
        code, np_, nt, _ = match.groups()
        text = log.read_text(errors="replace")
        rate = atv_steps_per_s(text) if code == "atv" else lmp_steps_per_s(
            text)
        if rate is None:
            print(f"warning: {log.name} has no timing (failed run?)")
            continue
        runs[(int(np_), int(nt))][code].append(
            {"steps_per_s": rate,
             "sections": timed_sections(text, cumulative=code == "atv")})
    table = {}
    for (np_, nt), codes in sorted(runs.items()):
        row: dict = {"ranks": np_, "threads": nt}
        for code in ("atv", "lmp"):
            reps = codes[code]
            if not reps:
                continue
            row[code] = {
                "reps": len(reps),
                "steps_per_s": statistics.median(r["steps_per_s"]
                                                 for r in reps),
                "sections_s": {
                    s: statistics.median(r["sections"].get(s, 0.0)
                                         for r in reps) for s in SECTIONS},
            }
        if "atv" in row and "lmp" in row:
            row["wall_ratio"] = (row["lmp"]["steps_per_s"]
                                 / row["atv"]["steps_per_s"])
            row["section_ratio"] = {
                s: (row["atv"]["sections_s"][s] / row["lmp"]["sections_s"][s]
                    if row["lmp"]["sections_s"][s] > 0 else None)
                for s in SECTIONS}
        table[f"{np_}x{nt}"] = row
    return table


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("out", type=Path)
    ap.add_argument("--json", type=Path)
    args = ap.parse_args()
    table = summarise(args.out)
    print(f"{'config':>8}{'ATV st/s':>11}{'LMP st/s':>11}{'ATV/LMP':>9}  "
          + "".join(f"{s:>8}" for s in SECTIONS))
    for cfg, row in table.items():
        if "wall_ratio" not in row:
            continue
        ratios = "".join(
            f"{r:8.2f}" if r is not None else f"{'-':>8}"
            for r in row["section_ratio"].values())
        print(f"{cfg:>8}{row['atv']['steps_per_s']:11.1f}"
              f"{row['lmp']['steps_per_s']:11.1f}{row['wall_ratio']:9.3f}  "
              + ratios)
    path = args.json or args.out / "summary.json"
    path.write_text(json.dumps(table, indent=2) + "\n")


if __name__ == "__main__":
    main()
