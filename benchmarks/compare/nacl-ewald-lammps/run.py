#!/usr/bin/env python3
"""Time bench_nacl_ewald (one or more builds) and LAMMPS on the same 512-ion
NaCl system, state points, Ewald settings and thread counts; print ms/step
tables. Standard library only.

    run.py --atv before=/path/bench_nacl_ewald --atv after=/path/... \\
           --lmp /path/lmp_omp --out DIR [--threads 1 4 8 16 64]
    run.py --summarize --out DIR

Every run appends one JSON line to DIR/results.jsonl; --summarize only
rereads it. Engines run one after the other, never concurrently.
"""

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
BUCKETS = ("pair", "kspace", "integrate", "thermostat", "neigh", "other")
LAMMPS_SECTIONS = {
    "Pair": "pair",
    "Kspace": "kspace",
    "Neigh": "neigh",
    "Comm": "other",
    "Output": "other",
    "Modify": "integrate",
    "Other": "other",
    "Bond": "other",
}


def run_atv(label, binary, state, setting, threads, steps, warm, prefix, thermostat):
    env = dict(os.environ, OMP_NUM_THREADS=str(threads))
    out = subprocess.run(
        [
            binary,
            state,
            setting,
            str(threads),
            str(steps),
            str(warm),
            "--thermostat",
            thermostat,
            "--lammps",
            str(prefix),
        ],
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    line = next(ln for ln in out.stdout.splitlines() if ln.startswith("BENCH "))
    fields = dict(item.split("=", 1) for item in line.split()[1:])
    if thermostat != "gauss":
        label = f"{label}/{thermostat}"
    row = {
        "engine": label,
        "state": state,
        "setting": setting,
        "threads": threads,
        "nk": int(fields["nk"]),
        "ms": float(fields["loop_ms"]),
    }
    for b in BUCKETS:
        row[b] = float(fields[f"{b}_ms"])
    return row


def run_lammps(binary, state, setting, threads, steps, warm, prefix):
    env = dict(os.environ, OMP_NUM_THREADS=str(threads))
    cmd = [
        binary,
        "-in",
        str(HERE / "in.nacl"),
        "-var",
        "PREFIX",
        str(prefix),
        "-var",
        "WARM",
        str(warm),
        "-var",
        "STEPS",
        str(steps),
        "-log",
        f"{prefix}.lmp.t{threads}.log",
        "-screen",
        "none",
    ]
    if threads > 1:
        cmd += ["-sf", "omp", "-pk", "omp", str(threads)]
    proc = subprocess.run(cmd, check=False, env=env)
    log = Path(f"{prefix}.lmp.t{threads}.log")
    text = log.read_text() if log.exists() else ""
    # Judge the run by its log: some LAMMPS builds abort in teardown after a
    # complete run, and that exit status says nothing about the timings.
    if "Total wall time" not in text:
        raise RuntimeError(
            f"LAMMPS exited {proc.returncode} before finishing; see {log}"
        )
    if proc.returncode != 0:
        print(
            f"note: LAMMPS exited {proc.returncode} after a complete run ({log})",
            file=sys.stderr,
        )
    loop = re.findall(r"Loop time of ([0-9.eE+-]+) on .* for (\d+) steps", text)[-1]
    if int(loop[1]) != steps:
        raise RuntimeError(f"LAMMPS ran {loop[1]} steps, expected {steps}")
    nk = int(re.findall(r"KSpace vectors: actual max1d max3d = (\d+)", text)[-1])
    row = {
        "engine": "lammps",
        "state": state,
        "setting": setting,
        "threads": threads,
        "nk": nk,
        "ms": 1e3 * float(loop[0]) / steps,
    }
    for b in BUCKETS:
        row[b] = 0.0
    last = text[text.rfind("MPI task timing breakdown") :]
    for line in last.splitlines():
        cells = [c.strip() for c in line.split("|")]
        if cells[0] in LAMMPS_SECTIONS and len(cells) >= 3 and cells[2]:
            row[LAMMPS_SECTIONS[cells[0]]] += 1e3 * float(cells[2]) / steps
    return row


def summarize(out):
    rows = [
        json.loads(line) for line in (out / "results.jsonl").read_text().splitlines()
    ]
    engines = list(dict.fromkeys(r["engine"] for r in rows))
    threads = sorted({r["threads"] for r in rows})
    cases = list(dict.fromkeys((r["state"], r["setting"]) for r in rows))
    pick = {(r["engine"], r["state"], r["setting"], r["threads"]): r for r in rows}

    print("## ms/step (loop)\n")
    print(
        "| state | setting | engine | k-vectors | "
        + " | ".join(f"{t} thr" for t in threads)
        + " |"
    )
    print("|---" * (4 + len(threads)) + "|")
    for state, setting in cases:
        for e in engines:
            got = [pick.get((e, state, setting, t)) for t in threads]
            nk = next((g["nk"] for g in got if g), "")
            cells = [f"{g['ms']:.3f}" if g else "" for g in got]
            print(f"| {state} | {setting} | {e} | {nk} | " + " | ".join(cells) + " |")
    for state, setting in cases:
        print(f"\n## {state}, {setting}: ms/step per component\n")
        print("| engine | threads | " + " | ".join(BUCKETS) + " | loop |")
        print("|---" * (3 + len(BUCKETS)) + "|")
        for e in engines:
            for t in threads:
                g = pick.get((e, state, setting, t))
                if g:
                    print(
                        f"| {e} | {t} | "
                        + " | ".join(f"{g[b]:.3f}" for b in BUCKETS)
                        + f" | {g['ms']:.3f} |"
                    )


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--atv", action="append", default=[], metavar="LABEL=PATH")
    ap.add_argument(
        "--lmp", help="LAMMPS binary with KSPACE (+ OPENMP for threads > 1)"
    )
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--threads", type=int, nargs="+", default=[1, 4, 8, 16, 64])
    ap.add_argument("--states", nargs="+", default=["liquid", "fluid"])
    ap.add_argument("--settings", nargs="+", default=["paper", "accuracy"])
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--warm", type=int, default=200)
    ap.add_argument(
        "--thermostat",
        choices=("gauss", "config"),
        default="gauss",
        help="Atomiverse thermostat (LAMMPS always runs fix nvt)",
    )
    ap.add_argument("--summarize", action="store_true", help="only print the tables")
    a = ap.parse_args()
    if not a.summarize:
        if not a.atv:
            sys.exit("give at least one --atv LABEL=PATH")
        a.out.mkdir(parents=True, exist_ok=True)
        atv = [spec.split("=", 1) for spec in a.atv]
        with open(a.out / "results.jsonl", "a") as log:
            for state in a.states:
                for setting in a.settings:
                    prefix = a.out / f"{state}-{setting}"
                    for t in a.threads:
                        rows = [
                            run_atv(
                                label,
                                path,
                                state,
                                setting,
                                t,
                                a.steps,
                                a.warm,
                                prefix,
                                a.thermostat,
                            )
                            for label, path in atv
                        ]
                        if a.lmp:
                            rows.append(
                                run_lammps(
                                    a.lmp, state, setting, t, a.steps, a.warm, prefix
                                )
                            )
                        for row in rows:
                            log.write(json.dumps(row) + "\n")
                            log.flush()
                            print(json.dumps(row), flush=True)
    summarize(a.out)


if __name__ == "__main__":
    main()
