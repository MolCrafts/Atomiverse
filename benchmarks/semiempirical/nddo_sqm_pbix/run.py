"""AM1 single point, Atomiverse vs AmberTools SQM, on one 205-atom molecule.

NOT part of ctest and asserts no timing. Methodology and results: README.md.

    python benchmarks/semiempirical/nddo_sqm_pbix/run.py [--threads 1 2 4 10]
        [--repeats 3] [--sqm /path/to/sqm] [--deck trimer205.sqm.in]
        [--sqm-repeats 1]

Decks: ``trimer205.sqm.in`` (205 atoms) and ``chain674.sqm.in`` (674 atoms).

Each Atomiverse sample is a fresh interpreter (cold: OpenMP/BLAS start-up and
the first build are paid every time) running ``atomiverse.single_point`` with
``AM1(charge)``, i.e. SCF + Mulliken charges + analytic forces. SQM runs the
same deck (``maxcyc=0``: one SCF, gradient included). Both run at the same
``OMP_NUM_THREADS``; SQM's own SCF is serial. The script prints the median
wall time per thread count, the SCF iteration counts and the observable
differences (heat of formation, max |dq|) against SQM's printed values.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent

_CHILD = r"""
import json, sys, time
import numpy as np, molpy as mp, atomiverse as av
z, xyz, charge = json.loads(sys.stdin.read())
xyz = np.asarray(xyz, dtype=float)
frame = mp.Frame({"atoms": {"atomic_number": np.asarray(z, dtype=np.int64),
                            "x": xyz[:, 0], "y": xyz[:, 1], "z": xyz[:, 2]}})
method = av.AM1(charge=charge)
start = time.perf_counter()
av.single_point(frame, method)
seconds = time.perf_counter() - start
r = method.last_result()
print(json.dumps({"seconds": seconds, "iterations": r.iterations,
                  "hof": r.heat_of_formation_kcal_mol,
                  "energy_ev": r.total_energy_ev,
                  "charges": list(r.mulliken_charges)}))
"""


def read_deck(deck: Path) -> tuple[list[int], list[list[float]], int]:
    lines = deck.read_text().splitlines()
    charge = int(re.search(r"qmcharge\s*=\s*(-?\d+)", deck.read_text()).group(1))
    rows = [line.split() for line in lines[lines.index(" /") + 1 :] if line.strip()]
    return [int(r[0]) for r in rows], [[float(v) for v in r[2:5]] for r in rows], charge


def run_atomiverse(payload: str, threads: int) -> dict:
    env = dict(os.environ, OMP_NUM_THREADS=str(threads))
    for key in ("OMP_PROC_BIND", "OMP_PLACES"):
        env.pop(key, None)
    out = subprocess.run([sys.executable, "-c", _CHILD], input=payload, env=env,
                         capture_output=True, text=True, check=True).stdout
    return json.loads(out.strip().splitlines()[-1])


def run_sqm(sqm: str, deck: Path, threads: int) -> dict:
    with tempfile.TemporaryDirectory() as tmp:
        shutil.copy(deck, Path(tmp) / "sqm.in")
        env = dict(os.environ, OMP_NUM_THREADS=str(threads),
                   OPENBLAS_NUM_THREADS=str(threads))
        start = time.perf_counter()
        subprocess.run([sqm, "-O", "-i", "sqm.in", "-o", "sqm.out"], cwd=tmp,
                       env=env, check=True, capture_output=True)
        seconds = time.perf_counter() - start
        text = (Path(tmp) / "sqm.out").read_text()
    table = text.split("Mulliken Charge\n")[1].split("Total Mulliken Charge")[0]
    # Printed only at verbosity >= 2; the shipped deck leaves it at 0.
    cycles = re.search(r"SCF Converged to .* in:\s+(\d+) Cycles", text)
    return {"seconds": seconds,
            "iterations": int(cycles.group(1)) if cycles else None,
            "hof": float(re.search(r"Heat of formation\s*=\s*(-?[\d.]+)", text).group(1)),
            "charges": [float(line.split()[-1]) for line in table.strip().splitlines()]}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--threads", type=int, nargs="+", default=[1, 2, 4, 10])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--sqm-repeats", type=int, default=None,
                        help="SQM samples per thread count (default: --repeats)")
    parser.add_argument("--sqm", default=os.environ.get("SQM") or shutil.which("sqm"))
    parser.add_argument("--deck", type=Path, default=HERE / "trimer205.sqm.in")
    args = parser.parse_args()
    if not args.sqm:
        parser.error("no sqm executable: pass --sqm or set SQM")
    z, xyz, charge = read_deck(args.deck)
    payload = json.dumps([z, xyz, charge])
    print(f"{len(z)} atoms, load average {os.getloadavg()[0]:.1f}")
    print("threads  atv_median_s  sqm_median_s  atv/sqm  atv_it  sqm_it  dHOF_kcal  max|dq|")
    for threads in args.threads:
        atv = [run_atomiverse(payload, threads) for _ in range(args.repeats)]
        ref = [run_sqm(args.sqm, args.deck, threads)
               for _ in range(args.sqm_repeats or args.repeats)]
        t_atv = statistics.median(r["seconds"] for r in atv)
        t_sqm = statistics.median(r["seconds"] for r in ref)
        dq = max(abs(a - b) for a, b in zip(atv[0]["charges"], ref[0]["charges"]))
        print(f"{threads:7d}  {t_atv:12.3f}  {t_sqm:12.3f}  {t_atv / t_sqm:7.2f}  "
              f"{atv[0]['iterations']:6d}  {ref[0]['iterations'] or '-':>6}  "
              f"{atv[0]['hof'] - ref[0]['hof']:+9.4f}  {dq:7.4f}")
    print(f"load average {os.getloadavg()[0]:.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
