#!/usr/bin/env python3
"""ATV OpenMM CUDA vs LAMMPS Kokkos CUDA — LJ NVE A/B.

CUDA precision is a **compile-time mode** aligned with LAMMPS Kokkos
`KOKKOS_PREC` — one knob: double | mixed | single
(`-DATV_CUDA_PREC_MODE=…`; see build_atv_bench.sh). Pick the matching binary
with --atv-bench.

Neighbor / Newton model is a **run-time** parameter on the LAMMPS side
(--lmp-neigh, --lmp-newton, …). ATV classical CUDA path is FULL cluster CSR
today; for a fairer arithmetic match use LMP --lmp-neigh full --lmp-newton off.

Examples:
  # double ATV vs LMP KK double (both defaults)
  python3 run_compare.py \\
      --atv-bench build-cuda-bench-double/bin/bench_nonbonded_style --tag double

  # mixed ATV (float math + double acc) — rebuild LMP with -D KOKKOS_PREC=mixed
  # for a same-mode A/B
  python3 run_compare.py \\
      --atv-bench build-cuda-bench-mixed/bin/bench_nonbonded_style --tag mixed
"""
from __future__ import annotations

import argparse
import os
import re
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
IN = Path(__file__).resolve().parent / "in.lj_nve.kk"


def _default_lmp() -> Path:
    arch = os.uname().machine
    serial = Path.home() / f".local/{arch}/bin/lmp-kk-serial"
    if serial.is_file():
        return serial
    return Path.home() / f".local/{arch}/bin/lmp"


def steps_for(n: int) -> tuple[int, int]:
    if n <= 100_000:
        return 20, 100
    if n <= 500_000:
        return 15, 50
    if n <= 1_000_000:
        return 10, 40
    if n <= 2_000_000:
        return 8, 30
    if n <= 5_000_000:
        return 5, 20
    return 5, 15


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--atv-bench",
        type=Path,
        default=Path(
            os.environ.get(
                "ATV_BENCH",
                ROOT / "build-cuda-bench/bin/bench_nonbonded_style",
            )
        ),
        help="Path to ATV bench_nonbonded_style (float or double build)",
    )
    p.add_argument(
        "--lmp-bin",
        type=Path,
        default=Path(os.environ.get("LMP_BIN", _default_lmp())),
        help="LAMMPS binary (prefer lmp-kk-serial under nested srun)",
    )
    p.add_argument(
        "--sizes",
        default=os.environ.get(
            "SIZES", "100000 250000 500000 1000000 2000000 5000000 10000000"
        ),
        help="Space-separated target atom counts",
    )
    # LAMMPS neighbor / package API parameters (run-time)
    p.add_argument(
        "--lmp-neigh",
        choices=("half", "full"),
        default=os.environ.get("LMP_NEIGH", "half"),
        help="package kokkos neigh half|full (default half; full≈ATV FULL CSR)",
    )
    p.add_argument(
        "--lmp-newton",
        choices=("on", "off"),
        default=os.environ.get("LMP_NEWTON", "on"),
        help="package kokkos newton on|off",
    )
    p.add_argument(
        "--lmp-comm",
        choices=("", "no", "host", "device"),
        default=os.environ.get("LMP_COMM", ""),
        help="optional package kokkos comm {no|host|device}",
    )
    p.add_argument(
        "--lmp-sort",
        choices=("", "no", "host", "device"),
        default=os.environ.get("LMP_SORT", ""),
        help="optional package kokkos sort {no|host|device}",
    )
    p.add_argument(
        "--lmp-extra-pk",
        default=os.environ.get("LMP_EXTRA_PK", ""),
        help="Extra tokens after -pk kokkos (quoted string)",
    )
    p.add_argument(
        "--tag",
        default="",
        help="Suffix for output log (e.g. float / double)",
    )
    p.add_argument(
        "--sweep-lmp",
        action="store_true",
        help="Only sweep LAMMPS package combos at --sweep-n (no ATV)",
    )
    p.add_argument(
        "--sweep-n",
        type=int,
        default=1_000_000,
        help="Atom count for --sweep-lmp (default 1e6)",
    )
    return p.parse_args()


def lmp_pk_tokens(args: argparse.Namespace) -> list[str]:
    pk = ["kokkos", "newton", args.lmp_newton, "neigh", args.lmp_neigh]
    if args.lmp_comm:
        pk += ["comm", args.lmp_comm]
    if args.lmp_sort:
        pk += ["sort", args.lmp_sort]
    if args.lmp_extra_pk.strip():
        pk += args.lmp_extra_pk.split()
    return pk


def run_atv(bench: Path, n: int, steps: int, warm: int) -> tuple[int, float, float] | None:
    r = subprocess.run(
        [str(bench), str(n), str(steps), str(warm)],
        capture_output=True,
        text=True,
    )
    for line in r.stdout.splitlines():
        if not line.startswith("path="):
            continue
        m = re.search(
            r"N=(\d+).*ms/step=([0-9.]+).*Matom\*step/s=([0-9.]+)", line
        )
        if m:
            return int(m.group(1)), float(m.group(2)), float(m.group(3))
    return None


def run_lmp(
    lmp: Path,
    n: int,
    steps: int,
    warm: int,
    pk: list[str],
    screen: Path,
) -> tuple[int, int, float] | None:
    """Return (natoms, nsteps, loop_seconds) for the last timed run."""
    cmd = [
        str(lmp),
        "-k", "on", "g", "1",
        "-sf", "kk",
        "-pk", *pk,
        "-in", str(IN),
        "-var", "N", str(n),
        "-var", "WARM", str(warm),
        "-var", "STEPS", str(steps),
        "-log", "none",
        "-screen", str(screen),
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    text = screen.read_text(errors="replace") if screen.exists() else ""
    if r.returncode != 0 and not text:
        text = (r.stderr or "") + (r.stdout or "")
    loops = [ln for ln in text.splitlines() if "Loop time" in ln]
    if r.returncode != 0 or not loops:
        err = next((ln for ln in text.splitlines() if "ERROR" in ln), "")
        print(f"  LMP FAIL rc={r.returncode} pk={pk} {err[:100]}")
        return None
    m = re.search(
        r"Loop time of ([0-9.eE+-]*) on .* for ([0-9]+) steps with ([0-9]+) atoms",
        loops[-1],
    )
    if not m:
        print(f"  LMP PARSE fail pk={pk}")
        return None
    return int(m.group(3)), int(m.group(2)), float(m.group(1))


def sweep_lmp(args: argparse.Namespace) -> list[dict]:
    """Coarse package kokkos grid search."""
    combos: list[dict] = []
    for neigh in ("half", "full"):
        for newton in ("on", "off"):
            for comm in ("", "device"):
                for sort in ("", "device"):
                    # skip empty-empty already covered; allow all
                    combos.append(
                        {
                            "neigh": neigh,
                            "newton": newton,
                            "comm": comm,
                            "sort": sort,
                        }
                    )
    # de-dup
    seen = set()
    uniq = []
    for c in combos:
        key = (c["neigh"], c["newton"], c["comm"], c["sort"])
        if key in seen:
            continue
        seen.add(key)
        uniq.append(c)

    n = args.sweep_n
    warm, steps = 10, 40
    print(f"=== LMP package sweep N≈{n} warm={warm} steps={steps} ===")
    print(
        f"{'neigh':<6} {'newton':<7} {'comm':<7} {'sort':<7} "
        f"{'ms/step':<12} {'Matom*s':<12}"
    )
    results = []
    for c in uniq:
        args.lmp_neigh = c["neigh"]
        args.lmp_newton = c["newton"]
        args.lmp_comm = c["comm"]
        args.lmp_sort = c["sort"]
        pk = lmp_pk_tokens(args)
        screen = Path(f"/tmp/lmp_sweep_{c['neigh']}_{c['newton']}_{c['comm'] or 'd'}_{c['sort'] or 'd'}.log")
        got = run_lmp(args.lmp_bin, n, steps, warm, pk, screen)
        if not got:
            print(f"{c['neigh']:<6} {c['newton']:<7} {c['comm'] or '-':<7} {c['sort'] or '-':<7} FAIL")
            continue
        nat, st, loop_s = got
        ms = 1000.0 * loop_s / st
        mat = nat / ms * 1e-3
        print(
            f"{c['neigh']:<6} {c['newton']:<7} {c['comm'] or '-':<7} {c['sort'] or '-':<7} "
            f"{ms:<12.4f} {mat:<12.3f}"
        )
        results.append({**c, "ms": ms, "mat": mat, "nat": nat})
    if results:
        best = min(results, key=lambda x: x["ms"])
        print(
            f"BEST: neigh={best['neigh']} newton={best['newton']} "
            f"comm={best['comm'] or '-'} sort={best['sort'] or '-'} "
            f"ms/step={best['ms']:.4f} Matom*s={best['mat']:.3f}"
        )
    return results


def main() -> None:
    args = parse_args()
    if args.sweep_lmp:
        sweep_lmp(args)
        return

    sizes = [int(x) for x in args.sizes.split()]
    tag = args.tag or "default"
    out = ROOT / "logs" / f"lj_atv_vs_lammps_kk_{tag}.txt"
    out.parent.mkdir(parents=True, exist_ok=True)

    pk = lmp_pk_tokens(args)
    print(f"host={os.uname().nodename} date={time.strftime('%Y-%m-%dT%H:%M:%S')}")
    print(f"ATV={args.atv_bench}")
    print(f"LMP={args.lmp_bin}")
    print(f"LMP -pk {' '.join(pk)}")
    print(f"tag={tag}")
    print(
        f"{'N':<12} {'engine':<10} {'ms/step':<14} {'Matom*step/s':<14} note"
    )
    print("-" * 72)

    rows: list[tuple] = []
    for N in sizes:
        warm, steps = steps_for(N)

        atv = run_atv(args.atv_bench, N, steps, warm)
        if atv is None:
            print(f"{N:<12} {'atv':<10} {'FAIL':<14} {'-':<14}")
            atv_ms = None
        else:
            atv_n, atv_ms, atv_mat = atv
            print(f"{atv_n:<12} {'atv':<10} {atv_ms:<14.4f} {atv_mat:<14.3f}")
            rows.append(("atv", atv_n, atv_ms, atv_mat, None))

        screen = Path(f"/tmp/lmp_{tag}_{N}.log")
        got = run_lmp(args.lmp_bin, N, steps, warm, pk, screen)
        if got is None:
            print(f"{N:<12} {'lmp_kk':<10} {'FAIL':<14} {'-':<14}")
            continue
        nat, st, loop_s = got
        ms = 1000.0 * loop_s / st
        mat = nat / ms * 1e-3
        ratio = (atv_ms / ms) if atv_ms else float("nan")
        print(
            f"{nat:<12} {'lmp_kk':<10} {ms:<14.4f} {mat:<14.3f} "
            f"atv/lmp_ms={ratio:.3f}"
        )
        rows.append(("lmp_kk", nat, ms, mat, ratio))

    with out.open("w") as f:
        f.write("# ATV OpenMM CUDA vs LAMMPS Kokkos CUDA — LJ NVE GH200\n")
        f.write(
            f"# {time.strftime('%Y-%m-%d %H:%M:%S')} host={os.uname().nodename}\n"
        )
        f.write(
            f"# density=0.8 cut=2.5 skin=0.3 dt=0.005 eps=sig=1 tag={tag}\n"
        )
        f.write(f"# LMP -pk {' '.join(pk)}\n")
        f.write(f"# ATV_BENCH={args.atv_bench}\n")
        f.write(
            f"{'N':<12} {'engine':<10} {'ms_step':<14} "
            f"{'Matom_step_s':<14} {'atv_over_lmp_ms':<16}\n"
        )
        for eng, n, ms, mat, ratio in rows:
            rs = f"{ratio:.3f}" if ratio is not None else "-"
            f.write(f"{n:<12} {eng:<10} {ms:<14.4f} {mat:<14.3f} {rs:<16}\n")
    print("wrote", out)


if __name__ == "__main__":
    main()
