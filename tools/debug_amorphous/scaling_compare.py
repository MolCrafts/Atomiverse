#!/usr/bin/env python3
"""scaling_compare.py -- Atomiverse vs LAMMPS strong-scaling table.

Reads existing Atomiverse logs (log.atomiverse.np{1,2,4,8}.txt) and the LAMMPS
logs in /tmp/atv_lammps_bench/ (log.lammps.np{1,2,4,8}.txt). Both ran the
identical amorphous_short.in protocol on PE_nc10_cl1000.dat (10000 atoms,
1000-step NVE/limit + Langevin then 2000-step NoseHooverNPT)."""

import re
from pathlib import Path

ATV_DIR = Path("/Users/roykid/work/molcrafts/Atomiverse/debug_amorphous")
LMP_DIR = Path("/tmp/atv_lammps_bench")
NPS = [1, 2, 4, 8]


def atv_times(np_):
    txt = (ATV_DIR / f"log.atomiverse.np{np_}.txt").read_text()
    s1 = float(re.search(r"stage1 wall_seconds\s*=\s*([0-9.]+)", txt).group(1))
    s2 = float(re.search(r"stage2 wall_seconds\s*=\s*([0-9.]+)", txt).group(1))
    return s1, s2


def lmp_times(np_):
    txt = (LMP_DIR / f"log.lammps.np{np_}.txt").read_text()
    loops = [float(m.group(1)) for m in re.finditer(r"Loop time of ([0-9.]+)", txt)]
    return loops[0], loops[1]


def fmt_row(label, t1, t2, t1_ref, t2_ref):
    total = t1 + t2
    total_ref = t1_ref + t2_ref
    return (f"  {label:<10s}  {t1:7.2f}  {t2:7.2f}  {total:7.2f}  "
            f"{t1_ref/t1:5.2f}x  {t2_ref/t2:5.2f}x  {total_ref/total:5.2f}x")


print("PE amorphous strong-scaling: 10000 atoms, 3000 steps total")
print("(LAMMPS stable=22Jul2025 vs Atomiverse master)\n")

print("Section A: Wall time (seconds, lower=better)")
print("-" * 78)
print(f"  {'NP':<3s}  {'code':<10s}  {'stage1':>7s}  {'stage2':>7s}  {'total':>7s}  "
      f"{'speedup vs np=1 (total)':>22s}")
print("-" * 78)

atv_t1_ref = atv_times(1)
lmp_t1_ref = lmp_times(1)

atv_data = {}
lmp_data = {}
for np_ in NPS:
    atv_data[np_] = atv_times(np_)
    lmp_data[np_] = lmp_times(np_)

for np_ in NPS:
    a1, a2 = atv_data[np_]
    l1, l2 = lmp_data[np_]
    a_speedup = sum(atv_data[1]) / sum(atv_data[np_])
    l_speedup = sum(lmp_data[1]) / sum(lmp_data[np_])
    print(f"  {np_:<3d}  Atomiverse  {a1:7.2f}  {a2:7.2f}  {a1+a2:7.2f}  {a_speedup:5.2f}x")
    print(f"  {' ':<3s}  LAMMPS      {l1:7.2f}  {l2:7.2f}  {l1+l2:7.2f}  {l_speedup:5.2f}x")
    print()

print("Section B: Atomiverse vs LAMMPS ratio (>1 = LAMMPS faster)")
print("-" * 78)
print(f"  {'NP':<3s}  {'stage1':>7s}  {'stage2':>7s}  {'total':>7s}")
print("-" * 78)
for np_ in NPS:
    a1, a2 = atv_data[np_]
    l1, l2 = lmp_data[np_]
    print(f"  {np_:<3d}  {a1/l1:6.2f}x  {a2/l2:6.2f}x  {(a1+a2)/(l1+l2):6.2f}x")

print()
print("Section C: Parallel efficiency (ideal=1.00, ratio of speedup vs np)")
print("-" * 78)
for np_ in NPS:
    a_speedup = sum(atv_data[1]) / sum(atv_data[np_])
    l_speedup = sum(lmp_data[1]) / sum(lmp_data[np_])
    print(f"  np={np_}  Atomiverse {a_speedup/np_:5.2f}   LAMMPS {l_speedup/np_:5.2f}")


# ----- plot ------------------------------------------------------------
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    print("\n(matplotlib not available, skipping plot)")
    raise SystemExit(0)

fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

# (1) wall-clock total
ax = axes[0]
ax.plot(NPS, [sum(atv_data[n]) for n in NPS], "o-", label="Atomiverse", lw=2, ms=8)
ax.plot(NPS, [sum(lmp_data[n]) for n in NPS], "s-", label="LAMMPS",     lw=2, ms=8)
ax.set_xscale("log", base=2); ax.set_yscale("log")
ax.set_xticks(NPS); ax.set_xticklabels(NPS)
ax.set_xlabel("MPI ranks"); ax.set_ylabel("total wall time (s)")
ax.set_title("Total wall time (3000 steps, 10k atoms)")
ax.grid(True, which="both", alpha=0.3); ax.legend()

# (2) speedup vs ideal
ax = axes[1]
ideal = NPS
ax.plot(NPS, ideal, "k--", label="ideal (linear)", lw=1.5)
ax.plot(NPS, [sum(atv_data[1])/sum(atv_data[n]) for n in NPS], "o-", label="Atomiverse", lw=2, ms=8)
ax.plot(NPS, [sum(lmp_data[1])/sum(lmp_data[n]) for n in NPS], "s-", label="LAMMPS",     lw=2, ms=8)
ax.set_xscale("log", base=2)
ax.set_xticks(NPS); ax.set_xticklabels(NPS)
ax.set_xlabel("MPI ranks"); ax.set_ylabel("speedup (T1 / Tn)")
ax.set_title("Strong scaling (higher = better)")
ax.grid(True, which="both", alpha=0.3); ax.legend()

# (3) per-stage breakdown
ax = axes[2]
import numpy as np
xs = np.arange(len(NPS))
w = 0.18
ax.bar(xs - 1.5*w, [atv_data[n][0] for n in NPS], w, label="ATV stage1 (NVE+Lang)", color="#1f77b4")
ax.bar(xs - 0.5*w, [atv_data[n][1] for n in NPS], w, label="ATV stage2 (NPT)",      color="#1f77b4", alpha=0.5)
ax.bar(xs + 0.5*w, [lmp_data[n][0] for n in NPS], w, label="LMP stage1",            color="#ff7f0e")
ax.bar(xs + 1.5*w, [lmp_data[n][1] for n in NPS], w, label="LMP stage2",            color="#ff7f0e", alpha=0.5)
ax.set_yscale("log")
ax.set_xticks(xs); ax.set_xticklabels([f"np={n}" for n in NPS])
ax.set_ylabel("wall time per stage (s)")
ax.set_title("Stage breakdown")
ax.grid(True, axis="y", which="both", alpha=0.3); ax.legend(fontsize=8)

plt.tight_layout()
out = ATV_DIR / "scaling_compare.png"
plt.savefig(out, dpi=140)
print(f"\nplot saved: {out}")
