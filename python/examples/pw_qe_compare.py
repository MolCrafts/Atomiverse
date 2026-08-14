#!/usr/bin/env python3
"""Compare Atomiverse plane-wave DFT against Quantum ESPRESSO for Si diamond.

Both codes read the **same** norm-conserving pseudopotential file and are
given the same Hamiltonian, or the comparison means nothing:

* Pseudopotential — the in-tree ONCVPSP norm-conserving UPF
  ``tests/data/pw/Si_lda_nc.upf``, used verbatim by Atomiverse
  (``PW.upf_path``) and by QE (``ATOMIC_SPECIES``). The plane-wave path
  implements norm-conserving pseudopotentials only: it applies no
  augmentation charge, so an ultrasoft or PAW file is rejected on read
  rather than integrated as if it were norm-conserving.
* XC — the QE input pins ``input_dft = 'PBE'`` because the Python ``PW``
  surface runs its PBE default and does not currently expose an ``xc``
  member (see NOTE at ``QE_INPUT_TEMPLATE``). The PP is LDA-generated;
  both codes use it identically, so the two totals stay comparable.
* Occupations — fixed on both sides (Si diamond is a gapped insulator);
  the Python surface exposes no smearing member either.

Requires:
    - atomiverse (with PW enabled): ``pip install -e './python'``
    - examples extras: ``pip install -e './python[examples]'``
      (``molcrafts-molplot`` for figures, ``molcrafts-molpy`` for the
      MolCrafts system stack — not matplotlib / ASE)
    - Quantum ESPRESSO ``pw.x`` (skip with ``--skip-qe``)

Generates comparison plots (via **molplot**, not raw matplotlib):
    1. Total energy vs plane-wave cutoff (ecut)
    2. Total energy vs k-point mesh
    3. Total energy vs lattice constant (EOS)
    4. |E_AV − E_QE| vs ecut

Usage::

    python examples/pw_qe_compare.py
    python examples/pw_qe_compare.py --skip-qe       # Atomiverse only
    python examples/pw_qe_compare.py --skip-atomiverse  # QE only
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import molrs
import numpy as np

# ── Paths ──

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
# In-tree ONCVPSP norm-conserving UPF (pseudo_type="NC", Z_val = 4).
DEFAULT_UPF = REPO_ROOT / "tests" / "data" / "pw" / "Si_lda_nc.upf"
QE_BIN = "/tmp/qe-install/bin/pw.x"

# ── Si diamond structure builder ──


def build_si_diamond(a_lat: float = 5.43) -> "molrs.Frame":
    """Build the Si diamond primitive FCC cell (2 atoms) as a molrs.Frame."""
    a4 = a_lat / 4.0
    frame = molrs.Frame()
    atoms = molrs.Block()
    atoms.insert("x", np.array([0.0, a4]))
    atoms.insert("y", np.array([0.0, a4]))
    atoms.insert("z", np.array([0.0, a4]))
    atoms.insert("atomic_number", np.array([14, 14], dtype=np.uint32))
    atoms.insert("mass", np.full(2, 28.0855))
    frame["atoms"] = atoms

    h = a_lat / 2.0
    frame.box = molrs.Box(
        np.array(
            [
                [0.0, h, h],
                [h, 0.0, h],
                [h, h, 0.0],
            ]
        )
    )
    return frame


# ── QE helpers ──

# NOTE — the two settings that keep this an apples-to-apples comparison:
#   input_dft = 'PBE'      matches the Python PW surface, which runs its PBE
#                          default; the binding exposes no `xc` member, so the
#                          QE side is pinned instead of the Atomiverse side.
#   occupations = 'fixed'  matches the Python PW surface, which exposes no
#                          smearing member. Si diamond is a gapped insulator.
# `pseudo_dir` / the ATOMIC_SPECIES filename are both taken from the one
# resolved UPF path, so QE reads literally the file Atomiverse read.
QE_INPUT_TEMPLATE = """&control
    calculation  = 'scf'
    prefix       = 'si'
    pseudo_dir   = '{pseudo_dir}'
    outdir       = '{outdir}'
    verbosity    = 'low'
/
&system
    ibrav        = 2
    celldm(1)    = {alat_bohr}
    nat          = 2
    ntyp         = 1
    ecutwfc      = {ecut_ry}
    input_dft    = 'PBE'
    occupations  = 'fixed'
/
&electrons
    conv_thr     = {conv_thr}
    mixing_beta  = 0.4
/
ATOMIC_SPECIES
    Si   28.0855  {upf_file}
ATOMIC_POSITIONS crystal
    Si   0.000000  0.000000  0.000000
    Si   0.250000  0.250000  0.250000
K_POINTS automatic
    {kmesh} {kmesh} {kmesh}  0 0 0
"""


def write_qe_input(
    path: str,
    *,
    a_lat: float,
    ecut_ry: float,
    kmesh: int,
    upf_path: str,
    conv_thr: float = 1e-6,
) -> str:
    """Write a QE input file and return the outdir path.

    ``upf_path`` is the same file handed to ``PW.upf_path``; its directory
    becomes QE's ``pseudo_dir`` and its name the ``ATOMIC_SPECIES`` entry.
    """
    alat_bohr = a_lat / 0.52917721092  # Angstrom → Bohr
    outdir = str(Path(path).parent / "qe_out")
    upf = Path(upf_path).resolve()

    content = QE_INPUT_TEMPLATE.format(
        pseudo_dir=upf.parent,
        upf_file=upf.name,
        outdir=outdir,
        alat_bohr=f"{alat_bohr:.8f}",
        ecut_ry=f"{ecut_ry:.4f}",
        conv_thr=f"{conv_thr:.1e}",
        kmesh=kmesh,
    )
    Path(path).write_text(content)
    return outdir


def run_qe(input_path: str, outdir: str) -> dict | None:
    """Run pw.x and return parsed results, or None on failure."""
    try:
        result = subprocess.run(
            [QE_BIN, "-i", input_path],
            capture_output=True,
            text=True,
            timeout=300,
            env={**os.environ, "TMPDIR": outdir},
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None

    output = result.stdout + result.stderr

    # Parse total energy
    energy = None
    converged = False
    iterations = 0

    # Look for "!    total energy"
    for line in output.splitlines():
        m = re.search(r"!\s+total energy\s+=\s+([\d\.\-]+)\s+Ry", line)
        if m:
            energy = float(m.group(1))

    # Count SCF iterations
    for line in output.splitlines():
        if "estimated scf accuracy" in line:
            iterations += 1

    # Check convergence
    converged = "convergence has been achieved" in output

    if energy is None:
        return None

    return {
        "energy": energy,
        "converged": converged,
        "iterations": iterations,
    }


# ── Atomiverse helpers ──


def run_atomiverse(
    a_lat: float,
    ecut_ry: float,
    kmesh: int,
    upf_path: str,
    tol: float = 1e-6,
    max_iter: int = 80,
) -> dict:
    """Run Atomiverse PW and return results.

    ``upf_path`` is required: a plane-wave run has no default
    pseudopotential, and ``PW.build`` raises when neither ``upf_path`` nor
    ``gth_name`` is set.
    """
    import atomiverse as av

    frame = build_si_diamond(a_lat)
    pw = av.PW()
    pw.upf_path = upf_path
    pw.ecut_ry = ecut_ry
    pw.k_mesh = kmesh
    pw.tol = tol
    pw.max_iter = max_iter
    pw.build(frame)
    r = pw.run(frame)
    return {
        "energy": r.energy,
        "converged": r.converged,
        "iterations": r.iterations,
    }


# ── Parameter sweep ──


def sweep_ecut(skip_qe: bool, skip_av: bool, upf: str) -> list[dict]:
    """Energy vs plane-wave cutoff."""
    results = []
    ecuts = [10, 15, 20, 25, 30, 35, 40, 50, 60]

    for ecut in ecuts:
        point = {"param": "ecut", "value": ecut}

        if not skip_av:
            av = run_atomiverse(5.43, ecut, 4, upf, max_iter=80)
            point["atomiverse"] = av
            print(
                f"  AV  ecut={ecut:3.0f} Ry  E={av['energy']:15.6f}  iters={av['iterations']}"
            )

        if not skip_qe:
            with tempfile.TemporaryDirectory() as tmpdir:
                inp = os.path.join(tmpdir, "si.in")
                outdir = write_qe_input(
                    inp, a_lat=5.43, ecut_ry=ecut, kmesh=4, upf_path=upf
                )
                os.makedirs(outdir, exist_ok=True)
                qe = run_qe(inp, outdir)
                point["qe"] = qe
                if qe:
                    print(
                        f"  QE  ecut={ecut:3.0f} Ry  E={qe['energy']:15.6f}  iters={qe['iterations']}"
                    )

        results.append(point)

    return results


def sweep_kmesh(skip_qe: bool, skip_av: bool, upf: str) -> list[dict]:
    """Energy vs k-point mesh."""
    results = []
    kmeshes = [1, 2, 3, 4, 5, 6, 8]

    for k in kmeshes:
        point = {"param": "kmesh", "value": k}

        if not skip_av:
            av = run_atomiverse(5.43, 30, k, upf, max_iter=80)
            point["atomiverse"] = av
            print(f"  AV  kmesh={k}  E={av['energy']:15.6f}  iters={av['iterations']}")

        if not skip_qe:
            with tempfile.TemporaryDirectory() as tmpdir:
                inp = os.path.join(tmpdir, "si.in")
                outdir = write_qe_input(
                    inp, a_lat=5.43, ecut_ry=30, kmesh=k, upf_path=upf
                )
                os.makedirs(outdir, exist_ok=True)
                qe = run_qe(inp, outdir)
                point["qe"] = qe
                if qe:
                    print(
                        f"  QE  kmesh={k}  E={qe['energy']:15.6f}  iters={qe['iterations']}"
                    )

        results.append(point)

    return results


def sweep_alat(skip_qe: bool, skip_av: bool, upf: str) -> list[dict]:
    """Energy vs lattice constant (EOS scan)."""
    results = []
    alats = [5.20, 5.30, 5.35, 5.40, 5.43, 5.45, 5.50, 5.55, 5.60, 5.70]

    for a in alats:
        point = {"param": "alat", "value": a}

        if not skip_av:
            av = run_atomiverse(a, 30, 4, upf, max_iter=80)
            point["atomiverse"] = av
            print(
                f"  AV  alat={a:.2f} A  E={av['energy']:15.6f}  iters={av['iterations']}"
            )

        if not skip_qe:
            with tempfile.TemporaryDirectory() as tmpdir:
                inp = os.path.join(tmpdir, "si.in")
                outdir = write_qe_input(
                    inp, a_lat=a, ecut_ry=30, kmesh=4, upf_path=upf
                )
                os.makedirs(outdir, exist_ok=True)
                qe = run_qe(inp, outdir)
                point["qe"] = qe
                if qe:
                    print(
                        f"  QE  alat={a:.2f} A  E={qe['energy']:15.6f}  iters={qe['iterations']}"
                    )

        results.append(point)

    return results


# ── Plotting (molplot — not raw matplotlib) ──


def _series_av_qe(data: list) -> list[dict]:
    """Build molplot line series for Atomiverse vs QE from a sweep list."""
    x_vals = [p["value"] for p in data]
    av = [p.get("atomiverse", {}).get("energy") for p in data]
    qe = [p.get("qe", {}).get("energy") for p in data]
    series: list[dict] = []
    if any(v is not None for v in av):
        series.append(
            {
                "id": "av",
                "label": "Atomiverse",
                "x": x_vals,
                "y": av,
                "mode": "lines+markers",
            }
        )
    if any(v is not None for v in qe):
        series.append(
            {
                "id": "qe",
                "label": "Quantum ESPRESSO",
                "x": x_vals,
                "y": qe,
                "mode": "lines+markers",
            }
        )
    return series


def _series_abs_diff(data: list) -> list[dict]:
    """|E_AV − E_QE| in eV (from Ry)."""
    xs: list[float] = []
    ys: list[float] = []
    for p in data:
        av_e = p.get("atomiverse", {}).get("energy")
        qe_e = p.get("qe", {}).get("energy")
        if av_e is None or qe_e is None:
            continue
        xs.append(p["value"])
        ys.append(abs(av_e - qe_e) * 13.6057)  # Ry → eV
    if not xs:
        return []
    return [
        {
            "id": "dE",
            "label": "|E_AV − E_QE|",
            "x": xs,
            "y": ys,
            "mode": "lines+markers",
        }
    ]


def plot_comparison(results: dict, save_path: str = ""):
    """Write comparison figures with molplot (Vega-Lite → paper renderer).

    Four panels are saved next to ``save_path`` (or as
    ``pw_qe_compare_*.png`` when ``save_path`` is empty). Requires
    ``molcrafts-molplot`` (``pip install -e './python[examples]'``).
    """
    import molplot

    molplot.use("molplot")

    base = Path(save_path) if save_path else Path("pw_qe_compare.png")
    stem = base.with_suffix("")  # drop .png / .json

    panels = [
        (
            "ecut",
            results.get("ecut", []),
            "ecut (Ry)",
            "E (Ry)",
            False,
            _series_av_qe,
        ),
        (
            "kmesh",
            results.get("kmesh", []),
            "k-mesh",
            "E (Ry)",
            False,
            _series_av_qe,
        ),
        (
            "alat",
            results.get("alat", []),
            "a (Å)",
            "E (Ry)",
            False,
            _series_av_qe,
        ),
        (
            "delta",
            results.get("ecut", []),
            "ecut (Ry)",
            "ΔE (eV) |E_AV − E_QE|",
            True,
            _series_abs_diff,
        ),
    ]

    written: list[str] = []
    for name, data, x_label, y_label, y_log, builder in panels:
        series = builder(data) if data else []
        if not series:
            continue
        fig, _ax = molplot.line(
            series,
            x_label=x_label,
            y_label=y_label,
            y_log=y_log,
            show_legend=True,
            height=360,
        )
        out = f"{stem}_{name}.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        written.append(out)

    if written:
        print("\nSaved molplot figures:")
        for path in written:
            print(f"  {path}")
    else:
        print("\n(no sweep data to plot)")


# ── Main ──


def main():
    parser = argparse.ArgumentParser(
        description="Compare Atomiverse PW-DFT vs Quantum ESPRESSO for Si diamond"
    )
    parser.add_argument("--skip-qe", action="store_true", help="Skip QE calculations")
    parser.add_argument(
        "--skip-av",
        "--skip-atomiverse",
        action="store_true",
        dest="skip_av",
        help="Skip Atomiverse calculations",
    )
    parser.add_argument(
        "--sweep",
        choices=["ecut", "kmesh", "alat", "all"],
        default="all",
        help="Which sweep to run",
    )
    parser.add_argument(
        "--upf",
        default="",
        help="Path to a norm-conserving Si UPF "
        f"(default: {DEFAULT_UPF.relative_to(REPO_ROOT)})",
    )
    parser.add_argument("--output", default="", help="Save plot to file")
    args = parser.parse_args()

    # Pseudopotential: required, norm-conserving, and shared by both codes.
    # There is no toy fallback — a bare valence-charge potential is not
    # representable in a plane-wave basis at any finite cutoff.
    upf = args.upf or str(DEFAULT_UPF)
    if not Path(upf).exists():
        print(f"Pseudopotential not found: {upf}")
        print("Pass --upf with a norm-conserving (ONCVPSP-style) Si UPF.")
        sys.exit(1)
    print(f"Using pseudopotential: {upf}")

    if args.skip_qe:
        print("Skipping Quantum ESPRESSO calculations\n")
    else:
        if not Path(QE_BIN).exists():
            print(f"QE pw.x not found at {QE_BIN}")
            print("Install QE or use --skip-qe")
            sys.exit(1)
        print(f"Using QE at: {QE_BIN}\n")

    results = {}

    if args.sweep in ("ecut", "all"):
        print("=" * 60)
        print("Sweep: Energy vs plane-wave cutoff")
        print("=" * 60)
        results["ecut"] = sweep_ecut(args.skip_qe, args.skip_av, upf)

    if args.sweep in ("kmesh", "all"):
        print("\n" + "=" * 60)
        print("Sweep: Energy vs k-point mesh")
        print("=" * 60)
        results["kmesh"] = sweep_kmesh(args.skip_qe, args.skip_av, upf)

    if args.sweep in ("alat", "all"):
        print("\n" + "=" * 60)
        print("Sweep: Energy vs lattice constant (EOS)")
        print("=" * 60)
        results["alat"] = sweep_alat(args.skip_qe, args.skip_av, upf)

    # Save results
    out_file = args.output or "pw_qe_results.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {out_file}")

    # Plot
    plot_path = (
        args.output.replace(".json", ".png") if args.output else "pw_qe_compare.png"
    )
    plot_comparison(results, save_path=plot_path)


if __name__ == "__main__":
    main()
