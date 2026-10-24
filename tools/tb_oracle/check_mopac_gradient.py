#!/usr/bin/env python3
"""Independently assess archived gradients using re-SCF MOPAC HOF differences.

Optional offline qualification only; never called by ctest. Displaced raw outputs
are retained in --workdir. The report includes every HOF used in the differences.
"""
import argparse
import json
import os
from pathlib import Path
import re
import subprocess

from extract_mopac_pin import EV_TO_KCAL_MOL


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("binary", type=Path)
    parser.add_argument("archive", type=Path)
    parser.add_argument("--workdir", required=True, type=Path)
    args = parser.parse_args()
    binary = args.binary.resolve()
    args.workdir.mkdir(parents=True, exist_ok=True)
    report = {"displacements_angstrom": [0.001, 0.0005, 0.00025], "cases": {}}
    pins = json.loads((args.archive / "pins.json").read_text())
    for name, pin in pins.items():
        gradients = []
        samples = []
        for h in report["displacements_angstrom"]:
            gradient = []
            for axis in range(3 * len(pin["elements"])):
                energies = []
                for sign in (-1, 1):
                    xyz = [p.copy() for p in pin["coordinates_angstrom"]]
                    xyz[axis // 3][axis % 3] += sign * h
                    stem = f"{name}-h{h}-a{axis}-s{sign}"
                    deck = args.workdir / f"{stem}.mop"
                    # Same SCF/model controls as pins; no gradient calculation needed.
                    lines = [f'{pin["method"]} 1SCF XYZ AUX(PRECISION=9) PRECISE SCFCRT=1.D-12',
                             "Independent gradient qualification", "Displaced re-SCF energy"]
                    lines += [el + " " + " ".join(f"{v:.12f} 1" for v in p)
                              for el, p in zip(pin["elements"], xyz)]
                    deck.write_text("\n".join(lines) + "\n")
                    with deck.with_suffix(".stdout.txt").open("w") as stdout:
                        subprocess.run([str(binary), str(deck.resolve())], check=True,
                                       env={**os.environ, "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"},
                                       stdout=stdout, stderr=subprocess.STDOUT)
                    aux = deck.with_suffix(".aux").read_text()
                    out = deck.with_suffix(".out").read_text()
                    if "END OF MOPAC PROGRAM" not in aux or "SCF FIELD WAS ACHIEVED" not in out:
                        raise ValueError(f"Failed displaced SCF: {deck}")
                    value = float(re.search(r"HEAT_OF_FORMATION:KCAL/MOL=([^\n]+)", aux)[1].replace("D", "E"))
                    energies.append(value)
                samples.append({"h_angstrom": h, "coordinate_index": axis,
                                "hof_minus_kcal_mol": energies[0], "hof_plus_kcal_mol": energies[1]})
                gradient.append((energies[1] - energies[0]) / (2 * h * EV_TO_KCAL_MOL))
            gradients.append(gradient)
        richardson_coarse = [(4 * fine - coarse) / 3 for coarse, fine in zip(gradients[0], gradients[1])]
        richardson = [(4 * fine - coarse) / 3 for coarse, fine in zip(gradients[1], gradients[2])]
        oracle_gradient = [-v for f in pin["forces_ev_angstrom"] for v in f]
        report["cases"][name] = {
            "samples": samples,
            "richardson_gradient_ev_angstrom": richardson,
            "max_richardson_refinement_ev_angstrom": max(abs(a - b) for a, b in zip(richardson_coarse, richardson)),
            "max_fine_vs_richardson_ev_angstrom": max(abs(a - b) for a, b in zip(gradients[2], richardson)),
            "max_pinned_gradient_vs_richardson_ev_angstrom": max(abs(a - b) for a, b in zip(oracle_gradient, richardson)),
        }
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
