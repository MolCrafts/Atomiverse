#!/usr/bin/env python3
"""Extract offline MOPAC 23 AUX/output pairs; never launch an oracle from tests."""
import argparse
import json
import math
from pathlib import Path
import re

# OpenMOPAC v23.2.5 conref_C.F90, CODATA-2018 row; OLDCON is not used.
EV_TO_KCAL_MOL = 23.060547830619029


def extract(aux_path: Path, out_path: Path) -> dict:
    aux = aux_path.read_text()
    out = out_path.read_text()
    if "END OF MOPAC PROGRAM" not in aux or "SCF FIELD WAS ACHIEVED" not in out:
        raise ValueError(f"Incomplete or unconverged MOPAC output: {aux_path}")

    def scalar(key: str) -> str:
        matches = re.findall(r"^\s*" + re.escape(key) + r"=([^\n]+)", aux, re.M)
        if len(matches) != 1:
            raise ValueError(f"Expected exactly one {key} in {aux_path}")
        return matches[0].strip().strip('"')

    def array(key: str, numeric: bool = True) -> list:
        match = re.search(r"^\s*" + re.escape(key) + r"\[(\d+)\]=(.*?)(?=\n\s*[A-Z_][A-Z_.:]*(?:\[|=)|\Z)", aux, re.M | re.S)
        if not match:
            raise ValueError(f"Missing {key} in {aux_path}")
        tokens = match[2].split()
        if len(tokens) != int(match[1]):
            raise ValueError(f"Wrong array length for {key} in {aux_path}")
        return [float(v.replace("D", "E")) for v in tokens] if numeric else tokens

    def triples(values: list) -> list:
        if len(values) % 3:
            raise ValueError("Cartesian array length is not a multiple of three")
        return [values[i:i + 3] for i in range(0, len(values), 3)]

    elements = array("ATOM_EL", numeric=False)
    xyz = array("ATOM_X:ANGSTROMS")
    if xyz != array("ATOM_X_OPT:ANGSTROMS"):
        raise ValueError("Geometry changed during the oracle evaluation")
    charges = array("ATOM_CHARGES")
    gradient = array("GRADIENTS:KCAL/MOL/ANGSTROM")
    if len(xyz) != 3 * len(elements) or len(gradient) != len(xyz) or len(charges) != len(elements):
        raise ValueError("Atom, coordinate, charge and gradient counts disagree")
    if not all(math.isfinite(x) for x in xyz + charges + gradient):
        raise ValueError("Non-finite scientific output")
    energies = re.findall(r"TOTAL ENERGY\s*=\s*([-+\d.]+) KCAL/MOL", out)
    if len(energies) != 1:
        raise ValueError("Expected one total energy in kcal/mol; run with DISP")
    energy_kcal = float(energies[0])
    return {
        "method": scalar("METHOD"),
        "oracle_version": scalar("MOPAC_VERSION"),
        "keywords": scalar("KEYWORDS"),
        "elements": elements,
        "coordinates_angstrom": triples(xyz),
        "total_energy_kcal_mol_printed": energy_kcal,
        "total_energy_ev": energy_kcal / EV_TO_KCAL_MOL,
        "total_energy_ev_print_rounding_bound": 0.5e-5 / EV_TO_KCAL_MOL,
        "heat_of_formation_kcal_mol": float(scalar("HEAT_OF_FORMATION:KCAL/MOL").replace("D", "E")),
        "mulliken_charges_e": charges,
        "gradient_kcal_mol_angstrom": triples(gradient),
        "forces_ev_angstrom": triples([-x / EV_TO_KCAL_MOL for x in gradient]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path, help="Directory containing inputs/ and outputs/")
    args = parser.parse_args()
    rows = {}
    for deck in sorted((args.archive / "inputs").glob("*.mop")):
        rows[deck.stem] = extract(args.archive / "outputs" / f"{deck.stem}.aux.txt",
                                  args.archive / "outputs" / f"{deck.stem}.out.txt")
    if not rows:
        parser.error("No MOPAC input decks found")
    print(json.dumps(rows, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
