#!/usr/bin/env python3
"""Convert the benchmark prmtop to the LAMMPS data + force-field pair.

LAMMPS and Atomiverse both read the files this writes; OpenMM reads the
prmtop itself. The conversion is molpy's (``AmberPrmtopForcefieldReader`` ->
``LammpsForcefieldWriter`` / ``write_lammps_data``), which maps Amber terms to
``bond/angle harmonic``, ``dihedral fourier`` (every periodic term of a
torsion on one row), ``improper cvff`` (Amber atom order kept) and
``pair_modify mix arithmetic``.

Two adjustments, both for a bit-faithful three-way comparison:

* 1-4 weights are written as ``special_bonds amber`` (Coulomb 5/6 exactly)
  instead of the rounded ``coul 0 0 0.833333`` triple.
* Coefficient rows use numeric type ids (from the data file's
  ``* Type Labels`` sections) and 10 decimals, so no reader depends on type
  labels and no parameter is truncated below the prmtop's own precision.

Usage::

    python prmtop_to_lammps.py DIR    # reads DIR/system.{prmtop,inpcrd}
                                      # writes DIR/system.{data,ff}

Requires the ``molcrafts-molpy`` python package.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

COEFF_COMMANDS = {
    "pair_coeff": ("atom", 2),
    "bond_coeff": ("bond", 1),
    "angle_coeff": ("angle", 1),
    "dihedral_coeff": ("dihedral", 1),
    "improper_coeff": ("improper", 1),
}
LABEL_SECTION = re.compile(r"^(Atom|Bond|Angle|Dihedral|Improper) Type Labels$")


def read_type_labels(data: Path) -> dict[str, dict[str, int]]:
    """Return {kind: {label: id}} from the data file's label sections."""
    labels: dict[str, dict[str, int]] = {}
    kind = None
    for raw in data.read_text().splitlines():
        line = raw.strip()
        match = LABEL_SECTION.match(line)
        if match:
            kind = match.group(1).lower()
            labels[kind] = {}
            continue
        if kind is None or not line:
            continue
        tokens = line.split()
        if len(tokens) != 2 or not tokens[0].isdigit():
            kind = None  # next section header
            continue
        labels[kind][tokens[1]] = int(tokens[0])
    return labels


def numeric_ff(text: str, labels: dict[str, dict[str, int]]) -> str:
    """Rewrite coefficient rows from type labels to numeric ids."""
    out = []
    for line in text.splitlines():
        tokens = line.split()
        if tokens and tokens[0] in COEFF_COMMANDS:
            kind, count = COEFF_COMMANDS[tokens[0]]
            table = labels[kind]
            for k in range(1, 1 + count):
                if tokens[k] not in table:
                    raise ValueError(f"{tokens[0]}: unknown {kind} label "
                                     f"'{tokens[k]}'")
                tokens[k] = str(table[tokens[k]])
            line = " ".join(tokens)
        out.append(line)
    return "\n".join(out) + "\n"


def convert(directory: Path) -> None:
    import molpy.io as mio
    import molpy.io.amber as amber
    import molpy.io.lammps as lammps

    ff, frame = amber.AmberPrmtopForcefieldReader().read_system(
        directory / "system.prmtop")
    frame = mio.read_inpcrd(directory / "system.inpcrd", frame)
    data = directory / "system.data"
    mio.write_lammps_data(data, frame)
    labelled = directory / "system.labels.ff"
    lammps.LammpsForcefieldWriter(frame, precision=10, skip_pair_style=True,
                                  skip_special_bonds=True,
                                  skip_units=True).write(labelled, ff)
    text = numeric_ff(labelled.read_text(), read_type_labels(data))
    (directory / "system.ff").write_text("special_bonds amber\n" + text)
    labelled.unlink()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("directory", type=Path)
    convert(ap.parse_args().directory)


if __name__ == "__main__":
    main()
