#!/usr/bin/env python3
"""Parameterize ethanol with native GAFF2 and AM1-BCC, without AmberTools.

Run with matching molpy and Atomiverse native builds. The input molecule is
read-only to both models; charge assignment returns an owned array, and typing
returns a graph copy. AM1 uses the current geometry without optimizing it.
"""

from pathlib import Path

import atomiverse as atv
import molpy as mp


def main() -> None:
    molecule = mp.io.smiles.read_smiles("CCO").to_atomistic()
    molecule, _ = mp.conformer.Conformer(seed=7).generate(molecule)

    charge_model = atv.AM1BCCCharges(method=atv.AM1(charge=0), correction="bcc")
    charges = charge_model.assign(molecule)

    # Antechamber atom types first, then the GAFF2 bonded terms on them.
    typed = mp.ff.typifier.AtdTypifier(parameter_set="gaff2").typify(molecule)
    typifier = mp.ff.typifier.GaffTypifier(parameter_set="gaff2")
    typed = typifier.typify(typed)
    forcefield = typifier.forcefield()

    # MOL2 names atoms C1, H7, ... from the element when the frame has none.
    frame = typed.to_frame()
    frame["atoms"].insert("charge", charges)
    mp.io.write_mol2(Path("ethanol.mol2"), frame)
    mp.io.amber.AmberFrcmodWriter().write(Path("ethanol.frcmod"), forcefield)
    print(
        f"{forcefield.name}: {len(charges)} atoms, total charge {charges.sum():+.8f} e"
    )


if __name__ == "__main__":
    main()
