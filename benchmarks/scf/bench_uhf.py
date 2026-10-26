"""Unrestricted HF with explicitly distinct MINAO and Hcore O2 branches."""
from benchmarks.framework.specification import Method
from benchmarks.scf.hf import row

SPEC = Method(id="uhf", rows=[
    row("UHF", "sto3g.minao.energy", "energy", "STO-3G", ["h2_plus", "o2_triplet"], guess="minao"),
    row("UHF", "sto3g.minao.forces", "forces", "STO-3G", ["h2_plus", "o2_triplet"], guess="minao"),
    row("UHF", "sto3g.hcore.energy", "energy", "STO-3G", ["o2_triplet"], guess="hcore"),
    row("UHF", "sto3g.hcore.forces", "forces", "STO-3G", ["o2_triplet"], guess="hcore"),
    row("UHF", "def2svp.energy", "energy", "def2-SVP", ["o2_triplet"], guess="minao"),
    row("UHF", "def2svp.forces", "forces", "def2-SVP", ["o2_triplet"], guess="minao"),
])

if __name__ == "__main__":
    from benchmarks.__main__ import method_main
    raise SystemExit(method_main(SPEC))
