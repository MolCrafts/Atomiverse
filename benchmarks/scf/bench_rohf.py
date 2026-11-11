"""Restricted open-shell HF, including live native analytic-force oracles."""
from benchmarks.framework.specification import Method
from benchmarks.scf.hf import row

SPEC = Method(id="rohf", rows=[
    row("ROHF", "sto3g.energy", "energy", "STO-3G", ["h2_plus", "o2_triplet"], guess="minao"),
    row("ROHF", "sto3g.forces", "forces", "STO-3G", ["h2_plus", "o2_triplet"], guess="minao"),
    row("ROHF", "def2svp.energy", "energy", "def2-SVP", ["o2_triplet"], guess="minao"),
    row("ROHF", "def2svp.forces", "forces", "def2-SVP", ["o2_triplet"], guess="minao"),
])

if __name__ == "__main__":
    from benchmarks.__main__ import method_main
    raise SystemExit(method_main(SPEC))
