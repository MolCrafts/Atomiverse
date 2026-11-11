"""Open-shell KS, including a fully polarized one-electron channel."""

from benchmarks.framework.specification import Method
from benchmarks.scf.dft import row

SPEC = Method(
    id="uks",
    rows=[
        row("UKS", xc, task, ["h2_plus", "o2_triplet"])
        for xc in ["LSDA", "PBE", "PBEsol", "RevPBE", "RPBE", "PBE0", "CAMB3LYP"]
        for task in ["energy", "forces"]
    ]
    + [
        row("UKS", xc, task, ["h2_plus", "o2_triplet"], basis="def2-SVP")
        for xc in ["PBE", "PBE0", "CAMB3LYP"]
        for task in ["energy", "forces"]
    ],
)

if __name__ == "__main__":
    from benchmarks.__main__ import method_main

    raise SystemExit(method_main(SPEC))
