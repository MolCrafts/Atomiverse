"""Closed-shell KS energies and complete analytic molecular forces."""

from benchmarks.framework.specification import Method
from benchmarks.scf.dft import row

SPEC = Method(
    id="rks",
    rows=[
        *[
            row("RKS", xc, "energy", ["water"])
            for xc in [
                "LSDA",
                "PBE",
                "PBEsol",
                "RevPBE",
                "RPBE",
                "BLYP",
                "B3LYP",
                "PBE0",
                "CAMB3LYP",
            ]
        ],
        *[
            row("RKS", xc, "forces", ["h2_stretched", "water"])
            for xc in ["PBE", "PBE0", "CAMB3LYP"]
        ],
        *[
            row("RKS", xc, "forces", ["water_asymmetric"])
            for xc in ["PBEsol", "RevPBE", "RPBE"]
        ],
        *[
            row(
                "RKS",
                xc,
                task,
                ["water", "water_asymmetric"] if task == "forces" else ["water"],
                basis="def2-SVP",
            )
            for xc in ["PBE", "PBE0", "CAMB3LYP"]
            for task in ["energy", "forces"]
        ],
    ],
)

if __name__ == "__main__":
    from benchmarks.__main__ import method_main

    raise SystemExit(method_main(SPEC))
