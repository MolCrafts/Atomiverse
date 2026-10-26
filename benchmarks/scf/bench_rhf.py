"""Closed-shell molecular HF: independent energy and force workloads."""
from benchmarks.framework.specification import Method
from benchmarks.scf.hf import row

_chain = row("RHF", "sto3g.water_chain16.energy", "energy", "STO-3G", ["water_chain16"], energy_atol=1e-6)
_chain.model["conv_tol"] = 1e-10
SPEC = Method(id="rhf", rows=[
    row("RHF", "sto3g.energy", "energy", "STO-3G", ["h2", "water"]),
    row("RHF", "sto3g.forces", "forces", "STO-3G", ["h2_stretched", "water"]),
    row("RHF", "631gs.energy", "energy", "6-31G*", ["water"]),
    row("RHF", "631gs.forces", "forces", "6-31G*", ["water"]),
    row("RHF", "def2svp.energy", "energy", "def2-SVP", ["water"]),
    row("RHF", "def2svp.forces", "forces", "def2-SVP", ["water"]),
    _chain,
])

if __name__ == "__main__":
    from benchmarks.__main__ import method_main
    raise SystemExit(method_main(SPEC))
