"""Real GHF energies: H2 collapse and genuinely noncollinear frustrated H3.

PySCF GHF.nuc_grad_method raises NotImplementedError. No custom derivative is
presented as a native independent force oracle; that catalog gap stays visible.
"""
from benchmarks.framework.specification import Method
from benchmarks.scf.hf import row

SPEC = Method(id="ghf", rows=[
    row("GHF", "sto3g.collapse.energy", "energy", "STO-3G", ["h2"]),
    row("GHF", "sto3g.noncollinear.energy", "energy", "STO-3G", ["h3_triangle"], neel=True),
    row("GHF", "def2svp.collapse.energy", "energy", "def2-SVP", ["h2"]),
])

if __name__ == "__main__":
    from benchmarks.__main__ import method_main
    raise SystemExit(method_main(SPEC))
