"""All-electron canonical RMP2 through independent public engine calls."""
from benchmarks.framework.specification import Check, Method, Row
from benchmarks.scf.hf import eight_water_chain, row

_rhf = row("RHF", "sto3g.energy", "energy", "STO-3G", ["water"])
_rhf.systems[0].tiers.insert(0, "smoke")
_participants = [p for p in _rhf.participants if p.id != "gpu4pyscf.cuda"]
_cuda = next(p for p in _participants if p.id == "atv.cuda")
_cuda.available = False
_cuda.unavailable_reason = "Native CUDA RMP2 awaits a public Python binding"

SPEC = Method(id="rmp2", rows=[Row(
    id="rmp2.sto3g.energy",
    label="Canonical RMP2/STO-3G energy",
    capability="post-hf.rmp2.energy",
    task="energy",
    model={**_rhf.model, "correlation": "MP2", "frozen": 0},
    systems=[*_rhf.systems, eight_water_chain()],
    participants=_participants,
    reference=_rhf.reference,
    checks=[
        Check(observable=observable, unit="Ha", atol=1e-6,
              justification="Live PySCF RHF and canonical MP2, with the existing water-chain energy tolerance.")
        for observable in ("energy_hartree", "hf_energy_hartree", "correlation_energy_hartree",
                           "opposite_spin_energy_hartree", "same_spin_energy_hartree")
    ],
)])

if __name__ == "__main__":
    from benchmarks.__main__ import method_main
    raise SystemExit(method_main(SPEC))
