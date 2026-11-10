"""All-electron canonical RMP3 against PySCF RHF and NumPy spatial diagrams."""
from benchmarks.framework.specification import Check, Method, Row
from benchmarks.scf.hf import eight_water_chain, row

_rhf = row("RHF", "sto3g.energy", "energy", "STO-3G", ["water"])
_rhf.systems[0].tiers.insert(0, "smoke")
_cpu, _cuda, _oracle, _ = _rhf.participants
_cuda.available = False
_cuda.unavailable_reason = "Native CUDA RMP3 is not implemented"
_oracle.id = "pyscf.numpy.cpu"

SPEC = Method(id="rmp3", rows=[Row(
    id="rmp3.sto3g.energy",
    label="Canonical RMP3/STO-3G energy",
    capability="post-hf.rmp3.energy",
    task="energy",
    model={**_rhf.model, "correlation": "MP3", "frozen": 0},
    systems=[*_rhf.systems, eight_water_chain()],
    participants=[_cpu, _cuda, _oracle],
    reference=_oracle.id,
    checks=[
        Check(observable=name, unit="Ha", atol=1e-7 if name == "hf_energy_hartree" else 1e-6,
              justification="Fresh PySCF RHF and independent spatial MP3 diagrams at the existing MP3 pin tolerances.")
        for name in ("energy_hartree", "hf_energy_hartree", "correlation_energy_hartree",
                     "second_order_energy_hartree", "third_order_energy_hartree")
    ],
)])

if __name__ == "__main__":
    from benchmarks.__main__ import method_main
    raise SystemExit(method_main(SPEC))
