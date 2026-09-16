# DFTB+ twin note — `dftb3-water-mio`

```text
oracle: dftb+
oracle_version: formula-Gaus2011
```

No host `dftb+` binary was available when this pin was taken (2026-07-27).
ATV uses static mio-**style** SK in `dftb_mio_tables.h`, not DFTB.org mio-1-1
files. Absolute Ha energy vs production DFTB+ is therefore not comparable yet.

When DFTB+ + matching SK land offline:

1. Build `hsd` input for H2O at the REFERENCE geometry with
   `ThirdOrderFull = Yes` (or diagonal DFTB3) and the same SCC tol.
2. Convert total energy to eV; write into
   `regressions/semiempirical/dftb3-water-mio/REFERENCE.md`.
3. Point regression `kPinEnergyEV` at that number only after SK fidelity
   matches (see queue slug `dftb2-3ob-organic` / energy-pin).

Until then the regression pins the **ATV Hamiltonian twin** energy and
asserts \(E_{\mathrm{DFTB3}}\neq E_{\mathrm{DFTB2}}\).
