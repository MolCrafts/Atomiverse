# DFTB3 Γ-PBC analytic forces

```text
oracle: literature
oracle_version: Gaus, Cui, Elstner, J. Chem. Theory Comput. 7, 931 (2011),
  DOI 10.1021/ct100684s; lattice walk mirrors DFTB2
family_oracle: dftb+
```

**This pin is not `oracle: dftb+` and must not be labelled as such.** Two
named blockers:

1. **Mixed parameter set.** The run declares
   `d3.sk_set = cpu::semiempirical::DftbSkSet::MioSketch` and therefore pairs
   mio-1-1 SK/repulsive (`src/cpu/semiempirical/dftb3.cpp`) with 3ob-class
   Hubbard derivatives (`hubbard_du_ha = basis_data::k3obHubbardDerivsHa`).
   The engine `ATV_WARN`s. This is not a DFTB+-reproducible Hamiltonian.
2. **Missing \(\gamma^h\) X–H damping (\(\zeta=4.00\)).** Follow-on slug:
   `dftb-gamma-h-damping`.

The absolute energy below is an **ATV twin snapshot**, not an oracle pin.
The real gate is analytic nuclear forces vs re-SCC central FD of the
lattice-summed energy.

## Gate

Off-eq C-chain (r = 1.20 Å, \(L_x=2.8\) Å, \(L_y=L_z=12\) Å) so left/right
lattice shells differ. Analytic \(F\) vs re-SCC FD, tol **1×10⁻⁵ eV/Å**.
Net force ~0.

This geometry is **homonuclear**, so \(\Delta q\equiv 0\), \(E^{(3)}=0\), and
DFTB3 reduces to DFTB2 (the Γ control of `dftb2-multi-k-forces`). It still
exercises the lattice-image walk on the DFTB3 path.

## ATV twin snapshot (not an oracle pin)

Measured 2026-09-04, Release, `OMP_NUM_THREADS=1`:

```text
E = -93.1115376115 eV
F[0] = (-14.8265853692, 0, 0) eV/Å
F[1] = ( 14.8265853692, 0, 0) eV/Å
F0x analytic=-14.8265853692  fd=-14.8265872815  |d|=1.912e-06 eV/Å
```

## Perf

`perf_exempt: true` — no matched DFTB+ DFTB3 Hamiltonian to wall against.
