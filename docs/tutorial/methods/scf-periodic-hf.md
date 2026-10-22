# Periodic Hartree–Fock

Gaussian-basis Hartree–Fock in a periodic cell. Two public paths exist: the
Γ-point `cpu::scf::RHF{basis}` with `periodic = true` (a truncated real-space
lattice; section `periodic-rhf-gamma`, H₂ / STO-3G in a 20 Å cube, oracle
record in the case `REFERENCE.md`), and the Monkhorst–Pack k-point methods
`cpu::scf::KRHF{basis, k_mesh, charge, tol, max_iter}` and
`cpu::scf::KUHF{basis, k_mesh, n_alpha, n_beta, charge, tol, max_iter}`,
documented below. A device `cuda::scf::KRHF` exists; no section of the case
runs it, and both sections run on CPU only.

Case: `ctest -R '^scf/periodic-hf$'` — sources and oracle record in [`regressions/scf/periodic-hf/`](../../../regressions/scf/periodic-hf/). One section of the case per `##` heading below: `atv_regressions --case scf/periodic-hf --section <name>`.

## periodic-kpoint-hf — Periodic k-point Hartree–Fock {#periodic-kpoint-hf}

Restricted and unrestricted Hartree–Fock with a Monkhorst–Pack mesh on
an orthorhombic cell. The Coulomb and exchange matrices are the PySCF
FFT density-fitting kernels (discrete Fourier transform of the AO
products, Coulomb kernel `4π/G²`, Ewald probe charge on the exchange
`G = 0` term). The nuclear repulsion is the Ewald sum.

```cpp
cpu::Frame frame = /* orthorhombic cell, positions in Å */;
cpu::scf::KRHF krhf{cpu::scf::BasisSet::STO3G, /*k_mesh=*/{2, 1, 1},
                    /*charge=*/0, /*tol=*/1e-10, /*max_iter=*/40};
krhf.run(frame);                 // kEnergy and kEnergyElectronic [eV]
double e_ha = krhf.energy();     // Hartree

// Mesh totals, not a per-k copy. Triplet H₂ with cell.spin = 2 on a
// 2-point mesh is Nα = 3, Nβ = 1 (PySCF KUHF.nelec).
cpu::scf::KUHF kuhf{cpu::scf::BasisSet::STO3G, {2, 1, 1},
                    /*n_alpha=*/3, /*n_beta=*/1, 0, 1e-10, 40};
kuhf.run(frame);

// Restricted SCF on the device. The step (Coulomb, exchange, Fock,
// diagonalize, density) stays on device; l > 0 throws (`krhf_ao_lgt0`).
// There is no device KUHF.
cuda::scf::KRHF gpu{cpu::scf::BasisSet::STO3G, {2, 1, 1}};
gpu.run(frame);
```

`k_mesh = {1,1,1}` is the single Γ point of this kernel (not the
truncated real-space lattice behind `RHF::periodic`). Shells with
angular momentum above s throw (`krhf_ao_lgt0`). `forces` throws
(`krhf_nuclear_forces`) — there is no analytic nuclear gradient on
this path. `provides_force()` is false.

H₂ / STO-3G in a 4 Bohr cube (shorter than the STO-3G pair reach) has
PySCF total energies −1.4471748215213558 Ha for `KRHF` with
`k_mesh = {2,1,1}`, and −0.7658470471886634 Ha for triplet `KUHF` on that
mesh (`cell.spin = 2`, Nα = 3, Nβ = 1). The section also gates `KRHF` with
`k_mesh = {1,1,2}` (along the bond) at −1.3541438798330374 Ha. All pins are
PySCF 2.14.0 `pyscf.pbc.scf` (default `FFTDF`, `exxdiv='ewald'`), gated at
`|E − E_pin| ≤ 1e-6` Ha.
