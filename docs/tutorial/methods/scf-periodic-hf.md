# Periodic Hartree–Fock

> The regression runner has been removed. Regression commands below are historical; case sources and oracle records are retained as archived evidence, not current executable checks.

Gaussian-basis Hartree–Fock in a periodic cell. Two public paths exist: the
Γ-point `cpu::scf::RHF{basis}` with `periodic = true` (a truncated real-space
lattice; section `periodic-rhf-gamma`, H₂ / STO-3G in a 20 Å cube, oracle
record in the case `REFERENCE.md`), and the Monkhorst–Pack k-point methods
`cpu::scf::KRHF{basis, k_mesh, charge, tol, max_iter}` and
`cpu::scf::KUHF{basis, k_mesh, n_alpha, n_beta, charge, tol, max_iter}`,
documented below. `cpu::scf::KRHF` and `cpu::scf::KUHF` are each
`cpu::Driver<scf::Phase>` plus `cpu::HF` (not `MeanFieldMethod`).
`cuda::scf::KRHF` is `cuda::Driver<scf::Phase>`, `cuda::DriverNode`, and
`cuda::HF`. There is no device KUHF. No section of the case runs the device
path; both sections run on CPU only.

Case: `ctest -R '^scf/periodic-hf$'` — sources and oracle record in [`regressions/scf/periodic-hf/`](../../../regressions/scf/periodic-hf/). One section of the case per `##` heading below: `atv_regressions --case scf/periodic-hf --section <name>`.

## periodic-kpoint-hf — Periodic k-point Hartree–Fock {#periodic-kpoint-hf}

Restricted and unrestricted Hartree–Fock with a Monkhorst–Pack mesh on
an orthorhombic cell. The Coulomb and exchange matrices are the PySCF
FFT density-fitting kernels (discrete Fourier transform of the AO
products, Coulomb kernel `4π/G²`, Ewald probe charge on the exchange
`G = 0` term). The nuclear repulsion is the Ewald sum.

### Call

`KRHF` and `KUHF` are `Driver<scf::Phase>` presets, the same shape as
`cpu::md::MD`. The constructor binds stages `fock`, `mix`, `diagonalize`,
`density`, and `check`. Mix is the identity (no DIIS). The absolute energy
change is published on `ConvergenceMetrics`; a Criterion judges it and
`Driver::converged()` is that verdict. The method does not assign
convergence. Basis, mesh, charge, `tol`, and `max_iter` are constructor
arguments. `KUHF` also takes mesh-total `n_alpha` and `n_beta`
(`KUHF.nelec`; spin is not scaled by the number of k-points). Its fock
stage builds `J = Ja+Jb` and `vhf = J−K` per spin (`KUHF.get_veff`);
occupation fill is 1 (`get_occ`).

Frame in is an orthorhombic cell: positions [Å], species in `kAtom_type`.
`run(Frame&)` overwrites `kEnergy` and `kEnergyElectronic` [eV].
`run(const Frame&)` writes nothing. `energy()` is the total in Hartree,
nuclear repulsion included. `iterations()` is the count of that run.
`forces(Frame&)` throws `krhf_nuclear_forces`. An outer driver that mounts
the method with `on(phase).add` accumulates energy instead of overwriting it.
Shells with l > 0 throw `krhf_ao_lgt0`. CPU `requires_full_frame()` is true.
`KUHF::forces` throws `kuhf_nuclear_forces`. There is no device KUHF.

```cpp
cpu::Frame frame = /* orthorhombic cell, positions in Å, species in kAtom_type */;
cpu::scf::KRHF krhf{cpu::scf::BasisSet::STO3G, /*k_mesh=*/{2, 1, 1},
                    /*charge=*/0, /*tol=*/1e-10, /*max_iter=*/40};
krhf.run(frame);                              // overwrite kEnergy, kEnergyElectronic [eV]
double e_ev = frame.meta.get(kEnergy);        // eV
double e_ha = krhf.energy();                  // Hartree
bool ok = krhf.converged();                   // Driver verdict

// Mesh totals, not a per-k copy. Triplet H₂ with cell.spin = 2 on a
// 2-point mesh is Nα = 3, Nβ = 1 (PySCF KUHF.nelec).
cpu::scf::KUHF kuhf{cpu::scf::BasisSet::STO3G, {2, 1, 1},
                    /*n_alpha=*/3, /*n_beta=*/1, 0, 1e-10, 40};
kuhf.run(frame);

// Same entries on device: cuda::Driver<scf::Phase> + DriverNode + HF.
// Coulomb, exchange, Fock, zheevd, occupation, density stay on device.
// l > 0 throws (`krhf_ao_lgt0`). There is no device KUHF.
cuda::scf::KRHF gpu{cpu::scf::BasisSet::STO3G, {2, 1, 1}};
gpu.run(frame);                               // same Frame overwrite [eV]
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
