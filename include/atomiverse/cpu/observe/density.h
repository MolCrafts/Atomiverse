/// @file cpu/observe/density.h
/// @brief Electron density grid evaluation on the CPU (STO-3G basis).
///
/// Called by cuda::DensityRecorder after async D2H of the density matrix P.
/// The CUDA component owns no computation -- it lives here.
///
/// The implementation resolves shells from the canonical auto-generated
/// STO-3G table (`scf/basis_data/sto3g.h`) via `scf::ShellList`
/// and evaluates basis functions with the shared `scf::eval_basis`
/// (`scf/basis_eval.h`) -- the same path the cube writers use. Like the
/// cube writers, the implementation is compiled only under
/// `ATV_BUILD_HF` (`ATV_HAS_HF`); its sole consumer,
/// `cuda::DensityRecorder`, is gated the same way.
///
/// **Phase 1 note:** unlike `cpu::observer::RDF`, this is **not** wrapped as
/// a `cpu::AsyncComponentBase` subclass. The evaluation requires a
/// density matrix P from SCF, which is not a `cpu::Frame` field and is not
/// carried in the default `FrameSnapshot`. The natural async-observer
/// integration point is an SCF-driver observer wrapper that uses
/// `FrameSnapshot::custom_payload` to shuttle P from the SCF solver to the
/// analysis thread. That hook depends on the concrete Gaussian method
/// (`cpu::scf::RHF` / `UHF` / `RKS` / `UKS`) being extended to publish P into
/// custom_payload, plus the matching `Publish`. Deferred to Phase 1b
/// or to the AIMD density-observation workstream.
///
/// @see cuda::DensityRecorder (cuda/recorder/density_recorder.cuh)
/// @see .claude/specs/parallel-md-engine.md  § B.1 custom_payload
#pragma once

#include <vector>

namespace ATV {
namespace cpu {

// -- Result type ---------------------------------------------------------------

/// @brief Uniform real-space grid of electron density rho(r).
///
/// The grid is axis-aligned and non-periodic.
/// Indexing: rho[ix * ny * nz + iy * nz + iz]  (C order).
struct DensityGrid {
    int nx = 0, ny = 0, nz = 0;    ///< Grid dimensions.
    double ox = 0, oy = 0, oz = 0; ///< Grid origin [Angstrom].
    double lx = 0, ly = 0, lz = 0; ///< Box extent [Angstrom].
    std::vector<double> rho;       ///< Density values [nx*ny*nz, e/Bohr^3].

    bool empty() const { return rho.empty(); }
};

// -- Free function -------------------------------------------------------------

/// @brief Evaluate SCF electron density on a uniform grid (canonical STO-3G).
///
/// rho(r) = sum_{mu,nu} P_{mu,nu} phi_mu(r) phi_nu(r)
///
/// The grid bounds are determined from atom positions +/- padding.
/// Element coverage follows the canonical STO-3G table
/// (`scf/basis_data/sto3g.h`, currently H through O, s and p shells);
/// returns an empty DensityGrid (rho.empty() == true) if any atom's Z
/// has no entry there, or if `n_basis` does not match the STO-3G AO
/// count of the geometry.
///
/// @param type_id   Atomic numbers [n_atoms].
/// @param x         Atom x positions [n_atoms, Angstrom].
/// @param y         Atom y positions [n_atoms, Angstrom].
/// @param z         Atom z positions [n_atoms, Angstrom].
/// @param P         Density matrix [n_basis * n_basis], row-major (P is symmetric).
/// @param n_basis   Number of STO-3G basis functions (Cartesian AOs).
/// @param nx        Grid points along x.
/// @param ny        Grid points along y.
/// @param nz        Grid points along z.
/// @param padding   Extra padding around atom bounds [Angstrom].
/// @return Populated DensityGrid (rho in e/Bohr^3), or empty on failure.
DensityGrid eval_density_grid(const std::vector<int>& type_id,
                              const std::vector<double>& x,
                              const std::vector<double>& y,
                              const std::vector<double>& z, const double* P,
                              int n_basis, int nx, int ny, int nz,
                              double padding);

} // namespace cpu
} // namespace ATV
