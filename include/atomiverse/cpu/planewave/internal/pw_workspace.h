/// @file cpu/planewave/internal/pw_workspace.h
/// @brief PWWorkspace — facade-private scratch for the plane-wave SCF.
///
/// Owned by `cpu::pw::PW` (the inner solver), never by `cpu::Frame`
/// and never threaded through Build/Launch contexts. Mirrors the gaussian
/// SCF precedent: the workspace is the solver's private state, and the
/// step helpers (`internal::PoissonSolver`, `XCBuilder`, `VeffBuilder`,
/// `DavidsonSolver`, `DensityBuilder`, `ConvergenceCheck`) receive it as
/// an explicit parameter.
#pragma once

#include "atomiverse/core/precision.h"
#include "atomiverse/cpu/planewave/internal/fft.h"
#include "atomiverse/cpu/planewave/internal/pw_operators.h"
#include <complex>
#include <memory>
#include <span>
#include <vector>
#include <xtensor/containers/xarray.hpp>

namespace ATV {
namespace cpu {
namespace pw {
namespace internal {

struct PWWorkspace {
    int n_g = 0;
    int nr = 0, nc = 0; // real and complex FFT grid sizes
    int fft_nx = 0, fft_ny = 0, fft_nz = 0;
    const GVectorList* gvec = nullptr; // current G-sphere (density or k)

    // Active basis and static ionic data are borrowed from setup.
    KBProjectorView kb_proj;
    AtomicOrbitalView atom_wfc;
    std::span<const Vec3> atom_pos_bohr;
    std::span<const std::complex<AccReal>> V_loc_G;

    void bind(PWKPointView point) {
        gvec = &point.modes;
        n_g = gvec->count();
        kb_proj = point.projectors;
        atom_wfc = point.orbitals;
    }

    PWKPointView point() const { return {*gvec, kb_proj, atom_wfc}; }

    PWProjectorView projectors() const {
        return {*gvec, kb_proj, atom_pos_bohr, omega};
    }

    PWDensityView density() const {
        return {{rho_G.data(), rho_G.size()}, rho_spin_G, nspin};
    }

    // FFT plan for r2c / c2r transforms
    std::unique_ptr<FFTPlan3D> fft;

    xt::xarray<std::complex<AccReal>> rho_G;     // charge density in G-space
    xt::xarray<std::complex<AccReal>> rho_G_old; // previous iteration density
    xt::xarray<std::complex<AccReal>> V_H_G;     // Hartree potential
    xt::xarray<std::complex<AccReal>> V_xc_G;    // XC potential (restricted)
    xt::xarray<std::complex<AccReal>> V_eff_G;   // total effective potential
    xt::xarray<AccReal> rho_r;                   // real-space density
    xt::xarray<AccReal> V_xc_r;                  // real-space XC potential
    xt::xarray<AccReal> eigenvalues;             // per-band eigenvalues
    xt::xarray<AccReal> V_eff_r; // V_eff on real-space grid [Hartree]

    // Wavefunctions: wfc[b][ig] for band b, G-vector ig
    xt::xarray<std::complex<AccReal>> wfc;

    // ── Collinear spin channel (nspin=2); empty when restricted ─────────
    int nspin = 1; ///< 1 = restricted, 2 = collinear spin-polarized.
    /// Spin densities ρ↑, ρ↓ in G-space (size 2 when nspin==2).
    std::vector<xt::xarray<std::complex<AccReal>>> rho_spin_G;
    std::vector<xt::xarray<std::complex<AccReal>>> rho_spin_G_old;
    /// Spin XC potentials Vxc↑, Vxc↓ in G-space.
    std::vector<xt::xarray<std::complex<AccReal>>> V_xc_spin_G;
    /// Real-space Vxc for the active spin channel (Davidson / Veff).
    xt::xarray<AccReal> V_xc_spin_r[2];
    /// Wavefunctions per spin (owned copies after each spin solve when needed).
    std::vector<xt::xarray<std::complex<AccReal>>> wfc_spin;
    std::vector<xt::xarray<AccReal>> eigenvalues_spin;
    /// Double-counting ∫(ρ↑ Vxc↑ + ρ↓ Vxc↓) [Hartree] for spin energy.
    AccReal e_xc_int_spin = AccReal(0);

    // Cell volume [Bohr³] for real-space integration
    AccReal omega = AccReal(0);

    // Accumulated XC energy [Hartree]
    AccReal E_xc = AccReal(0);

    // Pulay mixer history (managed by internal::DensityBuilder)
    std::vector<xt::xarray<std::complex<AccReal>>> pulay_res;
    std::vector<xt::xarray<std::complex<AccReal>>> pulay_rho;
    int pulay_depth = 5;
    AccReal mix_beta = AccReal(0.4);
};

} // namespace internal
} // namespace pw
} // namespace cpu
} // namespace ATV
