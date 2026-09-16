/// @file cpu/vibration/hessian_workspace.h
/// @brief cpu::vib::HessianWorkspace — shared scratch for the four
///        numerical-Hessian structural components.
///
/// The four numerical-Hessian components (`ReferenceGeometry`,
/// `DisplaceStep`, `HessianAccumulator`, `Symmetrize`) cooperate on one
/// `[3N,3N]` matrix plus the equilibrium-geometry cache. The driver
/// (`cpu::vib::Hessian`) owns one `HessianWorkspace` and hands each
/// component a non-owning pointer at construction — exactly as the
/// driver owns its component set today. No state lives on `cpu::Frame`.
#pragma once

#include <vector>

#include <xtensor/containers/xtensor.hpp>

namespace ATV {
namespace cpu {
namespace vib {

/// @brief Finite-difference step size in Angstrom.
///
/// Applied directly in the frame's Angstrom coordinates. The equivalent
/// in Bohr is `1.889726e-3` (non-binding note); the binding value is
/// `1.0e-3 Angstrom`. Central differencing has `O(h^2)` truncation
/// error with leading coefficient `g'''/6`.
inline constexpr double kHessianStepAngstrom = 1.0e-3;

/// @brief Shared scratch for the numerical-Hessian components.
///
/// Sized once by the driver in `build()` from `N = frame.nlocal()`.
/// Lifetime: owned by `cpu::vib::Hessian`; components hold raw pointers.
struct HessianWorkspace {
    /// @brief Number of atoms `N`.
    int n_atoms = 0;

    /// @brief Equilibrium Cartesian coordinates `R0`, flattened length `3N`
    ///        (atom-major: `[x0,y0,z0, x1,y1,z1, ...]`), in Angstrom.
    std::vector<double> r0;

    /// @brief True once `ReferenceGeometry` has cached `R0`.
    bool r0_ready = false;

    /// @brief Dense Hessian under construction, shape `[3N,3N]`, eV/Angstrom^2.
    xt::xtensor<double, 2> hessian;

    /// @brief Force vector from the most recent `+h` evaluation of the
    ///        current coordinate, flattened length `3N` (eV/Angstrom).
    ///        `HessianAccumulator` stores the `+h` forces here, then pairs
    ///        them with the next `-h` forces to write one Hessian row.
    std::vector<double> f_plus;

    // ── Vibrational-frequency link (phase4-vibrations-02) ──────────────
    //
    // The `MassWeighting` and `HessianDiagonalizer` components cooperate on the
    // fields below during the finalize pass, after `Symmetrize`.

    /// @brief Mass-weighted Hessian `H^mw` in atomic units
    ///        (Hartree/Bohr^2/amu), shape `[3N,3N]`. `MassWeighting`
    ///        derives it from `hessian` without mutating `hessian` — so
    ///        `HessianResult.hessian` keeps the raw link-1 eV/Angstrom^2
    ///        matrix. `HessianDiagonalizer` consumes this buffer.
    xt::xtensor<double, 2> hessian_mw;

    /// @brief True once `MassWeighting` has populated `hessian_mw`.
    ///        Guards `HessianDiagonalizer` against running on an empty buffer.
    bool mass_weighted = false;

    /// @brief Per-atom masses [amu] used for mass-weighting, length `N`.
    ///        Populated by `MassWeighting` from `mass_amu(frame type)`.
    std::vector<double> mass_amu;

    /// @brief Vibrational wavenumbers [cm^-1] produced by `HessianDiagonalizer`,
    ///        length `3N-6` / `3N-5`. Negative for imaginary modes.
    xt::xtensor<double, 1> frequencies;

    /// @brief Normal-mode eigenvectors `[3N, n_modes]` produced by
    ///        `HessianDiagonalizer` (columns lifted to the full Cartesian basis).
    xt::xtensor<double, 2> normal_modes;

    /// @brief Imaginary-mode count produced by `HessianDiagonalizer`.
    int n_imaginary = 0;

    /// @brief Linear-vs-nonlinear classification produced by `HessianDiagonalizer`.
    bool is_linear = false;

    // ── Thermochemistry link (phase4-vibrations-03) ────────────────────
    //
    // Written by `cpu::vib::Thermochem` during the `thermochem` pass;
    // `cpu::vib::Hessian::run()` copies them into `HessianResult`. All in
    // Hartree / Hartree-per-Kelvin, matching PySCF `thermo()`.

    /// @brief Harmonic zero-point energy [Hartree].
    double zpe = 0.0;

    /// @brief Total internal energy `U` [Hartree].
    double u_tot = 0.0;

    /// @brief Total enthalpy `H` [Hartree].
    double h_tot = 0.0;

    /// @brief Total entropy `S` [Hartree/K].
    double s_tot = 0.0;

    /// @brief Electronic entropy component [Hartree/K].
    double s_elec = 0.0;

    /// @brief Translational entropy component (Sackur-Tetrode) [Hartree/K].
    double s_trans = 0.0;

    /// @brief Rotational entropy component (rigid-rotor) [Hartree/K].
    double s_rot = 0.0;

    /// @brief Vibrational entropy component (harmonic) [Hartree/K].
    double s_vib = 0.0;

    /// @brief Total Gibbs free energy `G` [Hartree].
    double g_tot = 0.0;

    /// @brief Thermal Gibbs correction `G_tot - E0` [Hartree].
    double g_thermal = 0.0;

    /// @brief True once `Thermochem` has populated the thermochem fields.
    bool thermo_ready = false;

    /// @brief Allocate all buffers for an `N`-atom system. Called once
    ///        from the driver's `build()`; no allocation occurs in any
    ///        component `launch()`.
    /// @param n Number of atoms.
    void allocate(int n) {
        n_atoms = n;
        const int dim = 3 * n;
        r0.assign(dim, 0.0);
        r0_ready = false;
        hessian = xt::zeros<double>({dim, dim});
        f_plus.assign(dim, 0.0);
        hessian_mw = xt::zeros<double>({dim, dim});
        mass_weighted = false;
        mass_amu.assign(n, 0.0);
        frequencies = xt::zeros<double>({0});
        normal_modes = xt::zeros<double>({dim, 0});
        n_imaginary = 0;
        is_linear = false;
        zpe = 0.0;
        u_tot = 0.0;
        h_tot = 0.0;
        s_tot = 0.0;
        s_elec = 0.0;
        s_trans = 0.0;
        s_rot = 0.0;
        s_vib = 0.0;
        g_tot = 0.0;
        g_thermal = 0.0;
        thermo_ready = false;
    }
};

} // namespace vib
} // namespace cpu
} // namespace ATV
