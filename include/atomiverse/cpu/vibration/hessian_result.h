/// @file cpu/vibration/hessian_result.h
/// @brief cpu::vib::HessianResult — output of the numerical-Hessian driver.
///
/// Frame purity is preserved by NOT adding a Hessian field to
/// `cpu::Frame`. Instead `cpu::vib::Hessian::run()` returns this plain
/// struct by value (mirroring `cpu::minimizer::Minimize::run()` returning `int`).
/// Vibrational-frequency and thermochemistry links extend this struct
/// with mass-weighted frequencies / normal modes / thermo quantities.
#pragma once

#include <string>

#include <xtensor/containers/xtensor.hpp>

namespace ATV {
namespace cpu {
namespace vib {

/// @brief Raw numerical (central-difference) molecular Hessian.
///
/// The Hessian is the matrix of second derivatives of the
/// Born-Oppenheimer energy with respect to nuclear Cartesian
/// coordinates, `H[i.alpha, j.beta] = d2E / dR_{i.alpha} dR_{j.beta}`.
/// It is assembled directly in the frame's working units, eV/Angstrom^2
/// (positions in Angstrom, forces in eV/Angstrom). Conversion to
/// Hartree/Bohr^2 is deferred to the mass-weighting / frequency link.
struct HessianResult {
    /// @brief Dense symmetric Hessian, shape `[3N, 3N]`, in eV/Angstrom^2.
    ///        Row/column index `c` maps to atom `c/3`, axis `c%3` (x,y,z).
    xt::xtensor<double, 2> hessian;

    /// @brief Number of atoms `N` (so the matrix is `3N x 3N`).
    int n_atoms = 0;

    /// @brief Physical unit tag of `hessian`. Always "eV/Angstrom^2"
    ///        for the numerical-Hessian link.
    std::string units = "eV/Angstrom^2";

    // ── Vibrational-frequency link (phase4-vibrations-02) ──────────────
    //
    // Populated by `cpu::vib::HessianDiagonalizer` after `cpu::vib::MassWeighting`
    // has mass-weighted the Hessian. Empty until the frequency link runs.

    /// @brief Vibrational wavenumbers [cm^-1], length `3N-6` (nonlinear)
    ///        or `3N-5` (linear), ascending — and EMPTY for a single atom,
    ///        which has no vibrational degree of freedom to report.
    ///        Translations and rotations are projected out before
    ///        diagonalization, so this never carries five or six near-zero
    ///        entries to filter. Imaginary modes (force constant
    ///        `lambda < 0`) are reported as negative values, the universal
    ///        `imaginary_freq=False` chemistry convention.
    xt::xtensor<double, 1> frequencies;

    /// @brief Normal-mode eigenvectors, shape `[3N, n_modes]`. Column `k`
    ///        is mode `k` lifted back to the full `3N` Cartesian basis;
    ///        `n_modes == frequencies.size()`.
    xt::xtensor<double, 2> normal_modes;

    /// @brief Number of imaginary modes (negative-wavenumber entries in
    ///        `frequencies`). Zero at a genuine minimum; one at a
    ///        first-order saddle point.
    int n_imaginary = 0;

    /// @brief True when `_get_rotor_type` classified the molecule as
    ///        linear (`3N-5` vibrational modes), false for a regular
    ///        nonlinear rotor (`3N-6` modes).
    bool is_linear = false;

    // ── Dipole derivatives / IR intensities (hf-parity-04) ─────────────
    //
    // The atomic polar tensor (APT) ∂μ_a/∂R_x, populated by the analytic
    // gradient method (scf::Hessian<RHF>) alongside the Hessian, reusing the CPHF
    // density response. IR intensities per normal mode follow by projecting
    // this tensor onto `normal_modes`. Empty for the finite-difference
    // vib::Hessian driver (which never solves CPHF).

    /// @brief Atomic polar tensor ∂μ_a/∂R_x, shape `[3N, 3]` in atomic units
    ///        (e·Bohr per Bohr = e). Row `x = 3·atom + axis` is the nuclear
    ///        coordinate; column `a` is the dipole Cartesian axis. Σ_atoms of
    ///        each column-block is ~0 (translational invariance of the dipole
    ///        derivative). Empty unless produced by the analytic Hessian.
    xt::xtensor<double, 2> dipole_derivatives;

    // ── Thermochemistry link (phase4-vibrations-03) ────────────────────
    //
    // Populated by `cpu::vib::Thermochem` during the `thermochem` pass,
    // a faithful port of PySCF `pyscf/hessian/thermo.py::thermo()`. All
    // quantities are reported in Hartree (Eh) / Eh/K, exactly as PySCF
    // `thermo()` does. Zero until the thermochemistry link runs, and zero
    // for good on the `Hessian::run_from_cartesian` path, which never runs
    // that pass.
    //
    // THE ELECTRONIC ENERGY E0 IS CARRIED AS ZERO (`thermochem.cpp`): this
    // driver never adds the potential energy of the equilibrium geometry.
    // `u_tot`, `h_tot` and `g_tot` are therefore the THERMAL parts of U, H
    // and G, not absolute energies — add your own E0 [Hartree] for those.
    // One visible consequence: `g_tot` and `g_thermal` are identically the
    // same number.

    /// @brief Harmonic zero-point energy `ZPE = 0.5 R_Eh sum_i theta_i`
    ///        [Hartree]. Imaginary modes are excluded from the sum.
    double zpe = 0.0;

    /// @brief Thermal internal energy `U = ZPE + (E_vib-ZPE) + E_trans
    ///        + E_rot` at the configured temperature [Hartree]. The `E0`
    ///        term of `U = E0 + …` is zero here, per the convention above.
    double u_tot = 0.0;

    /// @brief Thermal enthalpy `H = H_vib + H_trans + H_rot` [Hartree]; the
    ///        `+kB*T` PV term lives in `H_trans`. The `H_elec = E0` term is
    ///        zero here, per the convention above.
    double h_tot = 0.0;

    /// @brief Total entropy `S = S_elec + S_trans + S_rot + S_vib`
    ///        [Hartree/K].
    double s_tot = 0.0;

    /// @brief Electronic entropy `S_elec = R_Eh ln(multiplicity)`
    ///        [Hartree/K]. Zero for a closed-shell singlet.
    double s_elec = 0.0;

    /// @brief Translational entropy (Sackur-Tetrode) [Hartree/K].
    double s_trans = 0.0;

    /// @brief Rotational entropy (rigid-rotor) [Hartree/K].
    double s_rot = 0.0;

    /// @brief Vibrational entropy (harmonic-oscillator) [Hartree/K].
    double s_vib = 0.0;

    /// @brief Thermal Gibbs free energy `G = H_tot - T*S_tot` [Hartree].
    ///        Not an absolute free energy: with `E0 = 0` it equals
    ///        `g_thermal` identically.
    double g_tot = 0.0;

    /// @brief Thermal correction to the Gibbs free energy `G_tot - E0`
    ///        [Hartree]. Independent of the electronic energy — this is
    ///        the quantity the acceptance test compares against PySCF. It
    ///        is numerically `g_tot` (E0 is carried as zero) and is
    ///        published under its own name because the correction, not the
    ///        absolute G, is what compares across codes.
    double g_thermal = 0.0;
};

} // namespace vib
} // namespace cpu
} // namespace ATV
