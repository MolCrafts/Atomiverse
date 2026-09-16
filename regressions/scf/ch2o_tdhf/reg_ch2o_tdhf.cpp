/// @file regressions/scf/ch2o_tdhf/reg_ch2o_tdhf.cpp
/// @brief Formaldehyde HF/6-31G* TDA (CIS) and TDHF/RPA singlet excitation
///        energies + oscillator strengths vs the PySCF-pinned oracle
///        (Hirata & Head-Gordon 1999, DOI 10.1016/S0009-2614(99)01149-5;
///        Dreuw & Head-Gordon 2005, DOI 10.1021/cr0505627; Szabo & Ostlund
///        §7).
///
/// Out-of-library consumer: public headers only (`cpu/scf/methods.h`,
/// `cpu/scf/tdhf.h` via rhf.h, `cpu/frame/frame.h`, `core/units.h`), links
/// `atomiverse`.
///
/// CH2O, r(CO) = 1.203 Å, r(CH) = 1.099 Å, ∠HCH = 116.5°, planar (xz
/// plane, O on +z), HF/6-31G* Cartesian, RHF conv 1e-12 (identical
/// geometry in REFERENCE.md's PySCF run):
///  1. First 3 singlet TDA roots vs pinned PySCF `tdscf.TDA` and first 3
///     singlet TDHF roots vs `tdscf.TDHF`, each to 1e-4 eV (pins in
///     Hartree; both sides converted with the SAME factor, see below).
///  2. n→π* assignment: root 1 (A2 symmetry, HOMO n_O → π*_CO) is dark in
///     both theories (f < 1e-3).
///  3. π→π* bright state (root 3, transition dipole along the C=O axis):
///     length-gauge f vs PySCF to 1e-3.
///  4. Variational ordering ω_RPA ≤ ω_TDA root by root.
///
/// Exit non-zero on mismatch, printing got / want / tol. <= 60 s, single
/// CPU.

#include <cmath>
#include <cstdio>

#include "atomiverse/core/units.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/scf/methods.h"

namespace {

bool check(const char* what, double got, double want, double tol) {
    if (!std::isfinite(got) || std::abs(got - want) > tol) {
        std::fprintf(stderr, "FAIL: %s got=%.10f want=%.10f tol=%.1e\n", what,
                     got, want, tol);
        return false;
    }
    std::printf("  %-40s %12.7f (want %12.7f +/- %.1e)\n", what, got, want,
                tol);
    return true;
}

} // namespace

int main() {
    using namespace ATV;

    // ── Geometry (Å) — identical literals in REFERENCE.md's PySCF run ────
    cpu::Frame frame(4);
    const double xs[4] = {0.0, 0.0, 0.9345370953, -0.9345370953};
    const double ys[4] = {0.0, 0.0, 0.0, 0.0};
    const double zs[4] = {0.0, 1.203, -0.5783091021, -0.5783091021};
    const int types[4] = {6, 8, 1, 1};
    for (int i = 0; i < 4; ++i) {
        frame.span(kAtom_x)[i] = xs[i];
        frame.span(kAtom_y)[i] = ys[i];
        frame.span(kAtom_z)[i] = zs[i];
        frame.span(kAtom_type)[i] = types[i];
    }
    frame.box = cpu::Box::cubic(30.0);

    // ── PySCF 2.14.0 pins (REFERENCE.md; cart=True, conv_tol=1e-12) ──────
    constexpr double kERhf = -113.865575118441; // Ha
    constexpr double kTdaHa[3] = {0.171581500059, 0.365672890232,
                                  0.378524177461}; // Ha
    constexpr double kTdhfHa[3] = {0.165369739863, 0.356745006619,
                                   0.358521315202}; // Ha
    constexpr double kTdaF3 = 0.2165661719;         // π→π* bright state, root 3
    constexpr double kTdhfF3 = 0.1795988961;        // π→π* bright state, root 3
    constexpr double kTolE = 1e-6;                  // Ha, ground state
    constexpr double kTolEv = 1e-4;                 // eV, excitation energies
    constexpr double kTolF = 1e-3; // dimensionless, oscillator strengths

    // ── RHF/6-31G* (Cartesian d, engine convention) ──────────────────────
    cpu::scf::RHF rhf;
    rhf.basis = cpu::scf::BasisSet::Pople631Gs;
    rhf.tol = 1e-12;
    rhf.max_iter = 200;
    rhf.run(frame);
    if (!rhf.converged()) {
        std::fprintf(stderr, "FAIL: RHF/6-31G*(CH2O) did not converge\n");
        return 1;
    }
    std::printf("CH2O HF/6-31G* (Cartesian):\n");
    if (!check("E_RHF [Ha]", rhf.energy(), kERhf, kTolE))
        return 1;

    // ── TDA and TDHF, first 3 singlet roots ──────────────────────────────
    const cpu::scf::ExcitedStatesResult tda = rhf.tda(3);
    const cpu::scf::ExcitedStatesResult tdhf = rhf.tdhf(3);

    std::printf("\n  %-6s %-14s %-14s %-12s\n", "root", "TDA [eV]", "TDHF [eV]",
                "f(TDA)/f(TDHF)");
    for (int s = 0; s < 3; ++s)
        std::printf("  %-6d %-14.6f %-14.6f %.6f / %.6f\n", s + 1,
                    tda.energy_ev[static_cast<std::size_t>(s)],
                    tdhf.energy_ev[static_cast<std::size_t>(s)],
                    tda.oscillator_strength[static_cast<std::size_t>(s)],
                    tdhf.oscillator_strength[static_cast<std::size_t>(s)]);
    std::printf("\n");

    for (int s = 0; s < 3; ++s) {
        char label[64];
        std::snprintf(label, sizeof label, "TDA root %d [eV]", s + 1);
        if (!check(label, tda.energy_ev[static_cast<std::size_t>(s)],
                   kTdaHa[s] * units::kHartreeInEv, kTolEv))
            return 1;
        std::snprintf(label, sizeof label, "TDHF root %d [eV]", s + 1);
        if (!check(label, tdhf.energy_ev[static_cast<std::size_t>(s)],
                   kTdhfHa[s] * units::kHartreeInEv, kTolEv))
            return 1;
    }

    // ── ω_RPA ≤ ω_TDA per root (variational property of dropping B) ─────
    for (int s = 0; s < 3; ++s)
        if (tdhf.energy_ev[static_cast<std::size_t>(s)]
            > tda.energy_ev[static_cast<std::size_t>(s)] + 1e-9) {
            std::fprintf(stderr,
                         "FAIL: omega_RPA > omega_TDA at root %d "
                         "(%.8f > %.8f eV)\n",
                         s + 1, tdhf.energy_ev[static_cast<std::size_t>(s)],
                         tda.energy_ev[static_cast<std::size_t>(s)]);
            return 1;
        }
    std::printf("  omega_RPA <= omega_TDA per root          OK\n");

    // ── Assignments: root 1 = n→π* (dark, A2), root 3 = π→π* (bright,
    //    transition dipole along the C=O axis = z) ─────────────────────────
    if (!check("n->pi* (root 1) f_TDA (dark)", tda.oscillator_strength[0], 0.0,
               kTolF))
        return 1;
    if (!check("n->pi* (root 1) f_TDHF (dark)", tdhf.oscillator_strength[0],
               0.0, kTolF))
        return 1;
    if (!check("pi->pi* (root 3) f_TDA", tda.oscillator_strength[2], kTdaF3,
               kTolF))
        return 1;
    if (!check("pi->pi* (root 3) f_TDHF", tdhf.oscillator_strength[2], kTdhfF3,
               kTolF))
        return 1;

    const double z2 =
        tda.transition_dipole_au(2, 2) * tda.transition_dipole_au(2, 2);
    const double mu2 =
        tda.transition_dipole_au(2, 0) * tda.transition_dipole_au(2, 0)
        + tda.transition_dipole_au(2, 1) * tda.transition_dipole_au(2, 1) + z2;
    if (z2 < 0.99 * mu2) {
        std::fprintf(stderr,
                     "FAIL: pi->pi* transition dipole is not along the C=O "
                     "axis (|mu_z|^2 = %.6f of |mu|^2 = %.6f)\n",
                     z2, mu2);
        return 1;
    }
    std::printf("  pi->pi* transition dipole along C=O      OK\n");

    std::printf("\nPASS: ch2o_tdhf\n");
    return 0;
}
