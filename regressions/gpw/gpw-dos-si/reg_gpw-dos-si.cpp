/// @file regressions/gpw/gpw-dos-si/reg_gpw-dos-si.cpp
/// @brief END-TO-END: multi-k GPW DOS on diamond Si (BCT 4-atom) vs CP2K pin.
///
/// Dual gate:
///   A1 energy |E_atv − E_cp2k| ≤ 1e-3 Ha (fair GTH-GPW twin)
///   A2 DOS character: n_elec_to_vbm ≈ 16, open gap, finite D(E)
/// Wall vs the same CP2K twin SCF is recorded in REFERENCE.md (offline).
///
/// Public headers only. Requires ATV_BUILD_GPW + ATV_BUILD_PW.

#include <chrono>
#include <cmath>
#include <cstdio>
#include <vector>

#include "atomiverse/cpu/gpw/bands_dos.h"
#include "atomiverse/cpu/gpw/gaussian_basis.h"
#include "atomiverse/cpu/gpw/gpw_solver.h"
#include "atomiverse/cpu/gpw/gth_potential.h"
#include "atomiverse/molrs.hpp"

namespace {

constexpr double kA2B = 1.0 / 0.52917720859; // Å → Bohr
constexpr double kHa2Ev = 27.211386245988;

// Offline CP2K 2026.1 pin (see REFERENCE.md) — replace after re-pin.
// Placeholder updated by the pin run in this wave.
constexpr double kECp2kHa =
    -15.802416412655166; // same twin class as gpw-kpoints-si
constexpr double kETolHa = 1e-3;
constexpr double kNelec = 16.0; // 4 Si × 4 valence e⁻
constexpr double kNelecTol = 1.5;
constexpr double kGapLoEv = 0.1;

std::vector<ATV::cpu::gpw::GpwAtom> si_bct(double a_bohr) {
    using namespace ATV;
    const double at = a_bohr / std::sqrt(2.0);
    const ATV::cpu::gpw::GaussianBasis bSi(molrs::Element::Si,
                                           "DZVP-MOLOPT-GTH");
    const ATV::cpu::gpw::GthPotential pSi(molrs::Element::Si, "GTH-PBE-q4");
    const double frac[4][3] = {
        {0.0, 0.0, 0.0}, {0.5, 0.5, 0.5}, {0.5, 0.0, 0.25}, {0.0, 0.5, 0.75}};
    std::vector<ATV::cpu::gpw::GpwAtom> atoms;
    for (int i = 0; i < 4; ++i) {
        const double pos[3] = {frac[i][0] * at, frac[i][1] * at,
                               frac[i][2] * a_bohr};
        atoms.emplace_back(pos, bSi, pSi, 14);
    }
    return atoms;
}

} // namespace

int main() {
    using namespace ATV;
    using cpu::gpw::Dos;
    using cpu::gpw::DosResult;
    using cpu::gpw::GpwAtom;
    using cpu::gpw::GpwDft;
    using cpu::gpw::XcKind;

    const double a = 5.43 * kA2B; // 10.26120 Bohr
    auto atoms = si_bct(a);

    GpwDft method;
    method.cell[0] = method.cell[1] = a / std::sqrt(2.0);
    method.cell[2] = a;
    method.cutoff_ry = 150.0;
    method.n_occ = 8;
    method.max_iter = 60;
    method.tol = 1e-8;
    method.xc = XcKind::PBE;
    method.eps_default = 1e-7;
    method.diis_depth = 6;
    method.k_time_reversal = true;
    method.build(atoms);

    Dos dos;
    dos.k_mesh = 2;
    dos.sigma_ha = 0.025; // ~0.68 eV Gaussian width
    dos.n_bins = 400;
    // Full AO spectrum from multi-k diagonalization (n_bands = 0).

    const auto t0 = std::chrono::steady_clock::now();
    const DosResult dr = dos.compute(method, atoms);
    const auto t1 = std::chrono::steady_clock::now();
    const double wall_s = std::chrono::duration<double>(t1 - t0).count();

    if (!dr.scf_converged) {
        std::printf("FAIL: SCF did not converge E=%g Ha\n", dr.scf_energy_ha);
        return 1;
    }
    if (dr.n_kpts < 2) {
        std::printf("FAIL: expected multi-k mesh, n_kpts=%d\n", dr.n_kpts);
        return 1;
    }
    if (dr.n_bands < 9) {
        // need occupied (8) + ≥1 empty for gap
        std::printf("FAIL: n_bands=%d too small for gap (need ≥9)\n",
                    dr.n_bands);
        return 1;
    }
    if (dr.energy_ha.size() != static_cast<std::size_t>(dos.n_bins)
        || dr.dos.size() != dr.energy_ha.size()) {
        std::printf("FAIL: DOS grid size mismatch bins=%zu dos=%zu want=%d\n",
                    dr.energy_ha.size(), dr.dos.size(), dos.n_bins);
        return 1;
    }

    // A1: absolute energy vs offline CP2K twin.
    const double de = std::abs(dr.scf_energy_ha - kECp2kHa);
    if (!(de <= kETolHa)) {
        std::printf("FAIL: |E_atv-E_cp2k|=%.6e Ha > tol=%.1e "
                    "(E_atv=%.12f E_cp2k=%.12f)\n",
                    de, kETolHa, dr.scf_energy_ha, kECp2kHa);
        return 1;
    }

    // A2: DOS character — occupied integral and open gap.
    if (!(std::isfinite(dr.n_elec_to_vbm)
          && std::abs(dr.n_elec_to_vbm - kNelec) <= kNelecTol)) {
        std::printf("FAIL: n_elec_to_vbm got=%.4f want=%.1f ± %.1f\n",
                    dr.n_elec_to_vbm, kNelec, kNelecTol);
        return 1;
    }
    const double gap_ev = dr.gap_ha * kHa2Ev;
    if (!(gap_ev >= kGapLoEv)) {
        std::printf("FAIL: gap_ev=%.4f < %.2f (open Si gap on multi-k mesh)\n",
                    gap_ev, kGapLoEv);
        return 1;
    }
    // DOS must be non-negative and have at least one positive peak.
    double dmax = 0;
    for (double d : dr.dos) {
        if (!(d >= -1e-12) || !std::isfinite(d)) {
            std::printf("FAIL: non-finite or negative DOS bin\n");
            return 1;
        }
        if (d > dmax)
            dmax = d;
    }
    if (!(dmax > 0.0)) {
        std::printf("FAIL: max D(E)=0 (empty DOS)\n");
        return 1;
    }

    std::printf("PASS: gpw-dos-si Si BCT a=5.43 A cutoff=150 Ry MP2x2x2  "
                "E=%.12f Ha |dE_cp2k|=%.3e  gap=%.4f eV  n_elec_to_vbm=%.3f  "
                "n_kpts=%d n_bands=%d dmax=%.4f  wall_scf+dos=%.3f s\n",
                dr.scf_energy_ha, de, gap_ev, dr.n_elec_to_vbm, dr.n_kpts,
                dr.n_bands, dmax, wall_s);
    return 0;
}
