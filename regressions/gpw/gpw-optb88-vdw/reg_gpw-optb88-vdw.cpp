/// @file regressions/gpw/gpw-optb88-vdw/reg_gpw-optb88-vdw.cpp
/// @brief Periodic water-dimer GPW optB88-vdW (optB88+PW92 + DRSLL) offline pin.
///
/// Out-of-library consumer. Semi-local XC is `XcKind::OptB88VdW`; nonlocal
/// correlation is post-SCF DRSLL via `vdw_df` (REFERENCE.md). No live oracle
/// in ctest.

#include <cmath>
#include <cstdio>
#include <vector>

#include "atomiverse/cpu/gpw/gaussian_basis.h"
#include "atomiverse/cpu/gpw/gpw_solver.h"
#include "atomiverse/cpu/gpw/gth_potential.h"
#include "atomiverse/molrs.hpp"

namespace {

std::vector<ATV::cpu::gpw::GpwAtom> water_dimer_atoms() {
    using ATV::cpu::gpw::GaussianBasis;
    using ATV::cpu::gpw::GpwAtom;
    using ATV::cpu::gpw::GthPotential;
    constexpr double kA2B = 1.0 / 0.52917720859;
    const double ang[6][3] = {
        {0.000000, 0.000000, 0.000000},  // O
        {0.758602, 0.000000, 0.504284},  // H
        {0.758602, 0.000000, -0.504284}, // H
        {3.000000, 0.000000, 0.000000},  // O
        {3.758602, 0.000000, 0.504284},  // H
        {3.758602, 0.000000, -0.504284}, // H
    };
    const int Z[6] = {8, 1, 1, 8, 1, 1};
    const GaussianBasis bO(molrs::Element::O, "DZVP-MOLOPT-SR-GTH");
    const GaussianBasis bH(molrs::Element::H, "DZVP-MOLOPT-SR-GTH");
    const GthPotential pO(molrs::Element::O, "GTH-PBE-q6");
    const GthPotential pH(molrs::Element::H, "GTH-PBE-q1");

    std::vector<GpwAtom> atoms;
    atoms.reserve(6);
    for (int i = 0; i < 6; ++i) {
        const double pos[3] = {ang[i][0] * kA2B, ang[i][1] * kA2B,
                               ang[i][2] * kA2B};
        if (Z[i] == 8)
            atoms.emplace_back(pos, bO, pO, 8);
        else
            atoms.emplace_back(pos, bH, pH, 1);
    }
    return atoms;
}

ATV::cpu::gpw::GpwResult run_gpw(ATV::cpu::gpw::XcKind xc, bool vdw_df,
                                 std::vector<ATV::cpu::gpw::GpwAtom>& atoms) {
    using ATV::cpu::gpw::GpwDft;
    constexpr double kA2B = 1.0 / 0.52917720859;
    GpwDft gpw;
    const double L = 12.0 * kA2B;
    gpw.cell[0] = gpw.cell[1] = gpw.cell[2] = L;
    gpw.cutoff_ry = 200.0;
    gpw.n_occ = 8;
    gpw.max_iter = 40;
    gpw.tol = 1e-7;
    gpw.mix = 0.4;
    gpw.xc = xc;
    gpw.nonperiodic = false;
    gpw.vdw_df = vdw_df;
    gpw.build(atoms);
    return gpw.run(atoms);
}

} // namespace

int main() {
    using namespace ATV;
    using cpu::gpw::XcKind;

    // Offline pins (REFERENCE.md) — first-green ATV freeze; replace with CP2K
    // twin digits when available offline.
    // Placeholders updated after first green run of this binary.
    constexpr double kEDispRef = 1.5207282287995946e-01; // Ha, first-green freeze
    constexpr double kETotRef = -3.4481701207152035e+01; // Ha, first-green freeze
    constexpr double kTolDisp = 5.0e-5; // Ha
    constexpr double kTolTot = 1.0e-3;  // Ha (setup integrity until CP2K)
    constexpr double kDualFloor = 1.0e-4; // Ha |E_opt - E_pbe|

    auto atoms = water_dimer_atoms();

    // Dual gate: semi-local OptB88VdW (no nonlocal) must shift vs PBE.
    const auto res_pbe = run_gpw(XcKind::PBE, /*vdw_df=*/false, atoms);
    const auto res_opt = run_gpw(XcKind::OptB88VdW, /*vdw_df=*/false, atoms);
    if (!std::isfinite(res_pbe.e_total) || !std::isfinite(res_opt.e_total)) {
        std::fprintf(stderr, "FAIL: non-finite semi-local totals "
                             "PBE=%.14e Opt=%.14e\n",
                     res_pbe.e_total, res_opt.e_total);
        return 1;
    }
    const double d_semi = std::abs(res_opt.e_total - res_pbe.e_total);
    if (d_semi <= kDualFloor) {
        std::fprintf(stderr,
                     "FAIL: OptB88VdW vs PBE dual gate |dE|=%.3e Ha "
                     "(need > %.1e); E_opt=%.14e E_pbe=%.14e\n",
                     d_semi, kDualFloor, res_opt.e_total, res_pbe.e_total);
        return 1;
    }

    // Full optB88-vdW: OptB88VdW + DRSLL.
    const auto res = run_gpw(XcKind::OptB88VdW, /*vdw_df=*/true, atoms);
    if (!std::isfinite(res.e_disp) || !std::isfinite(res.e_total)) {
        std::fprintf(stderr, "FAIL: non-finite optB88-vdW E_disp=%.14e "
                             "e_total=%.14e\n",
                     res.e_disp, res.e_total);
        return 1;
    }
    if (!(res.e_disp != 0.0)) {
        std::fprintf(stderr, "FAIL: E_disp is zero (vdw_df inactive?)\n");
        return 1;
    }

    if (std::abs(res.e_disp - kEDispRef) > kTolDisp) {
        std::fprintf(stderr,
                     "FAIL: E_disp optB88-vdW got=%.14e want=%.14e "
                     "tol=%.1e Ha\n",
                     res.e_disp, kEDispRef, kTolDisp);
        return 1;
    }
    if (std::abs(res.e_total - kETotRef) > kTolTot) {
        std::fprintf(stderr,
                     "FAIL: e_total optB88-vdW got=%.14e want=%.14e "
                     "tol=%.1e Ha\n",
                     res.e_total, kETotRef, kTolTot);
        return 1;
    }

    std::printf("PASS: optB88-vdW water dimer E_disp=%.14e Ha "
                "(want %.14e +/- %.1e); e_total=%.14e Ha; "
                "dual |E_opt-E_pbe|=%.6e Ha; SCF iters=%d\n",
                res.e_disp, kEDispRef, kTolDisp, res.e_total, d_semi,
                res.iters);
    return 0;
}
