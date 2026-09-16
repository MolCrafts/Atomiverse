/// @file regressions/gpw/gpw-hirshfeld-kpoints/reg_gpw-hirshfeld-kpoints.cpp
/// @brief Water-dimer multi-k GPW PBE + Hirshfeld-scaled MBD@rsSCS vs offline pin.
///
/// Out-of-library consumer: public GPW headers only. Geometry and E_disp pin
/// are in REFERENCE.md (no live oracle in ctest).

#include <cmath>
#include <cstdio>
#include <vector>

#include "atomiverse/cpu/gpw/gaussian_basis.h"
#include "atomiverse/cpu/gpw/gpw_solver.h"
#include "atomiverse/cpu/gpw/gth_potential.h"
#include "atomiverse/molrs.hpp"

int main() {
    using namespace ATV;
    using cpu::gpw::GaussianBasis;
    using cpu::gpw::GpwAtom;
    using cpu::gpw::GpwDft;
    using cpu::gpw::GthPotential;
    using cpu::gpw::XcKind;

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

    // Offline pin (REFERENCE.md): multi-k (2×2×2) Hirshfeld-scaled MBD@rsSCS,
    // isolated MBD geometry with volumes from the multi-k GPW PBE density.
    constexpr double kEDispRef = -1.1479849553026611e-03; // Ha
    constexpr double kTol = 1.0e-8;                       // Ha

    GpwDft gpw;
    const double L = 12.0 * kA2B;
    gpw.cell[0] = gpw.cell[1] = gpw.cell[2] = L;
    gpw.cutoff_ry = 200.0;
    gpw.n_occ = 8;
    gpw.max_iter = 16;
    gpw.tol = 1e-6;
    gpw.mix = 0.4;
    gpw.xc = XcKind::PBE;
    gpw.nonperiodic = false;
    gpw.mbd = true;
    gpw.hirshfeld = true;
    gpw.k_mesh[0] = gpw.k_mesh[1] = gpw.k_mesh[2] = 2;
    gpw.k_time_reversal = true;

    gpw.build(atoms);
    const auto& res = gpw.run(atoms);

    std::printf("PIN_CANDIDATE E_disp=%.16e Ha  volumes:", res.e_disp);
    for (double v : res.volume_ratios)
        std::printf(" %.8f", v);
    std::printf("  e_total=%.10f iters=%d conv=%d n_kpts=%d\n", res.e_total,
                res.iters, res.converged ? 1 : 0, res.n_kpts);

    if (res.n_kpts < 2) {
        std::fprintf(stderr, "FAIL: expected multi-k (n_kpts>=2), got %d\n",
                     res.n_kpts);
        return 1;
    }
    if (res.volume_ratios.size() != 6) {
        std::fprintf(stderr, "FAIL: expected 6 volume ratios, got %zu\n",
                     res.volume_ratios.size());
        return 1;
    }
    for (std::size_t i = 0; i < res.volume_ratios.size(); ++i) {
        if (!(res.volume_ratios[i] > 0.3) || !(res.volume_ratios[i] < 1.5)) {
            std::fprintf(stderr, "FAIL: volume_ratios[%zu]=%.6f out of range\n",
                         i, res.volume_ratios[i]);
            return 1;
        }
    }
    if (!std::isfinite(res.e_disp) || std::abs(res.e_disp - kEDispRef) > kTol) {
        std::fprintf(stderr,
                     "FAIL: E_disp multi-k Hirshfeld-MBD water dimer "
                     "got=%.14e want=%.14e tol=%.1e Ha\n",
                     res.e_disp, kEDispRef, kTol);
        return 1;
    }
    if (!std::isfinite(res.e_total) || res.e_disp >= 0.0) {
        std::fprintf(stderr,
                     "FAIL: e_total non-finite or E_disp not attractive\n");
        return 1;
    }

    std::printf("PASS: E_disp PBE-MBD@rsSCS multi-k Hirshfeld water dimer = "
                "%.14e Ha (want %.14e +/- %.1e)\n",
                res.e_disp, kEDispRef, kTol);
    return 0;
}
