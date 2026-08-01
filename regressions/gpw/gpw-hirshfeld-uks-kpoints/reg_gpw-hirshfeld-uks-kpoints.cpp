/// @file regressions/gpw/gpw-hirshfeld-uks-kpoints/reg_gpw-hirshfeld-uks-kpoints.cpp
/// @brief Multi-k collinear UKS GPW PBE + Hirshfeld-scaled MBD@rsSCS vs offline pin.
///
/// Out-of-library consumer: public GPW headers only. Hard gate is E_disp from
/// the multi-k UKS total density path (`fold_dispersion_k_uks`); geometry and
/// pin are in REFERENCE.md. Stock CP2K has no TS/MBD — oracle is libMBD /
/// literature (never live in ctest).
///
/// Dual gate (one product SCF):
///   A1 free-atom MBD (geometry-only offline dual) vs Hirshfeld product E_disp
///   A2 absolute E_disp + physical volume ratios
///
/// Exit non-zero on failure.

#include <chrono>
#include <cmath>
#include <cstdio>
#include <vector>

#include "atomiverse/cpu/gpw/gaussian_basis.h"
#include "atomiverse/cpu/gpw/gpw_solver.h"
#include "atomiverse/cpu/gpw/gth_potential.h"
#include "atomiverse/molrs.hpp"

namespace {

int g_fail = 0;

void check(bool ok, const char* what) {
    if (!ok) {
        std::fprintf(stderr, "FAIL: %s\n", what);
        ++g_fail;
    } else {
        std::printf("  ok: %s\n", what);
    }
}

constexpr double kA2B = 1.0 / 0.52917720859;

std::vector<ATV::cpu::gpw::GpwAtom> o2_triplet(double L_bohr) {
    using namespace ATV;
    constexpr double R_ang = 1.2075;
    const double z0 = 0.5 * L_bohr - 0.5 * R_ang * kA2B;
    const double z1 = 0.5 * L_bohr + 0.5 * R_ang * kA2B;
    const cpu::gpw::GaussianBasis bO(molrs::Element::O, "DZVP-MOLOPT-SR-GTH");
    const cpu::gpw::GthPotential pO(molrs::Element::O, "GTH-PBE-q6");
    std::vector<cpu::gpw::GpwAtom> atoms;
    const double p0[3] = {0.5 * L_bohr, 0.5 * L_bohr, z0};
    const double p1[3] = {0.5 * L_bohr, 0.5 * L_bohr, z1};
    atoms.emplace_back(p0, bO, pO, 8);
    atoms.emplace_back(p1, bO, pO, 8);
    return atoms;
}

/// Free-atom MBD@rsSCS on the same O₂ nuclear geometry (periodic 10 Å cube,
/// volume_ratio = 1). Geometry-only dual of `mbd_rsscs_pbe` (REFERENCE.md);
/// measured offline via public `GpwDft{mbd=true, hirshfeld=false}`.
constexpr double kEDispFreeAtom = -1.1699749375715385e-04; // Ha

} // namespace

int main() {
    using namespace ATV;
    using cpu::gpw::GpwDft;
    using cpu::gpw::XcKind;
    using clock = std::chrono::steady_clock;
    const auto t0 = clock::now();

    // Offline pin (REFERENCE.md): Hirshfeld-scaled MBD@rsSCS from multi-k UKS
    // PBE density (CUTOFF 120 Ry, MP 2×2×1 TR → 2 irr. k, DZVP-MOLOPT-SR-GTH).
    constexpr double kEDispRef = -1.0427466367435301e-04; // Ha (PIN_CANDIDATE)
    constexpr double kTol = 1.0e-8;                       // Ha

    const double L = 10.0 * kA2B;
    auto atoms = o2_triplet(L);

    GpwDft gpw;
    gpw.cell[0] = gpw.cell[1] = gpw.cell[2] = L;
    gpw.cutoff_ry = 120.0;
    gpw.n_alpha = 7;
    gpw.n_beta = 5;
    gpw.max_iter = 80;
    gpw.tol = 1e-6;
    gpw.mix = 0.45;
    gpw.diis_depth = 10;
    gpw.xc = XcKind::PBE;
    gpw.mbd = true;
    gpw.hirshfeld = true;
    // 2×2×1 MP + TR → multi-k (n_kpts = 2). 2×1×1 folds to Γ under TR.
    gpw.k_mesh[0] = 2;
    gpw.k_mesh[1] = 2;
    gpw.k_mesh[2] = 1;
    gpw.k_time_reversal = true;

    gpw.build(atoms);
    const auto& res = gpw.run(atoms);

    std::printf("PIN_CANDIDATE E_disp=%.16e Ha  volumes:", res.e_disp);
    for (double v : res.volume_ratios)
        std::printf(" %.8f", v);
    std::printf("  e_total=%.10f s2=%.6f iters=%d conv=%d n_kpts=%d\n",
                res.e_total, res.s2, res.iters, res.converged ? 1 : 0,
                res.n_kpts);

    check(res.converged, "SCF converged");
    check(res.n_kpts >= 2, "multi-k (n_kpts>=2)");
    check(res.volume_ratios.size() == 2, "two volume ratios");
    for (std::size_t i = 0; i < res.volume_ratios.size(); ++i) {
        if (!(res.volume_ratios[i] > 0.2) || !(res.volume_ratios[i] < 2.0)) {
            std::fprintf(stderr, "FAIL: volume_ratios[%zu]=%.6f out of range\n",
                         i, res.volume_ratios[i]);
            ++g_fail;
        } else {
            std::printf("  ok: volume_ratios[%zu] in (0.2, 2.0)\n", i);
        }
    }
    check(std::isfinite(res.e_disp) && res.e_disp < 0.0,
          "E_disp finite attractive");

    // A1: Hirshfeld density scaling moves E_disp vs free-atom (v=1).
    const double e_free = kEDispFreeAtom;
    std::printf("A1 free=%.16e  product=%.16e  d=%.3e\n", e_free, res.e_disp,
                res.e_disp - e_free);
    check(std::isfinite(e_free) && e_free < 0.0, "A1 free-atom MBD attractive");
    check(std::abs(res.e_disp - e_free) > 1e-6,
          "A1 Hirshfeld moves E_disp vs free-atom (>1e-6 Ha)");
    check(std::abs(res.e_disp) < std::abs(e_free),
          "A1 |E_disp Hirshfeld| < |E_disp free-atom|");

    // A2 absolute pin.
    if (!std::isfinite(res.e_disp) || std::abs(res.e_disp - kEDispRef) > kTol) {
        std::fprintf(stderr,
                     "FAIL: E_disp multi-k UKS Hirshfeld-MBD O2 "
                     "got=%.14e want=%.14e tol=%.1e Ha\n",
                     res.e_disp, kEDispRef, kTol);
        ++g_fail;
    } else {
        std::printf("  ok: A2 E_disp pin within %.1e Ha\n", kTol);
    }

    const auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                        clock::now() - t0)
                        .count();
    std::printf("WALL_MS=%ld  fails=%d\n", static_cast<long>(ms), g_fail);
    if (g_fail != 0) {
        std::fprintf(stderr, "FAILED %d gate(s)\n", g_fail);
        return 1;
    }
    std::printf("PASS: multi-k UKS Hirshfeld-MBD O2 E_disp=%.14e Ha "
                "(want %.14e +/- %.1e)  volumes=%.6f %.6f\n",
                res.e_disp, kEDispRef, kTol, res.volume_ratios[0],
                res.volume_ratios[1]);
    return 0;
}
