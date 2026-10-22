/// @file regressions/gpw/gpw-bands-si/reg_gpw-bands-si.cpp
/// @brief END-TO-END: GPW RKS band structure on diamond Si (bct 4-atom)
///        vs offline CP2K Quickstep BAND_STRUCTURE twin.
///
/// Public GPW headers only. See REFERENCE.md for the frozen pin.

#include <chrono>
#include <cmath>
#include <cstdio>
#include <vector>

#include "atomiverse/core/frame_schema.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/gpw/bands_dos.h"
#include "atomiverse/cpu/gpw/gaussian_basis.h"
#include "atomiverse/cpu/gpw/gpw_solver.h"
#include "atomiverse/cpu/gpw/gth_potential.h"
#include "atomiverse/molrs.hpp"

namespace {

constexpr double kHa2Ev = 27.211386245988;

int g_fail = 0;

void check(bool ok, const char* what) {
    if (!ok) {
        std::fprintf(stderr, "FAIL: %s\n", what);
        ++g_fail;
    } else {
        std::printf("  ok: %s\n", what);
    }
}

bool near_k(const std::array<double, 3>& a, const std::array<double, 3>& b,
            double tol = 1e-8) {
    return std::abs(a[0] - b[0]) < tol && std::abs(a[1] - b[1]) < tol
           && std::abs(a[2] - b[2]) < tol;
}

/// Diamond Si as the 4-atom bct cell of cubic lattice constant `a_ang` [Å].
ATV::cpu::Frame si_bct(double a_ang) {
    using namespace ATV;
    const double at = a_ang / std::sqrt(2.0);
    const double frac[4][3] = {
        {0.0, 0.0, 0.0}, {0.5, 0.5, 0.5}, {0.5, 0.0, 0.25}, {0.0, 0.5, 0.75}};
    cpu::Frame frame(4);
    for (int i = 0; i < 4; ++i) {
        frame.span(kAtom_type)[i] = 14;
        frame.span(kAtom_x)[i] = frac[i][0] * at;
        frame.span(kAtom_y)[i] = frac[i][1] * at;
        frame.span(kAtom_z)[i] = frac[i][2] * a_ang;
    }
    frame.box = cpu::Box::orthorhombic(at, at, a_ang);
    return frame;
}

} // namespace

int main() {
    using namespace ATV::cpu::gpw;

    // Offline CP2K 2026.1 pin — see REFERENCE.md / gpw-bands-si.inp
    // ENERGY| Total FORCE_EVAL ( QS ) energy [hartree]
    constexpr double kCp2kEtotHa = -15.802416412655166;
    constexpr double kEtotTolHa = 1e-3;
    // Path gap from CP2K si_gpw_bands (n_occ=8): VBM=CBM at Γ, gap 0.893516 eV.
    // ATV first green 0.894761 eV (Δ ≈ 1.2 meV). Window 50 meV for twin.
    constexpr double kCp2kGapEv = 0.893516;
    constexpr double kGapTolEv = 0.05;

    ATV::cpu::Frame frame = si_bct(5.43);

    GpwDft method{{Kind{molrs::Element::Si,
                        GaussianBasis(molrs::Element::Si, "DZVP-MOLOPT-GTH"),
                        GthPotential(molrs::Element::Si, "GTH-PBE-q4")}},
                  150.0,
                  XcKind::PBE};
    method.n_occ = 8; // 4 Si × 4 valence e⁻ / 2
    method.max_iter = 60;
    method.tol = 1e-8;
    method.eps_default = 1e-7;
    method.k_mesh[0] = method.k_mesh[1] = method.k_mesh[2] = 2;
    method.k_time_reversal = true;
    method.build(frame);

    Bands bands;
    bands.n_bands = 12; // n_occ + ADDED_MOS-class empty
    bands.set_diamond_path(4);

    const auto t0 = std::chrono::steady_clock::now();
    const BandsResult br = bands.compute(method, frame);
    const auto t1 = std::chrono::steady_clock::now();
    const double wall_s = std::chrono::duration<double>(t1 - t0).count();

    std::printf("gpw-bands-si: E=%.10f Ha  gap=%.6f eV  wall=%.3f s  "
                "nk_path=%zu  conv=%d\n",
                br.energy_ha, br.gap_ha * kHa2Ev, wall_s, br.kfrac.size(),
                br.scf_converged ? 1 : 0);

    check(br.scf_converged, "SCF converged");
    check(std::isfinite(br.energy_ha), "E finite");
    check(std::abs(br.energy_ha - kCp2kEtotHa) <= kEtotTolHa,
          "E_tot vs CP2K (1e-3 Ha)");

    check(!br.eigenvalues_ha.empty(), "path eigenvalues present");
    check(br.kfrac.size() == br.eigenvalues_ha.size(),
          "kfrac/evals size match");
    check(static_cast<int>(br.kfrac.size()) == 16,
          "diamond path n_seg=4 → 16 k");

    for (const auto& row : br.eigenvalues_ha) {
        if (static_cast<int>(row.size()) != bands.n_bands) {
            check(false, "n_bands per k");
            break;
        }
        for (std::size_t b = 1; b < row.size(); ++b) {
            if (row[b] + 1e-10 < row[b - 1]) {
                check(false, "eigenvalues sorted ascending");
                break;
            }
        }
    }

    // Γ and X on the path.
    const std::array<double, 3> G{0, 0, 0};
    const std::array<double, 3> X{0.5, 0.0, 0.5};
    int ik_g = -1, ik_x = -1;
    for (std::size_t ik = 0; ik < br.kfrac.size(); ++ik) {
        if (near_k(br.kfrac[ik], G) && ik_g < 0)
            ik_g = static_cast<int>(ik);
        if (near_k(br.kfrac[ik], X) && ik_x < 0)
            ik_x = static_cast<int>(ik);
    }
    check(ik_g >= 0 && ik_x >= 0, "path includes Γ and X");

    // Local gaps positive (no inverted spectrum).
    constexpr int kNOcc = 8;
    double min_local_ev = 1e300;
    for (const auto& row : br.eigenvalues_ha) {
        const double lg = (row[static_cast<std::size_t>(kNOcc)]
                           - row[static_cast<std::size_t>(kNOcc - 1)])
                          * kHa2Ev;
        if (lg < min_local_ev)
            min_local_ev = lg;
    }
    check(min_local_ev > 0.0, "min local gap > 0");

    const double gap_ev = br.gap_ha * kHa2Ev;
    check(std::abs(gap_ev - kCp2kGapEv) <= kGapTolEv,
          "path gap vs CP2K (50 meV)");

    if (ik_g >= 0 && ik_x >= 0) {
        const auto& eg = br.eigenvalues_ha[static_cast<std::size_t>(ik_g)];
        const auto& ex = br.eigenvalues_ha[static_cast<std::size_t>(ik_x)];
        const double e_v_g = eg[static_cast<std::size_t>(kNOcc - 1)];
        const double e_c_g = eg[static_cast<std::size_t>(kNOcc)];
        const double e_c_x = ex[static_cast<std::size_t>(kNOcc)];
        const double g_local_ev = (e_c_g - e_v_g) * kHa2Ev;
        const double gx_ev = (e_c_x - e_v_g) * kHa2Ev;
        std::printf("  Γ_v=%.6f Ha  Γ_c=%.6f Ha  X_c=%.6f Ha  "
                    "Γ_local=%.4f eV  Γ→X=%.4f eV  path_gap=%.4f eV\n",
                    e_v_g, e_c_g, e_c_x, g_local_ev, gx_ev, gap_ev);
        // On the bct cell, CP2K twin places VBM and CBM both at Γ (direct).
        check(std::abs(g_local_ev - kCp2kGapEv) <= kGapTolEv,
              "Γ local gap vs CP2K (50 meV)");
        check(br.vbm_k == ik_g, "VBM at Γ (twin character)");
        check(br.cbm_k == ik_g, "CBM at Γ (twin character)");
        check(br.gap_ha > 0.0, "path gap > 0");
    }

    // Fair wall report line for REFERENCE (ATV side).
    std::printf("ATV_WALL_S %.6f\n", wall_s);

    if (g_fail) {
        std::fprintf(stderr, "%d checks failed\n", g_fail);
        return 1;
    }
    std::printf("ALL CHECKS PASSED\n");
    return 0;
}
