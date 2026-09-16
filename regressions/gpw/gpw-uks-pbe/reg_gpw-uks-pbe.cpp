/// @file regressions/gpw/gpw-uks-pbe/reg_gpw-uks-pbe.cpp
/// @brief O₂ triplet Γ-UKS GPW vs offline CP2K Quickstep PBE pin.
///
/// Public-API consumer only. Geometry and oracle numbers in REFERENCE.md
/// (CP2K 2026.1 offline; no live CP2K in ctest).
///
/// Exit non-zero on failure. Target ≤ 60 s Release.

#include <chrono>
#include <cmath>
#include <cstdio>

#include "atomiverse/core/units.h"
#include "atomiverse/cpu/evaluate/single_point.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/gpw/gaussian_basis.h"
#include "atomiverse/cpu/gpw/gpw_evaluator.h"
#include "atomiverse/cpu/gpw/gth_potential.h"
#include "atomiverse/molrs.hpp"

int main() {
    using namespace ATV;
    using clock = std::chrono::steady_clock;

    // ── O₂, R = 1.2075 Å, 12 Å cubic cell (Å), MT / nonperiodic ─────────
    // Center the molecule in the cell: nonperiodic MT grids cover [0,L);
    // atoms near the edge truncate Gaussian tails and destroy the energy.
    cpu::Frame frame(2);
    constexpr double L = 12.0;
    constexpr double R = 1.2075;
    for (int i = 0; i < 2; ++i) {
        frame.span(kAtom_type)[i] = 8;
        frame.span(kAtom_mass)[i] = 15.999;
        frame.span(kAtom_x)[i] = 0.5 * L;
        frame.span(kAtom_y)[i] = 0.5 * L;
    }
    frame.span(kAtom_z)[0] = 0.5 * L - 0.5 * R;
    frame.span(kAtom_z)[1] = 0.5 * L + 0.5 * R;
    frame.box = cpu::Box::cubic(L);

    cpu::gpw::Kind O{
        molrs::Element::O,
        cpu::gpw::GaussianBasis(molrs::Element::O, "DZVP-MOLOPT-GTH"),
        cpu::gpw::GthPotential(molrs::Element::O, "GTH-PBE-q6")};

    cpu::gpw::GPWEvaluator eval({O});
    eval.cutoff_ry = 400.0;
    eval.max_iter = 200;
    eval.tol = 1e-7;
    eval.mix = 0.4;
    eval.xc = cpu::gpw::XcKind::PBE;
    eval.nonperiodic = true;
    eval.n_alpha = 7; // multiplicity 3, 12 valence e⁻
    eval.n_beta = 5;

    const auto t0 = clock::now();
    cpu::single_point(frame, eval);
    const auto t1 = clock::now();
    const double wall_s = std::chrono::duration<double>(t1 - t0).count();

    const cpu::gpw::GpwResult& r = eval.last_result();
    const double e_ha = eval.last_energy_hartree();
    const double e_frame = frame.meta.get(kEnergy) / units::kHartreeInEv;
    const double s2 = eval.last_s2();

    // CP2K 2026.1 offline pin (centered O₂, UKS-PBE) — REFERENCE.md
    constexpr double kE_cp2k = -31.92879303435954;
    constexpr double kS2_cp2k = 2.005556;
    constexpr double kTolE = 1.0e-3; // Ha (first-green grid twin)
    constexpr double kTolS2 = 0.15;

    std::printf("E(O2, UKS-PBE/DZVP-MOLOPT-GTH, MT) = %.12f Ha\n", e_ha);
    std::printf("  CP2K pin                         = %.12f Ha  |dE|=%.3e "
                "(tol %.1e)\n",
                kE_cp2k, std::fabs(e_ha - kE_cp2k), kTolE);
    std::printf("  <S2> = %.8f  (CP2K %.8f, |d|=%.3e, tol %.2f)\n", s2,
                kS2_cp2k, std::fabs(s2 - kS2_cp2k), kTolS2);
    std::printf("  converged=%d iters=%d wall=%.3f s\n",
                static_cast<int>(r.converged), r.iters, wall_s);

    bool ok = r.converged && std::isfinite(e_ha) && std::isfinite(s2);
    ok = ok && std::fabs(e_ha - kE_cp2k) <= kTolE;
    ok = ok && std::fabs(s2 - kS2_cp2k) <= kTolS2;
    ok = ok && std::isfinite(e_frame) && std::fabs(e_frame - e_ha) < 1e-9;

    if (!ok) {
        std::fprintf(stderr, "FAIL gpw-uks-pbe\n");
        return 1;
    }
    std::printf("PASS gpw-uks-pbe\n");
    return 0;
}
