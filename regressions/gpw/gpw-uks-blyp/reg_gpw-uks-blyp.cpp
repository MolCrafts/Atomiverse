/// @file regressions/gpw/gpw-uks-blyp/reg_gpw-uks-blyp.cpp
/// @brief O₂ triplet Γ-UKS GPW vs offline CP2K Quickstep BLYP pin + PBE dual-gate.
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

namespace {

ATV::cpu::Frame make_o2_mt() {
    ATV::cpu::Frame frame(2);
    constexpr double L = 12.0;
    constexpr double R = 1.2075;
    for (int i = 0; i < 2; ++i) {
        frame.span(ATV::kAtom_type)[i] = 8;
        frame.span(ATV::kAtom_mass)[i] = 15.999;
        frame.span(ATV::kAtom_x)[i] = 0.5 * L;
        frame.span(ATV::kAtom_y)[i] = 0.5 * L;
    }
    frame.span(ATV::kAtom_z)[0] = 0.5 * L - 0.5 * R;
    frame.span(ATV::kAtom_z)[1] = 0.5 * L + 0.5 * R;
    frame.box = ATV::cpu::Box::cubic(L);
    return frame;
}

ATV::cpu::gpw::GPWEvaluator make_eval(ATV::cpu::gpw::XcKind xc) {
    using namespace ATV;
    cpu::gpw::Kind O{
        molrs::Element::O,
        cpu::gpw::GaussianBasis(molrs::Element::O, "DZVP-MOLOPT-GTH"),
        cpu::gpw::GthPotential(molrs::Element::O, "GTH-PBE-q6")};
    cpu::gpw::GPWEvaluator eval({O});
    eval.cutoff_ry = 400.0;
    eval.max_iter = 200;
    eval.tol = 1e-7;
    eval.mix = 0.4;
    eval.xc = xc;
    eval.nonperiodic = true;
    eval.n_alpha = 7; // multiplicity 3, 12 valence e⁻
    eval.n_beta = 5;
    return eval;
}

} // namespace

int main() {
    using namespace ATV;
    using clock = std::chrono::steady_clock;

    // Dual-gate floor [Ha] — BLYP vs PBE must activate a different functional.
    constexpr double kDeFloor = 1.0e-4;
    // CP2K 2026.1 offline pin (centered O₂, UKS-BLYP) — REFERENCE.md
    // Placeholder until pin is measured; first-green may soft-band if needed.
    constexpr double kE_cp2k = -31.90397685949710;
    constexpr double kS2_cp2k = 2.006656;
    constexpr double kTolE = 1.0e-3; // Ha (first-green grid twin)
    constexpr double kTolS2 = 0.15;

    cpu::Frame frame_blyp = make_o2_mt();
    cpu::gpw::GPWEvaluator eval_blyp = make_eval(cpu::gpw::XcKind::BLYP);

    const auto t0 = clock::now();
    cpu::single_point(frame_blyp, eval_blyp);
    const auto t1 = clock::now();
    const double wall_blyp = std::chrono::duration<double>(t1 - t0).count();

    const cpu::gpw::GpwResult& r = eval_blyp.last_result();
    const double e_blyp = eval_blyp.last_energy_hartree();
    const double e_frame = frame_blyp.meta.get(kEnergy) / units::kHartreeInEv;
    const double s2 = eval_blyp.last_s2();

    cpu::Frame frame_pbe = make_o2_mt();
    cpu::gpw::GPWEvaluator eval_pbe = make_eval(cpu::gpw::XcKind::PBE);
    cpu::single_point(frame_pbe, eval_pbe);
    const double e_pbe = eval_pbe.last_energy_hartree();
    const bool conv_pbe = eval_pbe.last_result().converged;

    std::printf("E(O2, UKS-BLYP/DZVP-MOLOPT-GTH, MT) = %.12f Ha\n", e_blyp);
    std::printf("  CP2K pin                          = %.12f Ha  |dE|=%.3e "
                "(tol %.1e)\n",
                kE_cp2k, std::fabs(e_blyp - kE_cp2k), kTolE);
    std::printf("  E_PBE twin                        = %.12f Ha  |dE_blyp-pbe|=%.3e "
                "(floor %.1e)\n",
                e_pbe, std::fabs(e_blyp - e_pbe), kDeFloor);
    std::printf("  <S2> = %.8f  (CP2K %.8f, |d|=%.3e, tol %.2f)\n", s2,
                kS2_cp2k, std::fabs(s2 - kS2_cp2k), kTolS2);
    std::printf("  converged=%d iters=%d wall_blyp=%.3f s\n",
                static_cast<int>(r.converged), r.iters, wall_blyp);

    bool ok = r.converged && std::isfinite(e_blyp) && std::isfinite(s2);
    ok = ok && conv_pbe && std::isfinite(e_pbe);
    ok = ok && std::fabs(e_blyp - e_pbe) > kDeFloor;
    ok = ok && std::fabs(e_blyp - kE_cp2k) <= kTolE;
    ok = ok && std::fabs(s2 - kS2_cp2k) <= kTolS2;
    ok = ok && std::isfinite(e_frame) && std::fabs(e_frame - e_blyp) < 1e-9;

    if (!ok) {
        std::fprintf(stderr, "FAIL gpw-uks-blyp\n");
        return 1;
    }
    std::printf("PASS gpw-uks-blyp\n");
    return 0;
}
