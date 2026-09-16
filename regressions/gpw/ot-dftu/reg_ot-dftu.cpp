/// @file regressions/gpw/ot-dftu/reg_ot-dftu.cpp
/// @brief END-TO-END: Γ-UKS GPW OT + Dudarev DFT+U vs offline CP2K pin.
///
/// Public API only. Geometry and oracle numbers in REFERENCE.md
/// (CP2K offline; no live CP2K in ctest). Exit non-zero on failure. ≤ 60 s.

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>

#include "atomiverse/core/units.h"
#include "atomiverse/cpu/evaluate/single_point.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/gpw/gaussian_basis.h"
#include "atomiverse/cpu/gpw/gpw_evaluator.h"
#include "atomiverse/cpu/gpw/gth_potential.h"
#include "atomiverse/molrs.hpp"

using namespace ATV;

namespace {

cpu::Frame make_nio() {
    // NiO molecule, R(Ni–O)=1.668 Å, centered in 10 Å cubic cell (MT).
    constexpr double L = 10.0;
    constexpr double R = 1.668;
    cpu::Frame frame(2);
    frame.span(kAtom_type)[0] = 28;
    frame.span(kAtom_type)[1] = 8;
    frame.span(kAtom_mass)[0] = 58.693;
    frame.span(kAtom_mass)[1] = 15.999;
    frame.span(kAtom_x)[0] = 0.5 * L;
    frame.span(kAtom_y)[0] = 0.5 * L;
    frame.span(kAtom_z)[0] = 0.5 * L - 0.5 * R;
    frame.span(kAtom_x)[1] = 0.5 * L;
    frame.span(kAtom_y)[1] = 0.5 * L;
    frame.span(kAtom_z)[1] = 0.5 * L + 0.5 * R;
    frame.box = cpu::Box::cubic(L);
    return frame;
}

cpu::gpw::GPWEvaluator make_eval(bool use_ot) {
    cpu::gpw::Kind Ni{
        molrs::Element::Ni,
        cpu::gpw::GaussianBasis(molrs::Element::Ni, "SZV-GTH-PADE-q18"),
        cpu::gpw::GthPotential(molrs::Element::Ni, "GTH-PADE-q18")};
    cpu::gpw::Kind O{molrs::Element::O,
                     cpu::gpw::GaussianBasis(molrs::Element::O, "SZV-GTH-PADE"),
                     cpu::gpw::GthPotential(molrs::Element::O, "GTH-PADE-q6")};

    cpu::gpw::GPWEvaluator eval({Ni, O});
    eval.cutoff_ry = 250.0;
    eval.max_iter = 300;
    eval.tol = 1e-5;
    eval.mix = 0.35;
    eval.xc = cpu::gpw::XcKind::PADE;
    eval.nonperiodic = true;
    eval.use_ot = use_ot;
    // multiplicity 3, 24 valence e⁻ → Nα=13, Nβ=11
    eval.n_alpha = 13;
    eval.n_beta = 11;
    eval.plus_u.push_back({molrs::Element::Ni, /*l=*/2, /*u_ev=*/6.0});
    return eval;
}

} // namespace

int main() {
    // CP2K 2026.1 offline OT+PLUS_U pin — REFERENCE.md
    constexpr double kE_cp2k = -184.92466076474310;
    constexpr double kTolE = 5.0e-2; // Ha — single-grid vs multigrid+OT
    constexpr double kOtDiisTolHa = 1.0e-3;
    constexpr double kM_Ni_cp2k = 1.177632;
    constexpr double kTolM = 0.5;
    constexpr double kS2_cp2k = 2.019198;
    constexpr double kTolS2 = 0.25;

    unsetenv("ATV_GPW_OT");
    using clock = std::chrono::steady_clock;

    cpu::Frame frame_ot = make_nio();
    auto eval_ot = make_eval(/*use_ot=*/true);
    const auto t0 = clock::now();
    cpu::single_point(frame_ot, eval_ot);
    const auto t1 = clock::now();
    const double wall_ot = std::chrono::duration<double>(t1 - t0).count();

    const cpu::gpw::GpwResult& r_ot = eval_ot.last_result();
    const double e_ot = eval_ot.last_energy_hartree();
    const double e_frame = frame_ot.meta.get(kEnergy) / units::kHartreeInEv;
    const double s2 = eval_ot.last_s2();

    std::printf("E(NiO mol, OT+PADE+U/SZV, MT) = %.12f Ha\n", e_ot);
    std::printf("  e_dftu = %.8f Ha\n", r_ot.e_dftu);
    std::printf("  CP2K pin                     = %.12f Ha  |dE|=%.3e "
                "(tol %.1e)\n",
                kE_cp2k, std::fabs(e_ot - kE_cp2k), kTolE);
    std::printf("  <S2> = %.6f  (CP2K %.6f)\n", s2, kS2_cp2k);
    std::printf("  converged=%d iters=%d wall=%.3f s\n",
                static_cast<int>(r_ot.converged), r_ot.iters, wall_ot);

    bool ok = r_ot.converged && std::isfinite(e_ot) && std::isfinite(s2);
    ok = ok && std::isfinite(e_frame) && std::fabs(e_frame - e_ot) < 1e-9;
    ok = ok && std::fabs(e_ot - kE_cp2k) <= kTolE;
    ok = ok && std::fabs(s2 - kS2_cp2k) <= kTolS2;
    ok = ok && std::isfinite(r_ot.e_dftu) && (r_ot.e_dftu > 1e-4);

    if (r_ot.mulliken_spin.size() >= 1) {
        const double mNi = r_ot.mulliken_spin[0];
        std::printf("  m_Ni = %.4f  (CP2K %.4f, tol %.2f)\n", mNi, kM_Ni_cp2k,
                    kTolM);
        ok = ok && (mNi > 0.3);
        ok = ok && (std::fabs(mNi - kM_Ni_cp2k) <= kTolM);
    } else {
        std::fprintf(stderr, "FAIL: empty mulliken_spin\n");
        ok = false;
    }

    // OT vs DIIS fixed-point parity under the same PLUS_U settings.
    // Soft residual on both paths → milli-Ha class on this twin.
    cpu::Frame frame_diis = make_nio();
    auto eval_diis = make_eval(/*use_ot=*/false);
    const auto t2 = clock::now();
    cpu::single_point(frame_diis, eval_diis);
    const auto t3 = clock::now();
    const double wall_diis = std::chrono::duration<double>(t3 - t2).count();
    const double e_diis = eval_diis.last_energy_hartree();
    const double d_ot_diis = std::fabs(e_ot - e_diis);
    std::printf("  DIIS+U E=%.12f Ha (%d it, wall=%.3f s) |OT-DIIS|=%.3e "
                "(tol %.1e)\n",
                e_diis, eval_diis.last_result().iters, wall_diis, d_ot_diis,
                kOtDiisTolHa);
    std::printf("  fair wall OT-only: %.3f s  (CP2K pin ~13.65 s)\n", wall_ot);
    ok = ok && eval_diis.last_result().converged && std::isfinite(e_diis);
    ok = ok && (d_ot_diis <= kOtDiisTolHa);

    if (!ok) {
        std::fprintf(stderr, "FAIL ot-dftu\n");
        return 1;
    }
    std::printf("PASS ot-dftu (OT+U vs CP2K + OT/DIIS parity)\n");
    return 0;
}
