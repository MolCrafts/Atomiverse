/// @file regressions/gpw/gpw-dftu-nio/reg_gpw-dftu-nio.cpp
/// @brief NiO (molecule) Γ-UKS GPW + Dudarev DFT+U vs offline CP2K PLUS_U pin.
///
/// Public-API consumer only. Geometry and oracle numbers in REFERENCE.md
/// (CP2K offline; no live CP2K in ctest).
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
    // multiplicity 3, 24 valence e⁻ → Nα=13, Nβ=11
    eval.n_alpha = 13;
    eval.n_beta = 11;
    eval.plus_u.push_back({molrs::Element::Ni, /*l=*/2, /*u_ev=*/6.0});
    // Occupations already break spin symmetry (13,11); no AFM seed needed.

    const auto t0 = clock::now();
    cpu::single_point(frame, eval);
    const auto t1 = clock::now();
    const double wall_s = std::chrono::duration<double>(t1 - t0).count();

    const cpu::gpw::GpwResult& r = eval.last_result();
    const double e_ha = eval.last_energy_hartree();
    const double e_frame = frame.meta.get(kEnergy) / units::kHartreeInEv;
    const double s2 = eval.last_s2();

    // CP2K 2026.1 offline pin — REFERENCE.md (OT twin; SZV / PADE / U=6 eV)
    constexpr double kE_cp2k = -184.92466076474312;
    constexpr double kTolE =
        5.0e-2; // Ha — first green (single-grid vs multigrid+OT)
    constexpr double kM_Ni_cp2k = 1.177632;
    constexpr double kTolM = 0.5;
    constexpr double kS2_cp2k = 2.019198;
    constexpr double kTolS2 = 0.25;

    std::printf("E(NiO mol, PADE+U/SZV, MT) = %.12f Ha\n", e_ha);
    std::printf("  e_dftu = %.8f Ha\n", r.e_dftu);
    std::printf("  CP2K pin                     = %.12f Ha  |dE|=%.3e "
                "(tol %.1e)\n",
                kE_cp2k, std::fabs(e_ha - kE_cp2k), kTolE);
    std::printf("  <S2> = %.6f  (CP2K %.6f)\n", s2, kS2_cp2k);
    std::printf("  converged=%d iters=%d wall=%.3f s\n",
                static_cast<int>(r.converged), r.iters, wall_s);

    bool ok = r.converged && std::isfinite(e_ha) && std::isfinite(s2);
    ok = ok && std::isfinite(e_frame) && std::fabs(e_frame - e_ha) < 1e-9;
    ok = ok && std::fabs(e_ha - kE_cp2k) <= kTolE;
    ok = ok && std::fabs(s2 - kS2_cp2k) <= kTolS2;

    if (r.mulliken_spin.size() >= 1) {
        const double mNi = r.mulliken_spin[0];
        std::printf("  m_Ni = %.4f  (CP2K %.4f, tol %.2f)\n", mNi, kM_Ni_cp2k,
                    kTolM);
        ok = ok && (mNi > 0.3);
        ok = ok && (std::fabs(mNi - kM_Ni_cp2k) <= kTolM);
    } else {
        std::fprintf(stderr, "FAIL: empty mulliken_spin\n");
        ok = false;
    }

    // DFT+U should be a non-trivial positive contribution for partially filled d
    ok = ok && std::isfinite(r.e_dftu) && (r.e_dftu > 1e-4);

    if (!ok) {
        std::fprintf(stderr, "FAIL gpw-dftu-nio\n");
        return 1;
    }
    std::printf("PASS gpw-dftu-nio\n");
    return 0;
}
