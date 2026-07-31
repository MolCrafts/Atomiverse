/// @file regressions/gpw/gpw-vdw-uks/reg_gpw-vdw-uks.cpp
/// @brief Γ collinear-UKS nonlocal vdW-DF (DRSLL): O₂ triplet LSD vs offline
///        CP2K 2026.1 pin + closed-shell UKS-vs-RKS spin-degeneracy identity.
///
/// Public-API consumer only. V_nl is a functional of the TOTAL density
/// ρ = ρα + ρβ, added identically to both spin potentials each SCF
/// iteration; E_c^nl lands on GpwResult::e_disp. Oracle numbers in
/// REFERENCE.md (offline CP2K only; never live in ctest).
///
/// Exit non-zero on failure. Wall ~3–4 min Release (8 threads) — SC
/// nonlocal RPS every SCF iter on O₂@400 Ry; family precedent
/// `gpw-uks-pbe` ~177 s / `gpw-vdw-scf-potential` ~121 s.

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

// O₂, R = 1.2075 Å centered in a 12 Å cubic PERIODIC cell (Å) — the
// gpw-uks-pbe twin geometry moved to PERIODIC XYZ (RPS nonlocal kernel).
ATV::cpu::Frame make_o2() {
    ATV::cpu::Frame frame(2);
    constexpr double L = 12.0;
    constexpr double R = 1.2075;
    for (int i = 0; i < 2; ++i) {
        frame.span(ATV::kAtom_type)[i] = 8;
        frame.span(ATV::kAtom_x)[i] = 0.5 * L;
        frame.span(ATV::kAtom_y)[i] = 0.5 * L;
    }
    frame.span(ATV::kAtom_z)[0] = 0.5 * L - 0.5 * R;
    frame.span(ATV::kAtom_z)[1] = 0.5 * L + 0.5 * R;
    frame.box = ATV::cpu::Box::cubic(L);
    return frame;
}

// Same water dimer as gpw-vdw-df-ice / gpw-vdw-scf-potential (Å), 12 Å cell.
ATV::cpu::Frame make_water_dimer() {
    ATV::cpu::Frame frame(6);
    const int z[6] = {8, 1, 1, 8, 1, 1};
    const double x[6] = {0.0, 0.758602, 0.758602, 3.0, 3.758602, 3.758602};
    const double y[6] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    const double zc[6] = {0.0, 0.504284, -0.504284, 0.0, 0.504284, -0.504284};
    for (int i = 0; i < 6; ++i) {
        frame.span(ATV::kAtom_type)[i] = z[i];
        frame.span(ATV::kAtom_x)[i] = x[i];
        frame.span(ATV::kAtom_y)[i] = y[i];
        frame.span(ATV::kAtom_z)[i] = zc[i];
    }
    frame.box = ATV::cpu::Box::cubic(12.0);
    return frame;
}

} // namespace

int main() {
    using namespace ATV;
    using clock = std::chrono::steady_clock;

    // ── Part A: O₂ triplet Γ-UKS PBE + DRSLL vs offline CP2K 2026.1 LSD ──
    // CP2K 2026.1 (rev 85b8a9b49c) LSD PBE + NON_LOCAL TYPE DRSLL, CUTOFF
    // 400 / REL_CUTOFF 60 / NGRIDS 4 — REFERENCE.md.
    constexpr double kEtotCp2k = -31.81565852221005;   // Ha
    constexpr double kEDispCp2k = 1.1311983030413e-01; // Ha (SC E_c^nl)
    constexpr double kTolEtot = 1.0e-3;  // Ha (single-grid vs NGRIDS-4 twin)
    constexpr double kTolDisp = 1.0e-4;  // Ha (vdW-family precedent)

    const cpu::gpw::Kind O{
        molrs::Element::O,
        cpu::gpw::GaussianBasis(molrs::Element::O, "DZVP-MOLOPT-GTH"),
        cpu::gpw::GthPotential(molrs::Element::O, "GTH-PBE-q6")};

    cpu::Frame frame_o2 = make_o2();
    cpu::gpw::GPWEvaluator eval_o2({O});
    eval_o2.cutoff_ry = 400.0;
    eval_o2.max_iter = 200;
    eval_o2.tol = 1e-7;
    eval_o2.mix = 0.4;
    eval_o2.xc = cpu::gpw::XcKind::PBE;
    eval_o2.n_alpha = 7; // multiplicity 3, 12 valence e⁻
    eval_o2.n_beta = 5;
    eval_o2.vdw_df = true;

    const auto t0 = clock::now();
    cpu::single_point(frame_o2, eval_o2);
    const auto t1 = clock::now();

    const cpu::gpw::GpwResult& r_o2 = eval_o2.last_result();
    const double e_o2 = eval_o2.last_energy_hartree();
    const double e_frame = frame_o2.meta.get(kEnergy) / units::kHartreeInEv;

    std::printf("A: E(O2 UKS PBE+DRSLL) = %.12f Ha (CP2K %.12f, |dE|=%.3e, "
                "tol %.0e)\n",
                e_o2, kEtotCp2k, std::fabs(e_o2 - kEtotCp2k), kTolEtot);
    std::printf("A: E_disp              = %.12e Ha (CP2K %.12e, |dD|=%.3e, "
                "tol %.0e)\n",
                r_o2.e_disp, kEDispCp2k, std::fabs(r_o2.e_disp - kEDispCp2k),
                kTolDisp);
    std::printf("A: converged=%d iters=%d wall=%.1f s\n",
                static_cast<int>(r_o2.converged), r_o2.iters,
                std::chrono::duration<double>(t1 - t0).count());

    bool ok = r_o2.converged && std::isfinite(e_o2)
              && std::isfinite(r_o2.e_disp);
    ok = ok && std::fabs(e_o2 - kEtotCp2k) <= kTolEtot;
    ok = ok && std::fabs(r_o2.e_disp - kEDispCp2k) <= kTolDisp;
    ok = ok && std::fabs(e_frame - e_o2) < 1e-9; // Frame carries eV energy

    // ── Part B: closed-shell spin-degeneracy identity (water dimer) ──────
    // 16 valence e⁻ (2×O-q6 + 4×H-q1). RKS n_occ = 8 (double-occupied);
    // UKS nα = nβ = 8 must reproduce it with vdW-DF: same total energy
    // (≤ 3e-4 eV) and same SC E_c^nl (≤ 1e-5 Ha).
    constexpr double kTolIdentityEv = 3.0e-4; // eV
    constexpr double kTolIdentityDisp = 1.0e-5; // Ha

    const cpu::gpw::Kind Osr{
        molrs::Element::O,
        cpu::gpw::GaussianBasis(molrs::Element::O, "DZVP-MOLOPT-SR-GTH"),
        cpu::gpw::GthPotential(molrs::Element::O, "GTH-PBE-q6")};
    const cpu::gpw::Kind Hsr{
        molrs::Element::H,
        cpu::gpw::GaussianBasis(molrs::Element::H, "DZVP-MOLOPT-SR-GTH"),
        cpu::gpw::GthPotential(molrs::Element::H, "GTH-PBE-q1")};

    auto make_dimer_eval = [&]() {
        cpu::gpw::GPWEvaluator ev({Osr, Hsr});
        ev.cutoff_ry = 200.0;
        ev.max_iter = 120;
        ev.tol = 1e-7;
        ev.mix = 0.4;
        ev.xc = cpu::gpw::XcKind::PBE;
        ev.vdw_df = true;
        return ev;
    };

    // RKS: leave n_alpha/n_beta at 0 → evaluator auto-sets n_occ = valence/2.
    cpu::Frame frame_rks = make_water_dimer();
    cpu::gpw::GPWEvaluator eval_rks = make_dimer_eval();
    cpu::single_point(frame_rks, eval_rks);

    // UKS closed-shell: nα = nβ = 8 (16 valence electrons, spin-paired).
    cpu::Frame frame_uks = make_water_dimer();
    cpu::gpw::GPWEvaluator eval_uks = make_dimer_eval();
    eval_uks.n_alpha = 8;
    eval_uks.n_beta = 8;
    cpu::single_point(frame_uks, eval_uks);
    const auto t2 = clock::now();

    const double e_rks = eval_rks.last_energy_hartree();
    const double e_uks = eval_uks.last_energy_hartree();
    const double d_rks = eval_rks.last_result().e_disp;
    const double d_uks = eval_uks.last_result().e_disp;
    const double de_ev = std::fabs(e_uks - e_rks) * units::kHartreeInEv;

    std::printf("B: E_RKS=%.12f  E_UKS=%.12f Ha  |dE|=%.3e eV (tol %.0e)\n",
                e_rks, e_uks, de_ev, kTolIdentityEv);
    std::printf("B: D_RKS=%.10e  D_UKS=%.10e Ha  |dD|=%.3e (tol %.0e)\n",
                d_rks, d_uks, std::fabs(d_uks - d_rks), kTolIdentityDisp);
    std::printf("B: conv RKS=%d UKS=%d  wall(B)=%.1f s\n",
                static_cast<int>(eval_rks.last_result().converged),
                static_cast<int>(eval_uks.last_result().converged),
                std::chrono::duration<double>(t2 - t1).count());

    ok = ok && eval_rks.last_result().converged
         && eval_uks.last_result().converged;
    ok = ok && de_ev <= kTolIdentityEv;
    ok = ok && std::fabs(d_uks - d_rks) <= kTolIdentityDisp;
    ok = ok && std::isfinite(d_uks) && d_uks != 0.0; // E_c^nl actually folded

    if (!ok) {
        std::fprintf(stderr, "FAIL gpw-vdw-uks\n");
        return 1;
    }
    std::printf("PASS gpw-vdw-uks\n");
    return 0;
}
