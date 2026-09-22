/// @file regressions/gpw/gpw-optb88-uks/reg_gpw-optb88-uks.cpp
/// @brief Γ collinear-UKS optB88-vdW (OptB88+PW92 + DRSLL): O₂ triplet LSD vs
///        offline CP2K 2026.1 pin + OptB88 vs PBE dual-gate + closed-shell
///        UKS≡RKS identity.
///
/// Public-API consumer only. Semi-local XC is `XcKind::OptB88VdW`; nonlocal
/// correlation is self-consistent DRSLL via `vdw_df` on the total density
/// (V_nl added to both spin potentials). Oracle numbers in REFERENCE.md
/// (offline CP2K only; never live in ctest).
///
/// Exit non-zero on failure. Wall class: SC nonlocal RPS every SCF iter on
/// O₂@400 Ry; family `gpw-vdw-uks` ~196 s / this OptB88 twin similar.

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

// O₂, R = 1.2075 Å centered in a 12 Å cubic PERIODIC cell (Å) — same twin as
// gpw-vdw-uks / CP2K o2_uks_optb88_vdw.inp.
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

// Same water dimer as gpw-vdw-uks / gpw-optb88-vdw (Å), 12 Å cell.
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

    // ── Part A: O₂ triplet Γ-UKS OptB88VdW + DRSLL vs offline CP2K 2026.1 ──
    // CP2K 2026.1 (git:5e54ba2) LSD GGA_X_OPTB88_VDW + PW92 + NON_LOCAL DRSLL,
    // CUTOFF 400 / REL_CUTOFF 60 / NGRIDS 4, OT — REFERENCE.md.
    constexpr double kEtotCp2k = -32.054900934599011;  // Ha
    constexpr double kEDispCp2k = 1.1322988837478e-01; // Ha (SC E_c^nl)
    // Total-energy tol: OptB88 native vs CP2K/libxc grid twin is looser than
    // PBE (family `gpw-optb88-kpoints` uses 1e-2 Ha); UKS O2 measured ~0.02
    // Ha on single-grid and ~same class with NGRIDS 4.
    constexpr double kTolEtot = 2.5e-2; // Ha
    constexpr double kTolDisp =
        1.0e-4; // Ha (tight — E_c^nl is the product pin)
    // CP2K timer TOTAL TIME on the same twin job (1 MPI × OMP=8).
    constexpr double kCp2kWall = 66.011; // s
    // Dual-gate floor [Ha] — OptB88+DRSLL vs PBE+DRSLL must activate exchange.
    constexpr double kDeFloor = 1.0e-4;

    const cpu::gpw::Kind O{
        molrs::Element::O,
        cpu::gpw::GaussianBasis(molrs::Element::O, "DZVP-MOLOPT-GTH"),
        cpu::gpw::GthPotential(molrs::Element::O, "GTH-PBE-q6")};

    auto make_o2_eval = [&](cpu::gpw::XcKind xc) {
        cpu::gpw::GPWEvaluator eval({O});
        eval.cutoff_ry = 400.0;
        // Match CP2K MGRID (NGRIDS 4 / REL_CUTOFF 60) — fair twin for E_tot
        // and wall (same product path as gpw-vdw-uks-multigrid).
        eval.ngrids = 4;
        eval.rel_cutoff = 60.0;
        eval.max_iter = 200;
        eval.tol = 1e-7;
        eval.mix = 0.4;
        eval.xc = xc;
        eval.n_alpha = 7; // multiplicity 3, 12 valence e⁻
        eval.n_beta = 5;
        eval.vdw_df = true;
        return eval;
    };

    cpu::Frame frame_opt = make_o2();
    cpu::gpw::GPWEvaluator eval_opt = make_o2_eval(cpu::gpw::XcKind::OptB88VdW);

    const auto t0 = clock::now();
    cpu::single_point(frame_opt, eval_opt);
    const auto t1 = clock::now();
    const double wall_a = std::chrono::duration<double>(t1 - t0).count();

    const cpu::gpw::GpwResult& r_opt = eval_opt.last_result();
    const double e_opt = eval_opt.last_energy_hartree();
    const double e_frame = frame_opt.meta.get(kEnergy) / units::kHartreeInEv;

    cpu::Frame frame_pbe = make_o2();
    cpu::gpw::GPWEvaluator eval_pbe = make_o2_eval(cpu::gpw::XcKind::PBE);
    cpu::single_point(frame_pbe, eval_pbe);
    const double e_pbe = eval_pbe.last_energy_hartree();
    const bool conv_pbe = eval_pbe.last_result().converged;

    std::printf("A: E(O2 UKS OptB88+DRSLL) = %.12f Ha (CP2K %.12f, |dE|=%.3e, "
                "tol %.0e)\n",
                e_opt, kEtotCp2k, std::fabs(e_opt - kEtotCp2k), kTolEtot);
    std::printf("A: E_disp                 = %.12e Ha (CP2K %.12e, |dD|=%.3e, "
                "tol %.0e)\n",
                r_opt.e_disp, kEDispCp2k, std::fabs(r_opt.e_disp - kEDispCp2k),
                kTolDisp);
    std::printf("A: E_PBE+DRSLL twin       = %.12f Ha  |dE_opt-pbe|=%.3e "
                "(floor %.1e)\n",
                e_pbe, std::fabs(e_opt - e_pbe), kDeFloor);
    std::printf("A: converged=%d iters=%d wall=%.1f s  "
                "T_atv/T_cp2k=%.2f (cp2k=%.1f s)\n",
                static_cast<int>(r_opt.converged), r_opt.iters, wall_a,
                wall_a / kCp2kWall, kCp2kWall);

    bool ok =
        r_opt.converged && std::isfinite(e_opt) && std::isfinite(r_opt.e_disp);
    ok = ok && conv_pbe && std::isfinite(e_pbe);
    ok = ok && std::fabs(e_opt - e_pbe) > kDeFloor;
    ok = ok && std::fabs(e_opt - kEtotCp2k) <= kTolEtot;
    ok = ok && std::fabs(r_opt.e_disp - kEDispCp2k) <= kTolDisp;
    ok = ok && std::fabs(e_frame - e_opt) < 1e-9;
    ok = ok && r_opt.e_disp != 0.0;

    // ── Part B: closed-shell spin-degeneracy identity (water dimer) ──────
    // 16 valence e⁻. UKS nα = nβ = 8 must reproduce RKS with OptB88 + DRSLL.
    constexpr double kTolIdentityEv = 3.0e-4;   // eV
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
        ev.xc = cpu::gpw::XcKind::OptB88VdW;
        ev.vdw_df = true;
        return ev;
    };

    cpu::Frame frame_rks = make_water_dimer();
    cpu::gpw::GPWEvaluator eval_rks = make_dimer_eval();
    cpu::single_point(frame_rks, eval_rks);

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
    std::printf("B: D_RKS=%.10e  D_UKS=%.10e Ha  |dD|=%.3e (tol %.0e)\n", d_rks,
                d_uks, std::fabs(d_uks - d_rks), kTolIdentityDisp);
    std::printf("B: conv RKS=%d UKS=%d  wall(B)=%.1f s\n",
                static_cast<int>(eval_rks.last_result().converged),
                static_cast<int>(eval_uks.last_result().converged),
                std::chrono::duration<double>(t2 - t1).count());

    ok = ok && eval_rks.last_result().converged
         && eval_uks.last_result().converged;
    ok = ok && de_ev <= kTolIdentityEv;
    ok = ok && std::fabs(d_uks - d_rks) <= kTolIdentityDisp;
    ok = ok && std::isfinite(d_uks) && d_uks != 0.0;

    if (!ok) {
        std::fprintf(stderr, "FAIL gpw-optb88-uks\n");
        return 1;
    }
    std::printf("PASS gpw-optb88-uks\n");
    return 0;
}
