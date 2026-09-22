/// @file regressions/dynamics/xl-bomd-linearized-stress/reg_xl-bomd-linearized-stress.cpp
/// @brief END-TO-END: zero-SCF linearized XL-BOMD with analytic fixed-P
///        **shadow-density stress** (the product path that lifts the old
///        `compute_stress` refusal). Periodic H₂ NVE, four legs:
///        A = converged BOMD multi-k (CP2K-anchored stress trace),
///        B = linearized multi-k (product under test; warm iters == 0 + σ),
///        C = converged BOMD Γ, D = linearized Γ.
///
/// Dual gate at the seed geometry: state-overload vs classic gpw_stress
/// (identity) vs gpw_stress_fd. See REFERENCE.md. Public API only.
/// Requires ATV_BUILD_GPW + MD. Exit non-zero on failure.

#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <vector>

#include "atomiverse/core/frame_schema.h"
#include "atomiverse/core/stage.h"
#include "atomiverse/core/units.h"
#include "atomiverse/cpu/dynamics/md.h"
#include "atomiverse/cpu/dynamics/velocity_verlet.h"
#include "atomiverse/cpu/electronic/xl_bomd_force.h"
#include "atomiverse/cpu/evaluate/single_point.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/gpw/gaussian_basis.h"
#include "atomiverse/cpu/gpw/gpw_evaluator.h"
#include "atomiverse/cpu/gpw/gpw_solver.h"
#include "atomiverse/cpu/gpw/gth_potential.h"

using namespace ATV;

namespace {

constexpr double kBohr2Ang = 0.52917720859;
constexpr double kLBohr = 12.0;
constexpr double kR0Bohr = 1.0;
constexpr double kDt = 0.5; // fs
constexpr int kProd = 4;

// CP2K 2026.1 STRESS_TENSOR ANALYTICAL diagonals [Ha/Bohr³] from
// tools/cp2k_oracle/inputs/xl-bomd-linearized-stress (2026-07-31).
// Source: PROJECT-1.stress [bar] / 2.94210107994716e8.
// Voigt order per step: xx, yy, zz.
constexpr double kCp2kSxx[kProd + 1] = {1.998440567215e-04, 1.701541305617e-04,
                                        1.087027341964e-04, 4.492363435060e-05,
                                        -8.093268435310e-06};
constexpr double kCp2kSyy[kProd + 1] = {
    -7.010935192196e-08, -7.810856540460e-08, -9.651444883875e-08,
    -1.194770631848e-07, -1.406572107551e-07};
constexpr double kCp2kSzz[kProd + 1] = {
    -2.386241961179e-09, -2.134981164927e-09, -1.350517576735e-09,
    2.382815718937e-11, 1.879278928819e-09};

cpu::Frame make_frame() {
    cpu::Frame frame(2);
    const double L_ang = kLBohr * kBohr2Ang;
    const double x0[2] = {(kLBohr / 2.0 - 0.5 * kR0Bohr) * kBohr2Ang,
                          (kLBohr / 2.0 + 0.5 * kR0Bohr) * kBohr2Ang};
    for (int i = 0; i < 2; ++i) {
        frame.span(kAtom_type)[i] = 1;
        frame.span(kAtom_mass)[i] = 1.008;
        frame.span(kAtom_tag)[i] = i + 1;
        frame.span(kAtom_x)[i] = x0[i];
        frame.span(kAtom_y)[i] = L_ang / 2.0;
        frame.span(kAtom_z)[i] = L_ang / 2.0;
        frame.span(kAtom_vx)[i] = 0.0;
        frame.span(kAtom_vy)[i] = 0.0;
        frame.span(kAtom_vz)[i] = 0.0;
    }
    frame.span(kAtom_vx)[0] = -0.0010;
    frame.span(kAtom_vx)[1] = 0.0010;
    frame.box = cpu::Box::cubic(L_ang);
    return frame;
}

cpu::gpw::GPWEvaluator make_gpw(bool multi_k) {
    const cpu::gpw::Kind H{
        molrs::Element::H,
        cpu::gpw::GaussianBasis(molrs::Element::H, "DZV-GTH-PADE"),
        cpu::gpw::GthPotential(molrs::Element::H, "GTH-PADE-q1")};
    cpu::gpw::GPWEvaluator gpw(
        /*kinds*/ {H}, /*cutoff_ry*/ 300.0, /*max_iter*/ 60, /*tol*/ 1e-7,
        /*mix*/ 0.3, /*xc*/ cpu::gpw::XcKind::PADE, /*nonperiodic*/ false,
        /*compute_forces*/ true);
    gpw.compute_stress = true;
    gpw.stress_use_fd = false; // product analytic fixed-P (shadow on warm)
    if (multi_k) {
        gpw.k_mesh[0] = 1;
        gpw.k_mesh[1] = 1;
        gpw.k_mesh[2] = 2;
        gpw.k_time_reversal = false; // two sampled k (CP2K FULL_GRID)
    }
    return gpw;
}

struct LegResult {
    std::vector<double> pot_ha;
    std::vector<std::array<double, 6>> stress_ha; // last_stress_ha_bohr3
    std::vector<double> etot_ev;
    double r_hh = 0.0;
    int seed_iters = 0;
    int max_warm_iters = 0;
    bool warm_all_converged = true;
    int n_kpts = 0;
    double wall_s = 0.0;
};

bool run_leg(const char* name, bool multi_k, bool linearized, LegResult& out) {
    cpu::Frame frame = make_frame();
    cpu::gpw::GPWEvaluator base = make_gpw(multi_k);
    cpu::electronic::XlBomdForce eval =
        linearized
            ? cpu::electronic::XlBomdForce{std::move(base), 5,
                                           cpu::electronic::XlBomdLinearized{}}
            : cpu::electronic::XlBomdForce{std::move(base), 5};

    const auto t0 = std::chrono::steady_clock::now();
    cpu::md::MD sim;
    sim.set_integrator(cpu::md::VelocityVerlet{kDt});
    sim.set_evaluator_ref(eval);
    sim.enable_timing(false);
    sim.build();
    // Prime lifecycle BEFORE seed (GPWEvaluator::build resets XL histories).
    if (sim.run(frame, 0) != 0) {
        std::printf("FAIL[%s]: 0-step priming run stepped\n", name);
        return false;
    }
    cpu::single_point(frame, eval);
    if (!(eval.last_result().iters > 0) || !eval.last_result().converged) {
        std::printf("FAIL[%s]: seed must be full SCF (iters=%d conv=%d)\n",
                    name, eval.last_result().iters,
                    static_cast<int>(eval.last_result().converged));
        return false;
    }
    out.seed_iters = eval.last_result().iters;
    out.n_kpts = eval.last_result().n_kpts;
    if (multi_k && out.n_kpts < 2) {
        std::printf("FAIL[%s]: expected n_kpts >= 2, got %d\n", name,
                    out.n_kpts);
        return false;
    }
    // Seed stress (classic path through bottom stress block).
    const auto& s0 = eval.method().last_stress_ha_bohr3();
    for (int k = 0; k < 6; ++k) {
        if (!std::isfinite(s0[static_cast<std::size_t>(k)])) {
            std::printf("FAIL[%s]: non-finite seed stress component %d\n", name,
                        k);
            return false;
        }
    }
    if (std::fabs(s0[3]) > 1e-14 || std::fabs(s0[4]) > 1e-14
        || std::fabs(s0[5]) > 1e-14) {
        std::printf("FAIL[%s]: seed off-diagonal stress not exactly 0\n", name);
        return false;
    }
    // Frame kStress written [eV/Å³].
    const auto& st0 = frame.meta.get(kStress);
    if (!std::isfinite(st0[0])) {
        std::printf("FAIL[%s]: frame kStress not written at seed\n", name);
        return false;
    }

    const double pe0 = eval.last_result().e_total * units::kHartreeInEv;
    double ke0 = 0.0;
    for (int i = 0; i < 2; ++i) {
        const double vx = static_cast<double>(frame.span(kAtom_vx)[i]);
        ke0 += 0.5 * 1.008 * vx * vx / units::force_to_accel;
    }
    out.pot_ha.push_back(eval.last_result().e_total);
    out.stress_ha.push_back(s0);
    out.etot_ev.push_back(pe0 + ke0);

    for (int s = 0; s < kProd; ++s) {
        if (sim.run(frame, 1) != 1) {
            std::printf("FAIL[%s]: production step %d incomplete\n", name, s);
            return false;
        }
        const double pe = frame.meta.get(kEnergy);
        const double ke = frame.meta.get(kKinetic);
        if (!std::isfinite(pe) || !std::isfinite(ke) || !(ke >= 0.0)) {
            std::printf("FAIL[%s]: non-finite energy at step %d PE=%g KE=%g\n",
                        name, s, pe, ke);
            return false;
        }
        out.max_warm_iters =
            std::max(out.max_warm_iters, eval.last_result().iters);
        out.warm_all_converged =
            out.warm_all_converged && eval.last_result().converged;
        const auto& sh = eval.method().last_stress_ha_bohr3();
        for (int k = 0; k < 6; ++k) {
            if (!std::isfinite(sh[static_cast<std::size_t>(k)])) {
                std::printf("FAIL[%s]: non-finite stress step %d c %d\n", name,
                            s + 1, k);
                return false;
            }
        }
        if (std::fabs(sh[3]) > 1e-14 || std::fabs(sh[4]) > 1e-14
            || std::fabs(sh[5]) > 1e-14) {
            std::printf("FAIL[%s]: off-diagonal stress nonzero step %d\n", name,
                        s + 1);
            return false;
        }
        if (!std::isfinite(frame.meta.get(kStress)[0])) {
            std::printf("FAIL[%s]: kStress missing at step %d\n", name, s + 1);
            return false;
        }
        out.pot_ha.push_back(pe / units::kHartreeInEv);
        out.stress_ha.push_back(sh);
        out.etot_ev.push_back(pe + ke);
        for (int i = 0; i < 2; ++i)
            if (!std::isfinite(static_cast<double>(frame.span(kAtom_x)[i]))
                || !std::isfinite(static_cast<double>(frame.span(kAtom_y)[i]))
                || !std::isfinite(
                    static_cast<double>(frame.span(kAtom_z)[i]))) {
                std::printf("FAIL[%s]: non-finite coordinate step %d atom %d\n",
                            name, s, i);
                return false;
            }
    }
    out.wall_s =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - t0)
            .count();

    const double dx =
        static_cast<double>(frame.span(kAtom_x)[1] - frame.span(kAtom_x)[0]);
    const double dy =
        static_cast<double>(frame.span(kAtom_y)[1] - frame.span(kAtom_y)[0]);
    const double dz =
        static_cast<double>(frame.span(kAtom_z)[1] - frame.span(kAtom_z)[0]);
    out.r_hh = std::sqrt(dx * dx + dy * dy + dz * dz);
    return true;
}

double worst_rel_drift(const LegResult& leg) {
    const double e0 = leg.etot_ev.front();
    const double scale = std::max(1.0, std::fabs(e0));
    double worst = 0.0;
    for (double e : leg.etot_ev)
        worst = std::max(worst, std::fabs(e - e0) / scale);
    return worst;
}

/// Seed dual gate: state overload vs classic analytic vs multi-SCF FD at the
/// initial H₂ geometry (k mesh of leg A).
bool dual_gate_seed(double& id_err, double& an_fd_err) {
    const cpu::gpw::GaussianBasis basis(molrs::Element::H, "DZV-GTH-PADE");
    const cpu::gpw::GthPotential pot(molrs::Element::H, "GTH-PADE-q1");
    const double rA[3] = {kLBohr / 2.0 - 0.5 * kR0Bohr, kLBohr / 2.0,
                          kLBohr / 2.0};
    const double rB[3] = {kLBohr / 2.0 + 0.5 * kR0Bohr, kLBohr / 2.0,
                          kLBohr / 2.0};
    const std::vector<cpu::gpw::GpwAtom> atoms = {
        cpu::gpw::GpwAtom(rA, basis, pot, 1),
        cpu::gpw::GpwAtom(rB, basis, pot, 1)};

    auto make_cfg = []() {
        cpu::gpw::GpwDft m;
        m.cell[0] = m.cell[1] = m.cell[2] = kLBohr;
        m.cutoff_ry = 300.0;
        m.n_occ = 1;
        m.max_iter = 60;
        m.tol = 1e-7;
        m.mix = 0.3;
        m.xc = cpu::gpw::XcKind::PADE;
        m.eps_default = 1e-10;
        m.k_mesh[0] = 1;
        m.k_mesh[1] = 1;
        m.k_mesh[2] = 2;
        m.k_time_reversal = false;
        return m;
    };

    const double h = 2e-3;
    cpu::gpw::GpwDft solver = make_cfg();
    solver.build(atoms);
    const auto state = solver.run(atoms);
    if (!state.converged) {
        std::printf("FAIL[dual]: seed SCF did not converge\n");
        return false;
    }
    const auto s_state = cpu::gpw::gpw_stress(atoms, make_cfg(), state, h);
    const auto s_classic = cpu::gpw::gpw_stress(atoms, make_cfg(), h);
    const auto s_fd = cpu::gpw::gpw_stress_fd(atoms, make_cfg(), h);
    id_err = 0.0;
    an_fd_err = 0.0;
    for (int a = 0; a < 3; ++a) {
        id_err = std::max(id_err,
                          std::fabs(s_state[static_cast<std::size_t>(a)]
                                    - s_classic[static_cast<std::size_t>(a)]));
        an_fd_err =
            std::max(an_fd_err, std::fabs(s_state[static_cast<std::size_t>(a)]
                                          - s_fd[static_cast<std::size_t>(a)]));
        std::printf("  dual axis %d  state=%.6e classic=%.6e fd=%.6e\n", a,
                    s_state[static_cast<std::size_t>(a)],
                    s_classic[static_cast<std::size_t>(a)],
                    s_fd[static_cast<std::size_t>(a)]);
    }
    return true;
}

} // namespace

int main() {
    // Tolerances — measured first-green values go into REFERENCE.md.
    // CP2K σ_xx band: seed engine-model residual is ~3e-6 Ha/Bohr³; along the
    // aggressive 0.5 fs compressed-bond NVE the engines' geometries diverge
    // and the absolute residual grows to the low-1e-5 class (measured first-
    // green worst 6.026e-5). Soft absolute band with trajectory margin.
    constexpr double cp2k_sxx_band = 1e-4; // Ha/Bohr³
    // A/B and C/D shadow stress envelope — measured first-green ~2e-6.
    constexpr double ab_stress_band = 1e-5; // Ha/Bohr³
    constexpr double state_id_band = 1e-10; // Ha/Bohr³
    constexpr double an_fd_band = 1e-6;     // Ha/Bohr³ (spec dual gate)
    constexpr double drift_tol = 0.05;
    constexpr double rmin = 0.4;
    constexpr double rmax = 1.5;

    double id_err = 0.0, an_fd_err = 0.0;
    if (!dual_gate_seed(id_err, an_fd_err))
        return 1;
    std::printf("  dual |state-classic|=%.3e  |state-fd|=%.3e Ha/Bohr3\n",
                id_err, an_fd_err);
    if (!(id_err <= state_id_band)) {
        std::printf("FAIL: state-overload identity vs classic out of band\n");
        return 1;
    }
    if (!(an_fd_err <= an_fd_band)) {
        std::printf("FAIL: analytic/FD dual gate out of band\n");
        return 1;
    }

    LegResult A, B, C, D;
    if (!run_leg("A bomd-k", /*multi_k=*/true, /*linearized=*/false, A))
        return 1;
    if (!run_leg("B lin-k", /*multi_k=*/true, /*linearized=*/true, B))
        return 1;
    if (!run_leg("C bomd-G", /*multi_k=*/false, /*linearized=*/false, C))
        return 1;
    if (!run_leg("D lin-G", /*multi_k=*/false, /*linearized=*/true, D))
        return 1;

    // Zero-SCF warm steps on product legs.
    if (B.max_warm_iters != 0 || !B.warm_all_converged) {
        std::printf("FAIL: B linearized warm (iters=%d conv=%d)\n",
                    B.max_warm_iters, static_cast<int>(B.warm_all_converged));
        return 1;
    }
    if (D.max_warm_iters != 0 || !D.warm_all_converged) {
        std::printf("FAIL: D linearized warm (iters=%d conv=%d)\n",
                    D.max_warm_iters, static_cast<int>(D.warm_all_converged));
        return 1;
    }
    if (A.max_warm_iters <= 0 || C.max_warm_iters <= 0) {
        std::printf("FAIL: BOMD anchors should iterate (A=%d C=%d)\n",
                    A.max_warm_iters, C.max_warm_iters);
        return 1;
    }

    double worst_cp2k = 0.0, worst_ab = 0.0, worst_cd = 0.0;
    for (int s = 0; s <= kProd; ++s) {
        const std::size_t k = static_cast<std::size_t>(s);
        // Gate primarily on σ_xx (dominant compressed-bond component); yy/zz
        // are near-zero noise on this orthorhombic stretch trajectory.
        worst_cp2k =
            std::max(worst_cp2k, std::fabs(A.stress_ha[k][0] - kCp2kSxx[s]));
        for (int a = 0; a < 3; ++a) {
            worst_ab = std::max(
                worst_ab,
                std::fabs(B.stress_ha[k][static_cast<std::size_t>(a)]
                          - A.stress_ha[k][static_cast<std::size_t>(a)]));
            worst_cd = std::max(
                worst_cd,
                std::fabs(D.stress_ha[k][static_cast<std::size_t>(a)]
                          - C.stress_ha[k][static_cast<std::size_t>(a)]));
        }
    }
    const double drift_b = worst_rel_drift(B);
    const double drift_d = worst_rel_drift(D);

    std::printf("XL-BOMD linearized stress H2 NVE (K=5, 1 seed + %d steps):\n",
                kProd);
    std::printf("  A bomd-k:  seed=%d warm<=%d drift=%.3e r=%.3f wall=%.2fs "
                "nk=%d\n",
                A.seed_iters, A.max_warm_iters, worst_rel_drift(A), A.r_hh,
                A.wall_s, A.n_kpts);
    std::printf("  B lin-k:   seed=%d warm=%d  drift=%.3e r=%.3f wall=%.2fs "
                "nk=%d\n",
                B.seed_iters, B.max_warm_iters, drift_b, B.r_hh, B.wall_s,
                B.n_kpts);
    std::printf("  C bomd-G:  seed=%d warm<=%d drift=%.3e r=%.3f wall=%.2fs\n",
                C.seed_iters, C.max_warm_iters, worst_rel_drift(C), C.r_hh,
                C.wall_s);
    std::printf("  D lin-G:   seed=%d warm=%d  drift=%.3e r=%.3f wall=%.2fs\n",
                D.seed_iters, D.max_warm_iters, drift_d, D.r_hh, D.wall_s);
    for (int s = 0; s <= kProd; ++s) {
        const std::size_t k = static_cast<std::size_t>(s);
        std::printf("  step %d  σ_xx[Ha/Bohr3]: cp2k=%.6e A=%.6e B=%.6e "
                    "C=%.6e D=%.6e\n",
                    s, kCp2kSxx[s], A.stress_ha[k][0], B.stress_ha[k][0],
                    C.stress_ha[k][0], D.stress_ha[k][0]);
    }
    std::printf("  worst |A.sxx - cp2k| = %.3e (band %.1e)\n", worst_cp2k,
                cp2k_sxx_band);
    std::printf("  worst |B - A| diag  = %.3e (band %.1e)\n", worst_ab,
                ab_stress_band);
    std::printf("  worst |D - C| diag  = %.3e (band %.1e)\n", worst_cd,
                ab_stress_band);
    // silence unused pins (yy/zz recorded in REFERENCE; not hard-gated)
    (void)kCp2kSyy;
    (void)kCp2kSzz;

    if (!(worst_cp2k <= cp2k_sxx_band)) {
        std::printf("FAIL: BOMD-k σ_xx left the CP2K stress band\n");
        return 1;
    }
    if (!(worst_ab <= ab_stress_band)) {
        std::printf("FAIL: linearized vs BOMD A/B stress out of band\n");
        return 1;
    }
    if (!(worst_cd <= ab_stress_band)) {
        std::printf("FAIL: linearized vs BOMD C/D stress out of band\n");
        return 1;
    }
    if (!(drift_b <= drift_tol) || !(drift_d <= drift_tol)) {
        std::printf("FAIL: NVE drift envelope (B=%.3e D=%.3e tol=%.2f)\n",
                    drift_b, drift_d, drift_tol);
        return 1;
    }
    for (const LegResult* leg : {&A, &B, &C, &D})
        if (!(leg->r_hh > rmin) || !(leg->r_hh < rmax)) {
            std::printf("FAIL: H–H out of window r=%.3f Å\n", leg->r_hh);
            return 1;
        }

    std::printf("PASS: xl-bomd-linearized-stress (zero-SCF shadow stress; "
                "CP2K-anchored)\n");
    return 0;
}
