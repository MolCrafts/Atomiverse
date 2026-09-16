/// @file regressions/md/electrode-charge-predictor/reg_electrode-charge-predictor.cpp
/// @brief E2E: ASPC electrode charge predictor reduces CG iters vs cold start.
///
/// Public API only. Offline pin: regressions/md/electrode-charge-predictor/REFERENCE.md
/// (oracle = MetalWalls binary c8ad175d; the K=5 predictor coefficients are
/// the constants MetalWalls hard-codes in src/electrode_charge.F90:572-577).
/// Exits non-zero on failure.

#include <cmath>
#include <cstdio>
#include <vector>

#include "atomiverse/core/frame_schema.h"
#include "atomiverse/cpu/dynamics/const_pot_bo.h"
#include "atomiverse/cpu/dynamics/md.h"
#include "atomiverse/cpu/dynamics/velocity_verlet.h"
#include "atomiverse/cpu/electrodes/charge_aspc.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/neighbor/neighbor_list.h"
#include "atomiverse/cpu/potentials/pair_coul_cut.h"

using namespace ATV;

namespace {

/// Small madelung for CG-vs-dense pin (2 electrode sites).
cpu::Frame make_madelung() {
    cpu::Frame f(4);
    const double z[4] = {8.0, 12.0, 9.0, 11.0};
    const double q0[4] = {0.0, 0.0, 0.5, -0.5};
    const int typ[4] = {1, 2, 3, 3};
    for (int i = 0; i < 4; ++i) {
        f.span(kAtom_x)[i] = 0.0;
        f.span(kAtom_y)[i] = 0.0;
        f.span(kAtom_z)[i] = z[i];
        f.span(kAtom_charge)[i] = q0[i];
        f.span(kAtom_type)[i] = typ[i];
        f.span(kAtom_mass)[i] = (typ[i] < 3) ? 196.97 : 1.0;
        f.span(kAtom_tag)[i] = i + 1;
    }
    f.box = cpu::Box::orthorhombic(1.0, 1.0, 20.0);
    return f;
}

/// Multi-site parallel plate — enough electrodes for CG iters to matter.
cpu::Frame make_multi(int n_side = 3) {
    const int n_per = n_side * n_side;
    const int n_ions = 4;
    const int N = 2 * n_per + n_ions;
    cpu::Frame f(N);
    const double Lxy = 6.0;
    int idx = 0;
    for (int wall = 0; wall < 2; ++wall) {
        const double z = (wall == 0) ? 4.0 : 16.0;
        const int typ = (wall == 0) ? 1 : 2;
        for (int ix = 0; ix < n_side; ++ix)
            for (int iy = 0; iy < n_side; ++iy) {
                f.span(kAtom_x)[idx] = (ix + 0.5) * (Lxy / n_side);
                f.span(kAtom_y)[idx] = (iy + 0.5) * (Lxy / n_side);
                f.span(kAtom_z)[idx] = z;
                f.span(kAtom_type)[idx] = typ;
                f.span(kAtom_charge)[idx] = 0.0;
                f.span(kAtom_mass)[idx] = 12.0;
                f.span(kAtom_tag)[idx] = idx + 1;
                ++idx;
            }
    }
    for (int i = 0; i < n_ions; ++i) {
        const bool pos = (i % 2 == 0);
        f.span(kAtom_x)[idx] = (0.5 + (i % 2) * 0.4) * Lxy;
        f.span(kAtom_y)[idx] = (0.3 + (i / 2) * 0.4) * Lxy;
        f.span(kAtom_z)[idx] = pos ? 8.0 : 12.0;
        f.span(kAtom_type)[idx] = 3;
        f.span(kAtom_charge)[idx] = pos ? 0.5 : -0.5;
        f.span(kAtom_mass)[idx] = 1.0;
        f.span(kAtom_tag)[idx] = idx + 1;
        ++idx;
    }
    f.box = cpu::Box::orthorhombic(Lxy, Lxy, 20.0);
    return f;
}

cpu::md::ConstPotBO make_solver(int pred_order) {
    cpu::md::ConstPotBO c(2.0);
    c.add_electrode(1, 0.0);
    c.add_electrode(2, 1.0);
    c.electroneutrality = true;
    c.xy_image_range = 0;
    c.use_cg = true;
    c.cg_tol = 1e-10;
    c.max_cg_iters = 500;
    c.predictor_order = pred_order;
    return c;
}

/// One-shot BO solve (dt=0).
void solve_once(cpu::md::ConstPotBO& bo, cpu::Frame& f) {
    cpu::md::MD sim;
    sim.set_neighbor(cpu::md::NeighborList<>{12.0, 0.5});
    sim.set_evaluator(cpu::PairCoulCut{12.0});
    sim.set_integrator(cpu::md::VelocityVerlet{0.0});
    sim.set_observer_ref(bo);
    sim.build();
    sim.run(f, 1);
}

/// Multi-step ASPC path with geometry updates between steps (Frame-only).
/// Continuable MD::run(f,1) preserves BO/ASPC state; ~MD flushes destroy.
struct SteppedBo {
    cpu::md::ConstPotBO& bo;
    cpu::Frame& f;
    cpu::md::MD sim;

    SteppedBo(cpu::md::ConstPotBO& bo_, cpu::Frame& f_) : bo(bo_), f(f_) {
        sim.set_neighbor(cpu::md::NeighborList<>{12.0, 0.5});
        sim.set_evaluator(cpu::PairCoulCut{12.0});
        sim.set_integrator(cpu::md::VelocityVerlet{0.0});
        sim.set_observer_ref(bo);
        sim.build();
    }
    void step() { sim.run(f, 1); }
};

} // namespace

int main() {
    // ── ac-002: K=5 coefficients == MetalWalls' hard-coded constants ──
    // MetalWalls c8ad175d, src/electrode_charge.F90:572-577:
    //   B1=22/7  B2=-55/14  B3=55/21  B4=-22/21  B5=5/21  B6=-1/42
    // applied to the six stored charge vectors (QATOMS_NUM_STEPS = 7,
    // src/system.F90:107), newest first. ChargeASPC indexes oldest-first,
    // so coefficient(5-j+1)… → coefficient(6-j) is B_j.
    {
        const double B[6] = {22.0 / 7.0,   -55.0 / 14.0, 55.0 / 21.0,
                             -22.0 / 21.0, 5.0 / 21.0,   -1.0 / 42.0};
        cpu::electrodes::ChargeASPC mw(5);
        double sum = 0.0;
        for (int j = 1; j <= 6; ++j) {
            const double got = mw.coefficient(6 - j);
            const double want = B[j - 1];
            sum += got;
            if (std::fabs(got - want) > 1e-15 * std::fabs(want)) {
                std::printf("FAIL: ASPC K=5 B%d got %.17g expect %.17g\n", j,
                            got, want);
                return 1;
            }
        }
        if (std::fabs(sum - 1.0) > 1e-14) {
            std::printf("FAIL: ASPC K=5 coeffs sum to %.17g, expect 1\n", sum);
            return 1;
        }
        std::printf("ASPC K=5 == MetalWalls B1..B6 "
                    "{22/7,-55/14,55/21,-22/21,5/21,-1/42}, sum=%.17g\n",
                    sum);
    }

    // ── ac-003: K=2 coefficients + degree-1 exactness, degree-2 damping ──
    {
        cpu::electrodes::ChargeASPC aspc(2);
        // Kolafa ASPC K=2 is {5/2, -2, 1/2} newest → oldest — NOT the Lagrange
        // extrapolation {3, -3, 1} that the paper shows to be unstable.
        if (!(aspc.coefficient(0) == 0.5 && aspc.coefficient(1) == -2.0
              && aspc.coefficient(2) == 2.5)) {
            std::printf(
                "FAIL: ASPC K=2 coeffs got {%g,%g,%g} expect {0.5,-2,2.5}\n",
                aspc.coefficient(0), aspc.coefficient(1), aspc.coefficient(2));
            return 1;
        }
        for (int n = 0; n <= 2; ++n)
            aspc.record(std::vector<double>{static_cast<double>(n)});
        std::vector<double> pred;
        if (!aspc.predict(pred) || pred.size() != 1
            || std::fabs(pred[0] - 3.0) > 1e-12) {
            std::printf("FAIL: linear ASPC predict (got %g)\n",
                        pred.empty() ? -1.0 : pred[0]);
            return 1;
        }
        // Degree-2 history is damped, not reproduced (8, not the Lagrange 9).
        cpu::electrodes::ChargeASPC quad(2);
        for (int n = 0; n <= 2; ++n)
            quad.record(std::vector<double>{static_cast<double>(n * n)});
        std::vector<double> pq;
        if (!quad.predict(pq) || std::fabs(pq[0] - 8.0) > 1e-12) {
            std::printf("FAIL: quadratic ASPC damping (got %g, expect 8)\n",
                        pq.empty() ? -1.0 : pq[0]);
            return 1;
        }
        std::printf("ASPC K=2: omega={0.5,-2,2.5}, linear exact, "
                    "quadratic damped to 8\n");
    }

    // ── ac-004: CG matches dense on madelung ─────────────────────────
    {
        auto f_d = make_madelung();
        auto f_c = make_madelung();
        cpu::md::ConstPotBO dense(2.0);
        dense.add_electrode(1, 0.0).add_electrode(2, 1.0);
        dense.use_cg = false;
        dense.xy_image_range = 0;
        cpu::md::ConstPotBO cg = make_solver(/*pred_order=*/0);
        solve_once(dense, f_d);
        solve_once(cg, f_c);
        const double dQ = std::fabs(cg.total_charge(1) - dense.total_charge(1));
        if (dQ > 1e-8) {
            std::printf("FAIL: CG vs dense |ΔQ_top|=%.3e > 1e-8\n", dQ);
            return 1;
        }
        std::printf("CG vs dense: Q_top dense=%.8e cg=%.8e Δ=%.2e\n",
                    dense.total_charge(1), cg.total_charge(1), dQ);
    }

    // ── ac-005 / ac-006: predictor reduces mean CG iters ─────────────
    // Two warm solvers: K=2 (cheapest useful) and K=5 (MetalWalls' order).
    {
        auto cold = make_solver(0);
        auto warm2 = make_solver(2);
        auto warm5 = make_solver(5);
        auto f_cold = make_multi(3);
        auto f_warm2 = make_multi(3);
        auto f_warm5 = make_multi(3);
        SteppedBo run_cold(cold, f_cold);
        SteppedBo run_w2(warm2, f_warm2);
        SteppedBo run_w5(warm5, f_warm5);

        const int n = f_cold.nlocal();
        const int ion0 = n - 4;
        std::printf("Multi-site: n_atoms=%d n_electrode=%d\n", n,
                    cold.n_electrode());

        long sum_cold = 0, sum_w2 = 0, sum_w5 = 0;
        int n_post = 0;
        constexpr int n_steps = 16;
        // K=5 needs 6 recorded vectors, i.e. it first predicts at step 5.
        constexpr int first_scored = 6;
        for (int s = 0; s < n_steps; ++s) {
            const double dz = 0.03 * (s + 1);
            for (auto* f : {&f_cold, &f_warm2, &f_warm5}) {
                for (int k = 0; k < 4; ++k) {
                    const double z0 = (k % 2 == 0) ? 8.0 : 12.0;
                    const double sign = (k % 2 == 0) ? 1.0 : -0.5;
                    f->span(kAtom_z)[ion0 + k] = z0 + sign * dz;
                }
            }
            run_cold.step();
            run_w2.step();
            run_w5.step();
            if (s >= first_scored) {
                sum_cold += cold.last_iters();
                sum_w2 += warm2.last_iters();
                sum_w5 += warm5.last_iters();
                ++n_post;
                if (!warm2.last_used_predictor()
                    || !warm5.last_used_predictor()) {
                    std::printf("FAIL: expected ASPC warm-start at step %d\n",
                                s);
                    return 1;
                }
            }
            if (!(warm5.neutrality_residual() <= 1e-8)) {
                std::printf("FAIL: neutrality %.3e at step %d\n",
                            warm5.neutrality_residual(), s);
                return 1;
            }
        }
        if (n_post <= 0) {
            std::printf("FAIL: no post-fill steps\n");
            return 1;
        }
        const double mean_cold = static_cast<double>(sum_cold) / n_post;
        const double mean_w2 = static_cast<double>(sum_w2) / n_post;
        const double mean_w5 = static_cast<double>(sum_w5) / n_post;
        std::printf("Mean CG iters (post-fill n=%d): cold=%.3f "
                    "ASPC K=2 %.3f  ASPC K=5 (MetalWalls) %.3f\n",
                    n_post, mean_cold, mean_w2, mean_w5);
        if (!(mean_w2 < mean_cold) || !(mean_w5 < mean_cold)) {
            std::printf("FAIL: ASPC did not reduce mean CG iters "
                        "(K2=%.3f K5=%.3f cold=%.3f)\n",
                        mean_w2, mean_w5, mean_cold);
            return 1;
        }
        if (!(warm5.total_charge(1) > 0.0 && warm5.total_charge(0) < 0.0)) {
            std::printf("FAIL: charge polarity lost Q_bot=%.6e Q_top=%.6e\n",
                        warm5.total_charge(0), warm5.total_charge(1));
            return 1;
        }
    }

    std::printf("PASS: electrode-charge-predictor — ASPC coeffs, CG=dense, "
                "fewer CG iters with ASPC\n");
    return 0;
}
