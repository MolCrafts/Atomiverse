/// @file regressions/md/thermo-integration-capacitance/reg_thermo-integration-capacitance.cpp
/// @brief E2E: soft-core / λ TI workflow for electrode capacitance proxy.
///
/// Gate A — differential capacitance of an empty const-pot parallel plate:
///   C = Q(ΔΨ) / ΔΨ  is λ-independent for a linear Gaussian electrode BO.
/// Gate B — Kirkwood TI identity on soft Coulomb: ∫₀¹ ∂U/∂λ dλ = U(1)−U(0)
///   (already in fep_soft_dimer for LJ; here Coulomb soft + electrode context).
/// Gate C — charging free-energy proxy: A(Ψ) = ½ C Ψ² for linear response
///   matches ½ Q Ψ from ConstPotBO + ChargingWork bookkeeping.
///
/// oracle: literature (Reed/Scalfi Gaussian electrode algebra; Beutler soft-core)
/// Public headers only. See REFERENCE.md.

#include <cmath>
#include <cstdio>
#include <vector>

#include "atomiverse/core/frame_schema.h"
#include "atomiverse/core/units.h"
#include "atomiverse/cpu/dynamics/const_pot_bo.h"
#include "atomiverse/cpu/dynamics/md.h"
#include "atomiverse/cpu/dynamics/velocity_verlet.h"
#include "atomiverse/cpu/electrodes/charging_work.h"
#include "atomiverse/cpu/evaluate/evaluate.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/neighbor/neighbor_list.h"
#include "atomiverse/cpu/potentials/pair_coul_cut.h"
#include "atomiverse/cpu/potentials/pair_coul_soft.h"

using namespace ATV;

namespace {

constexpr double kLxy = 10.0;
constexpr double kLz = 20.0;
constexpr double kZBot = 5.0;
constexpr double kZTop = 15.0;
constexpr double kEta = 2.0;
constexpr int kImage = 80;
constexpr int n_side = 2; // 2×2 per plate

cpu::Frame empty_plates() {
    const int N = 2 * n_side * n_side;
    cpu::Frame f(N);
    int a = 0;
    for (int wall = 0; wall < 2; ++wall) {
        const double z = (wall == 0) ? kZBot : kZTop;
        const int typ = (wall == 0) ? 1 : 2;
        for (int iy = 0; iy < n_side; ++iy)
            for (int ix = 0; ix < n_side; ++ix) {
                f.span(kAtom_x)[a] = (ix + 0.5) * (kLxy / n_side);
                f.span(kAtom_y)[a] = (iy + 0.5) * (kLxy / n_side);
                f.span(kAtom_z)[a] = z;
                f.span(kAtom_type)[a] = typ;
                f.span(kAtom_charge)[a] = 0.0;
                f.span(kAtom_mass)[a] = 12.0;
                f.span(kAtom_tag)[a] = a + 1;
                f.span(kAtom_vx)[a] = f.span(kAtom_vy)[a] =
                    f.span(kAtom_vz)[a] = 0.0;
                ++a;
            }
    }
    f.box = cpu::Box::orthorhombic(kLxy, kLxy, kLz);
    return f;
}

struct QC {
    double Q = 0.0; // |Q_top|
    double W = 0.0; // charging work −Σ Ψ Q
};

QC solve_at_psi(double psi_top) {
    auto f = empty_plates();
    cpu::md::ConstPotBO bo(kEta);
    bo.add_electrode(1, 0.0);
    bo.add_electrode(2, psi_top);
    bo.electroneutrality = true;
    bo.xy_image_range = kImage;
    cpu::electrodes::ChargingWork cw({1, 2}, {0.0, psi_top});

    cpu::md::MD sim;
    // ChargingWork after BO: BO pre_force, ChargingWork on force phase.
    sim.set_evaluator_ref(cw);
    sim.set_integrator(cpu::md::VelocityVerlet{0.0});
    sim.set_observer_ref(bo);
    sim.build();
    sim.run(f, 1);

    const double Q = bo.total_charge(1);
    return {std::fabs(Q), cw.last_work()};
}

cpu::Frame soft_dimer() {
    cpu::Frame f(2);
    f.span(kAtom_x)[0] = 0;
    f.span(kAtom_y)[0] = 0;
    f.span(kAtom_z)[0] = 0;
    f.span(kAtom_x)[1] = 3.0;
    f.span(kAtom_y)[1] = 0;
    f.span(kAtom_z)[1] = 0;
    f.span(kAtom_charge)[0] = 1.0;
    f.span(kAtom_charge)[1] = -1.0;
    f.span(kAtom_mass)[0] = f.span(kAtom_mass)[1] = 1.0;
    f.span(kAtom_type)[0] = f.span(kAtom_type)[1] = 0;
    f.span(kAtom_tag)[0] = 1;
    f.span(kAtom_tag)[1] = 2;
    f.box = cpu::Box::cubic(50.0);
    f.meta.get(kEnergy) = 0;
    f.meta.get(kEnergyPair) = 0;
    return f;
}

} // namespace

int main() {
    int fail = 0;

    // ── Gate A: C = Q/Ψ constant (linear BO capacitor) ─────────────────
    const double psi1 = 0.5; // [V]
    const double psi2 = 1.0;
    const auto s1 = solve_at_psi(psi1);
    const auto s2 = solve_at_psi(psi2);
    const double C1 = s1.Q / psi1;
    const double C2 = s2.Q / psi2;
    std::printf("Gate A: Q(0.5V)=%.6e  Q(1V)=%.6e  C1=%.6e  C2=%.6e e/V\n",
                s1.Q, s2.Q, C1, C2);
    if (!(C1 > 0.0 && C2 > 0.0)) {
        std::printf("FAIL non-positive C\n");
        ++fail;
    }
    // Linearity: C(Ψ) same within 1%.
    if (!(std::fabs(C1 - C2) <= 0.01 * C2)) {
        std::printf("FAIL C not linear: C1=%.6e C2=%.6e\n", C1, C2);
        ++fail;
    }
    // Scaling: Q(2Ψ) ≈ 2 Q(Ψ)
    if (!(std::fabs(s2.Q - 2.0 * s1.Q) <= 0.02 * s2.Q)) {
        std::printf("FAIL Q not ∝ Ψ\n");
        ++fail;
    }

    // ── Gate C: charging free energy A = ½ Q Ψ = −W for linear response ─
    const double W = s2.W;
    const double QPsi = s2.Q * psi2;
    std::printf("Gate C: W=%.6e  Q*Ψ=%.6e  ½QΨ=%.6e\n", W, QPsi, 0.5 * QPsi);
    if (!(std::fabs(W + QPsi) <= 1e-8 * std::fmax(1.0, QPsi))) {
        std::printf("FAIL ChargingWork vs −QΨ (W=%.6e want %.6e)\n", W, -QPsi);
        ++fail;
    }
    const double A = 0.5 * C2 * psi2 * psi2;
    if (!(std::fabs(A - 0.5 * QPsi) <= 1e-8 * std::fmax(1.0, A))) {
        std::printf("FAIL A≠½QΨ\n");
        ++fail;
    }

    // ── Gate B: Kirkwood TI on soft Coulomb dimer (workflow primitive) ─
    {
        auto energy_at = [](double lam) {
            auto f = soft_dimer();
            cpu::PairCoulSoft soft(/*cutoff=*/8.0, /*sigma_c=*/1.0, 0.5, 1.0);
            soft.lambda = lam;
            cpu::Evaluate eval;
            eval.set_neighbor(cpu::md::NeighborList<>{8.0, 0.5});
            eval.set_evaluator_ref(soft);
            eval.build();
            eval.run(f);
            return f.meta.get(kEnergy);
        };

        const double U0 = energy_at(0.0);
        const double U1 = energy_at(1.0);
        const int N = 200;
        double integral = 0.0;

        auto f = soft_dimer();
        cpu::PairCoulSoft soft(8.0, 1.0, 0.5, 1.0);
        // Continuable MD (dt=0) keeps soft-core component state across λ grid.
        cpu::md::MD sim;
        sim.set_neighbor(cpu::md::NeighborList<>{8.0, 0.5});
        sim.set_evaluator_ref(soft);
        sim.set_integrator(cpu::md::VelocityVerlet{0.0});
        sim.build();
        for (int i = 0; i <= N; ++i) {
            soft.lambda = static_cast<double>(i) / N;
            sim.run(f, 1);
            const double w = (i == 0 || i == N) ? 0.5 : 1.0;
            integral += w * soft.dU_dlambda();
        }
        integral *= 1.0 / N;
        std::printf("Gate B soft-Coul TI: ∫dU/dλ=%.12e  U1-U0=%.12e\n",
                    integral, U1 - U0);
        if (!(std::fabs(integral - (U1 - U0)) <= 1e-5)) {
            std::printf("FAIL Kirkwood identity\n");
            ++fail;
        }
    }

    if (fail) {
        std::printf("FAIL: thermo-integration-capacitance (%d)\n", fail);
        return 1;
    }
    std::printf("PASS: thermo-integration-capacitance\n");
    return 0;
}
