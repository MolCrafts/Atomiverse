/// @file regressions/md/const-pot-bo-parallel-plate/reg_const-pot-bo-parallel-plate.cpp
/// @brief E2E: constant-potential BO electrode charges on parallel-plate madelung twin.
///
/// Offline pin: LAMMPS ELECTRODE madelung log (see REFERENCE.md).
/// Public headers + link atomiverse only. Exits non-zero on failure.

#include <cmath>
#include <cstdio>

#include "atomiverse/core/frame_schema.h"
#include "atomiverse/cpu/dynamics/const_pot_bo.h"
#include "atomiverse/cpu/dynamics/md.h"
#include "atomiverse/cpu/dynamics/velocity_verlet.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/neighbor/neighbor_list.h"
#include "atomiverse/cpu/potentials/pair_coul_cut.h"

using namespace ATV;

namespace {

void fill_madelung(cpu::Frame& frame, const double z[4], const double q0[4],
                   const int typ[4]) {
    for (int i = 0; i < 4; ++i) {
        frame.span(kAtom_x)[i] = 0.0;
        frame.span(kAtom_y)[i] = 0.0;
        frame.span(kAtom_z)[i] = z[i];
        frame.span(kAtom_charge)[i] = q0[i];
        frame.span(kAtom_type)[i] = typ[i];
        frame.span(kAtom_mass)[i] = (typ[i] < 3) ? 196.966553 : 1.0;
        frame.span(kAtom_tag)[i] = i + 1;
        frame.span(kAtom_vx)[i] = frame.span(kAtom_vy)[i] =
            frame.span(kAtom_vz)[i] = 0.0;
    }
    frame.box = cpu::Box::orthorhombic(1.0, 1.0, 20.0);
}

/// Algebraic BO solve via one MD step (Frame-only; dt=0 holds geometry).
void solve_bo(cpu::md::ConstPotBO& conp, cpu::Frame& frame) {
    cpu::md::MD sim;
    sim.set_neighbor(cpu::md::NeighborList<>{/*cutoff=*/12.0, /*skin=*/0.5});
    sim.set_evaluator(cpu::PairCoulCut{/*cutoff=*/12.0});
    sim.set_integrator(cpu::md::VelocityVerlet{/*dt [fs]=*/0.0});
    sim.set_observer_ref(conp);
    sim.build();
    sim.run(frame, 1);
}

} // namespace

int main() {
    // ── LAMMPS ELECTRODE madelung geometry (metal units) ─────────────
    // Pin: Q_top = +0.279214485147238 e at ΔΨ = 1 V, η = 2 Å⁻¹, symm on.
    constexpr double kQTopPin = 0.279214485147238; // [e], oracle lammps
    constexpr double kRelTol = 0.01;               // 1% (v1 vs Ewald electrode)
    constexpr double kEta = 2.0;                   // [1/Å]
    constexpr double kPsiBot = 0.0;                // [V]
    constexpr double kPsiTop = 1.0;                // [V]
    constexpr int kImageRange = 250;               // 2D xy Gaussian images

    const double z[4] = {8.0, 12.0, 9.0, 11.0};
    const double q0[4] = {0.0, 0.0, 0.5, -0.5};
    const int typ[4] = {1, 2, 3, 3};

    cpu::Frame frame(4);
    fill_madelung(frame, z, q0, typ);

    cpu::md::ConstPotBO conp(kEta);
    conp.add_electrode(/*type=*/1, kPsiBot);
    conp.add_electrode(/*type=*/2, kPsiTop);
    conp.electroneutrality = true;
    conp.xy_image_range = kImageRange;
    solve_bo(conp, frame);

    const double q_bot = conp.total_charge(0);
    const double q_top = conp.total_charge(1);
    const double neut = conp.neutrality_residual();
    const double rel = std::abs(q_top - kQTopPin) / std::abs(kQTopPin);

    std::printf("const-pot-bo-parallel-plate\n");
    std::printf("  Q_bot   = % .12e e\n", q_bot);
    std::printf("  Q_top   = % .12e e  (pin %+ .12e)\n", q_top, kQTopPin);
    std::printf("  |sum Q| = % .3e e\n", neut);
    std::printf("  rel err = % .4e  (tol %.2f)\n", rel, kRelTol);
    std::printf("  DeltaPsi = %.3f V\n", conp.delta_psi());

    int nfail = 0;
    if (!(neut <= 1e-10)) {
        std::fprintf(stderr, "FAIL neutrality |sum Q|=%g > 1e-10\n", neut);
        ++nfail;
    }
    if (!(rel <= kRelTol)) {
        std::fprintf(stderr, "FAIL Q_top rel err %g > %g\n", rel, kRelTol);
        ++nfail;
    }
    if (!(q_top > 0.0 && q_bot < 0.0)) {
        std::fprintf(stderr, "FAIL charge signs Q_bot=%g Q_top=%g\n", q_bot,
                     q_top);
        ++nfail;
    }
    // Frame write-back
    if (std::abs(static_cast<double>(frame.span(kAtom_charge)[1]) - q_top)
        > 1e-14) {
        std::fprintf(stderr,
                     "FAIL kAtom_charge not written for top electrode\n");
        ++nfail;
    }
    // Electrolyte charges must remain
    if (std::abs(static_cast<double>(frame.span(kAtom_charge)[2]) - 0.5) > 1e-14
        || std::abs(static_cast<double>(frame.span(kAtom_charge)[3]) + 0.5)
               > 1e-14) {
        std::fprintf(stderr, "FAIL electrolyte charges mutated\n");
        ++nfail;
    }

    // Rebuild-every-step path (one VelocityVerlet step with finite dt).
    {
        cpu::Frame f2(4);
        fill_madelung(f2, z, q0, typ);

        cpu::md::ConstPotBO conp2(kEta);
        conp2.add_electrode(1, kPsiBot);
        conp2.add_electrode(2, kPsiTop);
        conp2.electroneutrality = true;
        conp2.xy_image_range = kImageRange;
        conp2.rebuild_matrix_every_step = true;

        cpu::md::MD sim;
        sim.set_neighbor(
            cpu::md::NeighborList<>{/*cutoff=*/12.0, /*skin=*/0.5});
        sim.set_evaluator(cpu::PairCoulCut{/*cutoff=*/12.0});
        sim.set_integrator(cpu::md::VelocityVerlet{/*dt [fs]=*/0.1});
        sim.set_observer_ref(conp2); // default_bindings → pre_force @ -50
        sim.build();
        sim.run(f2, 1);

        const double q_top_md = conp2.total_charge(1);
        const double rel_md =
            std::abs(q_top_md - kQTopPin) / std::abs(kQTopPin);
        std::printf("  MD path Q_top = % .12e  rel=% .4e  |sumQ|=% .3e\n",
                    q_top_md, rel_md, conp2.neutrality_residual());
        if (!(rel_md <= kRelTol) || !(conp2.neutrality_residual() <= 1e-10)) {
            std::fprintf(stderr, "FAIL MD-bound ConstPotBO rel=%g neut=%g\n",
                         rel_md, conp2.neutrality_residual());
            ++nfail;
        }
    }

    if (nfail) {
        std::fprintf(stderr, "%d check(s) failed\n", nfail);
        return 1;
    }
    std::printf("PASS\n");
    return 0;
}
