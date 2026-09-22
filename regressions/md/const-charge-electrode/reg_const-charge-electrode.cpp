/// @file regressions/md/const-charge-electrode/reg_const-charge-electrode.cpp
/// @brief E2E: constant-charge (conq) electrode control on the LAMMPS
///        ELECTRODE madelung parallel-plate twin — Q fixed, V response.
///
/// Offline pin: LAMMPS ELECTRODE madelung conp log + `fix electrode/conq`
/// semantics (see REFERENCE.md). Public headers + link atomiverse only.
/// Exits non-zero on failure.

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

/// LAMMPS ELECTRODE madelung geometry (metal units), z shifted into (0, Lz).
cpu::Frame madelung_frame(bool with_electrolyte) {
    cpu::Frame f(with_electrolyte ? 4 : 2);
    const double z[4] = {8.0, 12.0, 9.0, 11.0};
    const double q0[4] = {0.0, 0.0, 0.5, -0.5};
    const int typ[4] = {1, 2, 3, 3};
    const int n = with_electrolyte ? 4 : 2;
    for (int i = 0; i < n; ++i) {
        f.span(kAtom_x)[i] = 0.0;
        f.span(kAtom_y)[i] = 0.0;
        f.span(kAtom_z)[i] = z[i];
        f.span(kAtom_charge)[i] = q0[i];
        f.span(kAtom_type)[i] = typ[i];
        f.span(kAtom_mass)[i] = (typ[i] < 3) ? 196.966553 : 1.0;
        f.span(kAtom_tag)[i] = i + 1;
        f.span(kAtom_vx)[i] = f.span(kAtom_vy)[i] = f.span(kAtom_vz)[i] = 0.0;
    }
    f.box = cpu::Box::orthorhombic(1.0, 1.0, 20.0);
    return f;
}

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
    // ── Pins (REFERENCE.md) ──────────────────────────────────────────
    // LAMMPS ELECTRODE madelung conp log: ΔΨ = 1 V ⇔ Q_top = +0.279214… e.
    constexpr double kQTopPin = 0.279214485147238; // [e], oracle lammps
    constexpr double kRelTolQ = 0.01; // 1% (v1 2D-image A vs ewald/electrode)
    constexpr double kEta = 2.0;      // [1/Å]
    constexpr int kImageRange = 250;  // 2D xy Gaussian images
    // Empty Gaussian-lattice capacitor V response (literature Madelung sum,
    // see REFERENCE.md): ΔΨ = 2 k_e [2π σ d + η√(2/π) + M_□ + g(η)] with
    // square-lattice Madelung constant M_□ = −3.90026492 and Gaussian
    // correction g(η=2) = −0.195386  ⇒  651.8103 V for Q = ±1 e,
    // A_cell = 1 Å², d = 4 Å. (The bare-sheet 4π k_e σ d = 723.88 V misses
    // the on-site discrete-lattice term.)
    constexpr double kPlateDV = 651.8103;  // [V]
    constexpr double kRelTolPlate = 0.015; // R=250 real-space tail ≈ 0.8%

    int nfail = 0;

    // ── Stage 1: LAMMPS-anchored conp → conq duality (Q fixed, V back) ──
    auto fp = madelung_frame(true);
    cpu::md::ConstPotBO conp(kEta);
    conp.add_electrode(/*type=*/1, /*Psi [V]=*/0.0);
    conp.add_electrode(/*type=*/2, /*Psi [V]=*/1.0);
    conp.electroneutrality = true;
    conp.xy_image_range = kImageRange;
    solve_bo(conp, fp);
    const double q_star = conp.total_charge(1);
    const double rel_q = std::abs(q_star - kQTopPin) / kQTopPin;

    auto fq = madelung_frame(true);
    cpu::md::ConstPotBO conq(kEta);
    conq.add_electrode_charge(/*type=*/1, -q_star);
    conq.add_electrode_charge(/*type=*/2, q_star);
    conq.xy_image_range = kImageRange;
    solve_bo(conq, fq);

    const double dv = conq.delta_psi();
    std::printf("const-charge-electrode\n");
    std::printf("  conp Q_top* = % .12e e (LAMMPS pin %+.12e, rel %.2e)\n",
                q_star, kQTopPin, rel_q);
    std::printf("  conq totals = % .12e / % .12e e (targets -+Q*)\n",
                conq.total_charge(0), conq.total_charge(1));
    std::printf("  conq dV     = % .10f V (conp applied 1.0)\n", dv);
    std::printf("  conq psi    = % .6f / % .6f V\n",
                conq.electrode_potential(0), conq.electrode_potential(1));

    if (!(rel_q <= kRelTolQ)) {
        std::fprintf(stderr, "FAIL conp Q* vs LAMMPS pin rel=%g > %g\n", rel_q,
                     kRelTolQ);
        ++nfail;
    }
    if (!(std::abs(conq.total_charge(0) + q_star) <= 1e-12
          && std::abs(conq.total_charge(1) - q_star) <= 1e-12)) {
        std::fprintf(stderr,
                     "FAIL conq group totals not exactly the targets\n");
        ++nfail;
    }
    if (!(std::abs(dv - 1.0) <= 1e-8)) {
        std::fprintf(stderr, "FAIL conq dV=%g != 1 V (duality, tol 1e-8)\n",
                     dv);
        ++nfail;
    }
    // Per-site duality: conq distribution == conp distribution.
    for (int i = 0; i < 2; ++i) {
        const double qc = static_cast<double>(fq.span(kAtom_charge)[i]);
        const double qp = static_cast<double>(fp.span(kAtom_charge)[i]);
        if (std::abs(qc - qp) > 1e-10) {
            std::fprintf(stderr, "FAIL per-site duality atom %d: %g vs %g\n", i,
                         qc, qp);
            ++nfail;
        }
    }
    // Electrolyte untouched.
    if (std::abs(static_cast<double>(fq.span(kAtom_charge)[2]) - 0.5) > 1e-14
        || std::abs(static_cast<double>(fq.span(kAtom_charge)[3]) + 0.5)
               > 1e-14) {
        std::fprintf(stderr, "FAIL electrolyte charges mutated\n");
        ++nfail;
    }

    // ── Stage 2: V response magnitude with the LAMMPS pin as target Q ──
    auto fpin = madelung_frame(true);
    cpu::md::ConstPotBO conq_pin(kEta);
    conq_pin.add_electrode_charge(1, -kQTopPin);
    conq_pin.add_electrode_charge(2, kQTopPin);
    conq_pin.xy_image_range = kImageRange;
    solve_bo(conq_pin, fpin);
    const double dv_pin = conq_pin.delta_psi();
    std::printf("  conq(LAMMPS Q_pin) dV = % .6f V (budget 1.0 +- 0.25)\n",
                dv_pin);
    if (!(std::abs(dv_pin - 1.0) <= 0.25)) {
        std::fprintf(stderr, "FAIL conq(Q_pin) dV=%g outside 1 +- 0.25 V\n",
                     dv_pin);
        ++nfail;
    }

    // ── Stage 3: empty capacitor — literature Madelung-sum V response ──
    auto fv = madelung_frame(false);
    cpu::md::ConstPotBO empty(kEta);
    empty.add_electrode_charge(1, -1.0);
    empty.add_electrode_charge(2, 1.0);
    empty.xy_image_range = kImageRange;
    solve_bo(empty, fv);
    const double dv_empty = empty.delta_psi();
    const double rel_plate = std::abs(dv_empty - kPlateDV) / kPlateDV;
    std::printf(
        "  empty capacitor dV = % .4f V (Madelung pin %.4f, rel %.2e)\n",
        dv_empty, kPlateDV, rel_plate);
    if (!(rel_plate <= kRelTolPlate)) {
        std::fprintf(stderr, "FAIL empty-capacitor dV rel err %g > %g\n",
                     rel_plate, kRelTolPlate);
        ++nfail;
    }

    // ── Stage 4: Component-on-MD path (one VelocityVerlet step) ─────────
    {
        auto fmd = madelung_frame(true);
        cpu::md::ConstPotBO conq_md(kEta);
        conq_md.add_electrode_charge(1, -q_star);
        conq_md.add_electrode_charge(2, q_star);
        conq_md.xy_image_range = kImageRange;
        conq_md.rebuild_matrix_every_step = true;

        cpu::md::MD sim;
        sim.set_neighbor(
            cpu::md::NeighborList<>{/*cutoff=*/12.0, /*skin=*/0.5});
        sim.set_evaluator(cpu::PairCoulCut{/*cutoff=*/12.0});
        sim.set_integrator(cpu::md::VelocityVerlet{/*dt [fs]=*/0.1});
        sim.set_observer_ref(conq_md); // default_bindings → pre_force @ -50
        sim.build();
        sim.run(fmd, 1);

        const double dv_md = conq_md.delta_psi();
        std::printf("  MD path: Q_top=% .12e dV=% .10f V\n",
                    conq_md.total_charge(1), dv_md);
        // Positions may have drifted by one Verlet half-kick before the
        // pre_force solve; the constraint stays exact, dV moves O(1e-3) V.
        if (std::abs(conq_md.total_charge(1) - q_star) > 1e-10
            || std::abs(dv_md - 1.0) > 0.05) {
            std::fprintf(stderr, "FAIL MD-bound conq Q=%g dV=%g\n",
                         conq_md.total_charge(1), dv_md);
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
