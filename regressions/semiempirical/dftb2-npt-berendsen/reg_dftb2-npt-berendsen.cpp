/// @file regressions/semiempirical/dftb2-npt-berendsen/reg_dftb2-npt-berendsen.cpp
/// @brief SCC-DFTB2 cell stress driving the Berendsen barostat.
///
/// Domain: Berendsen pressure coupling (Berendsen et al., J. Chem. Phys. 81,
/// 3684 (1984); LAMMPS `fix press/berendsen`) on the analytic SCC-DFTB2 cell
/// virial (Elstner et al., Phys. Rev. B 58, 7260 (1998)).
///
/// ## What is actually at risk here
///
/// No electronic evaluator declares `provides_system_stress()`. `md::Virial`
/// is auto-injected because the barostat requires a stress provider, and the
/// electronic contribution is picked up only because DFTB2 accumulates onto
/// the same `kStress` key at a compatible phase. That is a real coupling and
/// nothing was gating it: a build where the configurational virial silently
/// failed to reach the barostat would still run, still be stable, and still
/// look plausible — the barostat would just be steering on the kinetic term
/// alone.
///
/// So the case runs the **same** NPT trajectory twice, once with
/// `DFTB2::with_stress` on and once off, and requires the cell to move in
/// **opposite directions**. The cell is stretched, where the kinetic virial is
/// positive (expand) and the configurational virial is negative (contract):
/// if the electronic stress never arrived, the two runs would agree.
///
/// Public API only. <= 60 s.
#include <cmath>
#include <cstdio>

#include "atomiverse/core/frame_schema.h"
#include "atomiverse/core/stage.h"
#include "atomiverse/core/units.h"
#include "atomiverse/cpu/dynamics/barostat.h"
#include "atomiverse/cpu/dynamics/berendsen.h"
#include "atomiverse/cpu/dynamics/init.h"
#include "atomiverse/cpu/dynamics/md.h"
#include "atomiverse/cpu/dynamics/velocity_verlet.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/semiempirical/dftb2.h"

using namespace ATV;

namespace {

// 4 waters in a stretched 7.4 Å cell (~0.15 g/cm³): the molecules attract, so
// the configurational virial is in tension while the kinetic virial pushes out.
constexpr int kNat = 12;
constexpr int kZnum[] = {8, 1, 1, 8, 1, 1, 8, 1, 1, 8, 1, 1};
constexpr double kPx[] = {0.00, 0.96, -0.24, 3.70, 4.66, 3.46,
                          0.00, 0.96, -0.24, 3.70, 4.66, 3.46};
constexpr double kPy[] = {0.00, 0.00, 0.93, 0.00, 0.00, 0.93,
                          3.70, 3.70, 4.63, 3.70, 3.70, 4.63};
constexpr double kPz[] = {0.00, 0.00, 0.00, 1.85, 1.85, 1.85,
                          1.85, 1.85, 1.85, 0.00, 0.00, 0.00};

constexpr double kL0 = 7.4;    // Å
constexpr double kDt = 0.2;    // fs
constexpr double kTauT = 20.0; // fs
constexpr double kTauP = 20.0; // fs
/// Berendsen barostat modulus β [eV/Å³] (LAMMPS `bulkmodulus`). Deliberately
/// soft: the per-step dilation goes as dt/(τ_P·β), and at the default β = 1
/// this cell moves 0.01% in 60 steps — a sign test on a signal that small is
/// one rounding away from meaningless. Soft β makes the response legible; it
/// is a test knob, not a physical claim.
constexpr double kBulkModulus = 0.1;
constexpr double kT = 300.0; // K
constexpr int kSteps = 60;

cpu::Frame make_cell() {
    cpu::Frame f(kNat);
    for (int i = 0; i < kNat; ++i) {
        f.span(kAtom_type)[i] = kZnum[i];
        f.span(kAtom_x)[i] = kPx[i];
        f.span(kAtom_y)[i] = kPy[i];
        f.span(kAtom_z)[i] = kPz[i];
        f.span(kAtom_tag)[i] = i + 1;
        f.span(kAtom_mass)[i] = (kZnum[i] == 8) ? 15.999 : 1.008;
        f.span(kAtom_vx)[i] = f.span(kAtom_vy)[i] = f.span(kAtom_vz)[i] = 0;
    }
    f.box = cpu::Box::cubic(kL0);
    f.meta.get(kEnergy) = 0;
    return f;
}

/// Run NPT at P_target = 1 bar; returns the final cell volume [Å³].
double run(bool electronic_stress) {
    cpu::Frame f = make_cell();

    cpu::semiempirical::DFTB2 dftb;
    dftb.boundary = cpu::semiempirical::DftbBoundary::lattice;
    dftb.scc_tol = 1e-7;
    dftb.max_scc = 200;
    dftb.with_forces = true;
    dftb.with_stress = electronic_stress;
    dftb.predictor(3);

    cpu::md::MD sim;
    sim.set_integrator(cpu::md::VelocityVerlet{kDt});
    sim.set_observer(cpu::md::InitVelocity{kT, /*seed=*/5u});
    sim.set_thermostat(cpu::md::Berendsen{kT, kTauT, kDt});
    // 1 bar through the named conversion rather than a hand-copied factor.
    sim.set_barostat(cpu::md::Barostat{1.0 * units::bar_to_ev_per_a3, kTauP,
                                       kDt, kBulkModulus});
    sim.on(md::Phase::force).add_ref(dftb);
    sim.build();
    sim.run(f, kSteps);

    return f.box.volume();
}

} // namespace

int main() {
    const double v0 = kL0 * kL0 * kL0;
    const double v_on = run(true);
    const double v_off = run(false);

    std::printf("dftb2-npt-berendsen: 4 H2O, stretched %.1f A cell,"
                " %d steps, P_target 1 bar\n",
                kL0, kSteps);
    std::printf("  V0                     = %10.2f A^3\n", v0);
    std::printf("  V with DFTB2 stress    = %10.2f A^3  (dV = %+8.2f)\n", v_on,
                v_on - v0);
    std::printf("  V kinetic virial only  = %10.2f A^3  (dV = %+8.2f)\n", v_off,
                v_off - v0);

    if (!std::isfinite(v_on) || !std::isfinite(v_off) || v_on <= 0
        || v_off <= 0) {
        std::printf("FAIL cell went non-finite or non-positive\n");
        return 1;
    }

    // The split has to be big enough to mean something. 0.2% of the cell over
    // 60 steps is ~20x the drift either run shows against its own noise.
    constexpr double kMinSplitFrac = 0.002;
    const double split = (v_off - v_on) / v0;
    std::printf("  split (V_off - V_on)/V0 = %.4f   (need > %.4f)\n", split,
                kMinSplitFrac);
    if (!(split > kMinSplitFrac)) {
        std::printf("FAIL the two runs barely differ; the electronic stress"
                    " is not moving the barostat\n");
        return 1;
    }

    // Kinetic-only pressure is positive definite: that run can only expand.
    if (!(v_off > v0)) {
        std::printf("FAIL kinetic-only run did not expand — the barostat is"
                    " not responding to pressure at all\n");
        return 1;
    }

    // With the configurational virial the stretched cell is in tension and
    // must contract. Same trajectory, opposite sign: this is the gate that the
    // electronic stress reached the barostat.
    if (!(v_on < v0)) {
        std::printf("FAIL cell did not contract under the DFTB2 virial;"
                    " the electronic stress is not reaching the barostat\n");
        return 1;
    }

    std::printf("PASS\n");
    return 0;
}
