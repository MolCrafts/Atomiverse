/// @file regressions/md/pim-npt-polarizable-il/reg_pim-npt-polarizable-il.cpp
/// @brief END-TO-END: bulk polarizable IL under NPT (MetalWalls PIM product
///        twin) — PIM virial pin + NoseHooverNPT product invariants.
///
/// oracle: literature (Applequist/Thole induced-dipole form), offline numpy
/// pin; the charge–dipole kernel and the E_pol convention are cross-validated
/// against the real MetalWalls binary (git c8ad175d) in the isolated-pair
/// limit. MetalWalls cannot express the pinned lattice numbers themselves —
/// it has no Thole damping and no truncated-Coulomb mode. See REFERENCE.md
/// and tools/metalwalls_oracle/pin_notes_pim-npt-polarizable-il.md.
/// Public headers + link atomiverse only. Flat main(); exits non-zero on fail.

#include <cmath>
#include <cstdio>
#include <string>

#include "atomiverse/core/frame_schema.h"
#include "atomiverse/core/stage.h"
#include "atomiverse/core/units.h"
#include "atomiverse/cpu/component/criterion.h"
#include "atomiverse/cpu/dynamics/init.h"
#include "atomiverse/cpu/dynamics/md.h"
#include "atomiverse/cpu/dynamics/nose_hoover_npt.h"
#include "atomiverse/cpu/evaluate/evaluate.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/neighbor/neighbor_list.h"
#include "atomiverse/cpu/observe/virial_stress.h"
#include "atomiverse/cpu/potentials/pair_coul_cut.h"
#include "atomiverse/cpu/potentials/pair_lj_cut.h"
#include "atomiverse/cpu/potentials/pim_induced_dipole.h"

using namespace ATV;

namespace {

// Deterministic 4x4x4 rock-salt toy polarizable IL (pin legs must match the
// offline pin script tools/metalwalls_oracle/inputs/pim_npt_polarizable_il_pin.py
// at d = 2.35 A): (-1)^(ix+iy+iz) charges, cation type 0 (alpha 0), anion
// type 1 (alpha 3.0 A^3), disp[a] = 0.15*sin(1.7*(i+1) + 2.3*a).
constexpr int kNc = 4;
constexpr double kDPin = 2.35;  // pin-leg spacing (compressed; big PIM signal)
constexpr double kDNpt = 2.9;   // NPT-leg spacing (stretched; P0 < 0, tension)
constexpr double kMass = 100.0; // amu (heavy CG ions)

cpu::Frame make_start_frame(double d) {
    const int n = kNc * kNc * kNc;
    cpu::Frame f(n);
    int i = 0;
    for (int ix = 0; ix < kNc; ++ix)
        for (int iy = 0; iy < kNc; ++iy)
            for (int iz = 0; iz < kNc; ++iz) {
                const double dx = 0.15 * std::sin(1.7 * (i + 1) + 2.3 * 0.0);
                const double dy = 0.15 * std::sin(1.7 * (i + 1) + 2.3 * 1.0);
                const double dz = 0.15 * std::sin(1.7 * (i + 1) + 2.3 * 2.0);
                f.span(kAtom_x)[i] = ix * d + dx;
                f.span(kAtom_y)[i] = iy * d + dy;
                f.span(kAtom_z)[i] = iz * d + dz;
                const bool cation = ((ix + iy + iz) % 2 == 0);
                f.span(kAtom_charge)[i] = cation ? 1.0 : -1.0;
                f.span(kAtom_type)[i] = cation ? 0 : 1;
                f.span(kAtom_mass)[i] = kMass;
                f.span(kAtom_tag)[i] = i + 1;
                f.span(kAtom_fx)[i] = f.span(kAtom_fy)[i] =
                    f.span(kAtom_fz)[i] = 0;
                ++i;
            }
    f.box = cpu::Box::cubic(kNc * d);
    f.meta.get(kEnergy) = 0;
    for (int c = 0; c < 6; ++c)
        f.meta.get(kStress)[c] = 0;
    return f;
}

/// Volume-range tracker written OUTSIDE the library (the documented
/// user-extensible Criterion surface); verdict is always Continue.
class TrackVolume : public cpu::Criterion {
  public:
    std::string name() const override { return "track_volume"; }
    void launch(const cpu::LaunchContext&) override {}
    ATV::Verdict verdict(const cpu::LoopState& s) override {
        const double v = s.frame.box.volume();
        if (v < v_min)
            v_min = v;
        if (v > v_max)
            v_max = v;
        return ATV::Verdict::Continue;
    }
    double v_min = 1e300;
    double v_max = 0.0;
};

} // namespace

int main() {
    // ── Offline numpy pins of the TRUNCATED (rc = 4.0 A, minimum-image)
    //    Applequist/Thole induced-dipole model this component implements.
    //    NOT MetalWalls values: real MW (Ewald, no Thole) gives
    //    E_pol = -1.682169167852 eV on this same geometry — the 5.17x gap is
    //    the real-space truncation, see REFERENCE.md. Units: eV, eV, eV/A^3.
    const double E_pol_pin = -8.695749870555906;
    const double W_pin[6] = {-10.66184373345355, -12.125914211727187,
                             -9.657852793291474, -0.2325271422922529,
                             -2.2469237837238,   1.375774440282161};
    const double P_pol_pin = -0.013021203851134545;
    const double tol_E = 1e-8;
    const double tol_W = 1e-7;
    const double tol_P = 1e-10;

    const double pim_cut = 4.0;
    const double alpha_anion = 3.0;
    const double thole_a = 0.39;

    bool ok = true;

    // ════ Leg 1: PIM-only pin — E_pol + virial vs the offline oracle ════
    {
        auto f = make_start_frame(kDPin);
        cpu::PimInducedDipole pim(pim_cut, {0.0, alpha_anion}, thole_a,
                                  /*pol_tol=*/1e-13, /*max_iter=*/1000);
        cpu::Evaluate eval;
        eval.on(evaluate::Phase::evaluate).add_ref(pim);
        eval.build();
        eval.run(f);

        const double E = f.meta.get(kEnergy);
        const double trace = f.meta.get(kStress)[0] + f.meta.get(kStress)[1]
                             + f.meta.get(kStress)[2];
        const double P_pol = trace / (3.0 * f.box.volume());

        std::printf("PIM pin: E_pol=%.16e want=%.16e (iters=%d)\n", E,
                    E_pol_pin, pim.last_pol_iters());
        std::printf("  P_pol=%.16e want=%.16e [eV/A^3]\n", P_pol, P_pol_pin);
        if (!(std::fabs(E - E_pol_pin) <= tol_E)) {
            std::printf("FAIL E_pol vs pin (|d|=%.3e tol=%.1e)\n",
                        std::fabs(E - E_pol_pin), tol_E);
            ok = false;
        }
        if (!(std::fabs(pim.last_pol_energy() - E_pol_pin) <= tol_E)) {
            std::printf("FAIL last_pol_energy vs pin\n");
            ok = false;
        }
        for (int c = 0; c < 6; ++c) {
            const double w = f.meta.get(kStress)[c];
            std::printf("  W[%d]=%.16e want=%.16e\n", c, w, W_pin[c]);
            if (!(std::fabs(w - W_pin[c]) <= tol_W)) {
                std::printf("FAIL W[%d] vs pin (|d|=%.3e tol=%.1e)\n", c,
                            std::fabs(w - W_pin[c]), tol_W);
                ok = false;
            }
        }
        if (!(std::fabs(P_pol - P_pol_pin) <= tol_P)) {
            std::printf("FAIL P_pol vs pin (|d|=%.3e tol=%.1e)\n",
                        std::fabs(P_pol - P_pol_pin), tol_P);
            ok = false;
        }
        if (!pim.last_pol_converged()) {
            std::printf("FAIL SCC not converged in pin leg\n");
            ok = false;
        }
    }

    // ════ Leg 2: stress additivity — PIM shares the barostat's kStress ════
    // trace(LJ+coul+PIM) − trace(LJ+coul) must equal the pinned PIM trace,
    // i.e. the polarization pressure reaches the meta every barostat reads.
    double trace_lc = 0.0;
    {
        auto f = make_start_frame(kDPin);
        cpu::Evaluate eval;
        eval.set_evaluator(cpu::PairLJCut{0.01, 3.0, pim_cut});
        eval.set_evaluator(cpu::PairCoulCut{pim_cut});
        eval.set_neighbor(cpu::md::NeighborList<>{pim_cut, 0.3});
        eval.build();
        eval.run(f);
        trace_lc = f.meta.get(kStress)[0] + f.meta.get(kStress)[1]
                   + f.meta.get(kStress)[2];
    }
    double trace_all = 0.0;
    double P0_config = 0.0;
    {
        auto f = make_start_frame(kDPin);
        cpu::Evaluate eval;
        eval.set_evaluator(cpu::PairLJCut{0.01, 3.0, pim_cut});
        eval.set_evaluator(cpu::PairCoulCut{pim_cut});
        eval.set_evaluator(cpu::PimInducedDipole{
            pim_cut, {0.0, alpha_anion}, thole_a, 1e-13, 1000});
        eval.set_neighbor(cpu::md::NeighborList<>{pim_cut, 0.3});
        eval.build();
        eval.run(f);
        trace_all = f.meta.get(kStress)[0] + f.meta.get(kStress)[1]
                    + f.meta.get(kStress)[2];
        P0_config = trace_all / (3.0 * f.box.volume());
    }
    {
        const double w_trace_pin = W_pin[0] + W_pin[1] + W_pin[2];
        const double d = std::fabs(trace_all - trace_lc - w_trace_pin);
        std::printf("stress additivity: tr(all)-tr(lj+coul)=%.12e "
                    "want=%.12e (P0_config=%.6e eV/A^3)\n",
                    trace_all - trace_lc, w_trace_pin, P0_config);
        if (!(d <= 1e-6)) {
            std::printf("FAIL stress additivity (|d|=%.3e)\n", d);
            ok = false;
        }
    }

    // ════ Leg 3: short NPT product run — MTK barostat sees PIM pressure ════
    {
        auto f = make_start_frame(kDNpt);
        const double V0 = f.box.volume();
        const double T = 800.0;
        const int n_steps = 250;

        cpu::PimInducedDipole pim(pim_cut, {0.0, alpha_anion}, thole_a,
                                  /*pol_tol=*/1e-10, /*max_iter=*/500);
        TrackVolume vol;
        cpu::md::MD sim;
        sim.set_evaluator(cpu::PairLJCut{0.01, 3.0, pim_cut});
        sim.set_evaluator(cpu::PairCoulCut{pim_cut});
        sim.set_evaluator_ref(pim);
        sim.set_neighbor(cpu::md::NeighborList<>{pim_cut, 0.3});
        sim.set_integrator(cpu::md::NoseHooverNPT{
            /*T_target=*/T, /*tau_T=*/100.0, /*P_target=*/0.0,
            /*tau_P=*/500.0, /*dt=*/1.0});
        sim.set_observer(cpu::md::InitVelocity{T, 20260727u});
        sim.set_observer(cpu::md::Virial{});
        sim.add_criterion_ref(vol);
        sim.enable_timing(false);
        sim.build();
        sim.run(f, n_steps);

        const double V1 = f.box.volume();
        const double H = f.meta.get(kConserved);
        std::printf("NPT: V0=%.4f -> V1=%.4f A^3 (%.1f%%), Vmin=%.4f "
                    "Vmax=%.4f, conserved H=%.6f eV, SCC iters(last)=%d\n",
                    V0, V1, 100.0 * (V1 / V0 - 1.0), vol.v_min, vol.v_max, H,
                    pim.last_pol_iters());

        // The component throws on any unconverged SCC step, so a completed
        // run already proves per-step convergence; assert the flag anyway.
        if (!pim.last_pol_converged()) {
            std::printf("FAIL final-step SCC not converged\n");
            ok = false;
        }
        if (!(pim.last_pol_energy() < 0.0)
            || !std::isfinite(pim.last_pol_energy())) {
            std::printf("FAIL final E_pol not negative/finite (%.6e)\n",
                        pim.last_pol_energy());
            ok = false;
        }
        if (!std::isfinite(H)) {
            std::printf("FAIL non-finite conserved quantity\n");
            ok = false;
        }
        // Stretched start (P0 < 0 tension at d = 2.9 A): the MTK cell must
        // contract toward equilibrium — self-limiting on the r^-12 wall.
        if (!(vol.v_min < 0.97 * V0)) {
            std::printf("FAIL stretched cell did not contract "
                        "(Vmin/V0=%.3f)\n",
                        vol.v_min / V0);
            ok = false;
        }
        if (!(vol.v_min > 0.5 * V0)) {
            std::printf("FAIL unbounded contraction (Vmin/V0=%.3f)\n",
                        vol.v_min / V0);
            ok = false;
        }
        if (!(vol.v_max < 1.2 * V0)) {
            std::printf("FAIL cell rang above start volume (Vmax/V0=%.3f)\n",
                        vol.v_max / V0);
            ok = false;
        }
        if (!std::isfinite(V1) || !(V1 > 0.0)) {
            std::printf("FAIL non-finite final volume\n");
            ok = false;
        }
    }

    std::printf("%s: pim-npt-polarizable-il (PIM product twin under NPT)\n",
                ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
