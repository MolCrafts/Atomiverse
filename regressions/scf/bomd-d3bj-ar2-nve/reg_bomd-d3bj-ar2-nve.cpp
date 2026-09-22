/// @file regressions/scf/bomd-d3bj-ar2-nve/reg_bomd-d3bj-ar2-nve.cpp
/// @brief END-TO-END: Born–Oppenheimer NVE on a **dispersion-bound** dimer —
///        the witness that a d3bj mean field reports a force that is the
///        gradient of the energy it reports.
///
/// ## Why this case exists
///
/// It exists because it used to fail. `MeanFieldMethod` folds the geometry-only
/// DFT-D3(BJ) energy into its total (`e_total_ += e_disp_`), and until
/// 2026-07-28 `internal::ScfGradient` carried **no** dispersion term: with
/// `d3bj = true` the reported force was the gradient of a *different* function
/// than the reported energy. That is invisible in a single-point energy and
/// invisible in an optimisation that happens to land near the right minimum —
/// it shows up as an NVE trajectory that does not conserve energy, and as a
/// stationary point in the wrong place. Ar₂ is the sharpest possible probe: at
/// this level the dispersion force is **larger than the total force**, so
/// dropping it does not perturb the dynamics, it *unbinds the dimer*.
///
/// ## The surface
///
/// E(R) = E_RHF/def2-SVP(R) + E_D3(BJ)(R), the Grimme two-body C6/C8 correction
/// with Becke–Johnson damping at the PBE parameters (s6 = 1, s8 = 0.7875,
/// a1 = 0.4289, a2 = 4.4407 Bohr — the `scf::D3BJ` defaults the `d3bj` flag
/// uses). HF + PBE-D3(BJ) parameters is **not** a published functional pairing
/// (see REFERENCE.md §"What is NOT claimed"): this case pins conservativeness
/// on the surface the code evaluates, plus the dispersion term itself against
/// an external oracle. It is not a benchmark of Ar₂ binding energetics.
///
/// ## What is pinned
///
///   1. E_disp and ∂E_disp/∂R at the start geometry vs offline simple-dftd3
///      (pyscf-dispersion 1.5.0) — hard-coded, no oracle at build or run time.
///   2. Analytic total force vs a central difference of the total energy
///      through the public driver, all six degrees of freedom: this is the
///      wiring gate, and it is the one the old tree failed by 1.1e-2 eV/Å.
///   3. NVE conservation over ~one vibrational period of the dispersion well,
///      the dimer staying bound, and the *materiality* of the dispersion term
///      along that trajectory (E_disp swings by ~600× the drift envelope —
///      without the leg the drift would be the swing).
///
/// Grimme, Antony, Ehrlich, Krieg, *J. Chem. Phys.* **132**, 154104 (2010);
/// Grimme, Ehrlich, Goerigk, *J. Comput. Chem.* **32**, 1456 (2011).
/// See REFERENCE.md. Public API only. Requires ATV_BUILD_HF + ATV_BUILD_MD.
/// Exit non-zero on failure.

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <vector>

#include "atomiverse/core/frame_schema.h"
#include "atomiverse/core/stage.h"
#include "atomiverse/core/units.h"
#include "atomiverse/cpu/dynamics/md.h"
#include "atomiverse/cpu/dynamics/velocity_verlet.h"
#include "atomiverse/cpu/evaluate/evaluate.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/scf/basis_set.h"
#include "atomiverse/cpu/scf/dftd3.h"
#include "atomiverse/cpu/scf/methods.h"

using namespace ATV;

namespace {

/// Cubic cell edge [Å]. The SCF is an isolated-molecule solver; the box only
/// has to be wide enough that the dimer never touches a face.
constexpr double kBoxL = 24.0;
/// Ar mass [amu] (IUPAC 2021 standard atomic weight).
constexpr double kArMass = 39.948;
/// eV/Å per Hartree/Bohr.
constexpr double kHaBohrInEvAng = 1.0 / units::kEvPerAngstromInHartreePerBohr;

/// @brief Ar₂ at separation `r`, centred on the cell, at rest.
/// @param r Ar–Ar separation [Å].
/// @return Two-atom Frame (Z = 18, masses, tags, zero velocity).
cpu::Frame ar2(double r) {
    cpu::Frame f(2);
    for (int i = 0; i < 2; ++i) {
        f.span(kAtom_type)[i] = 18;
        f.span(kAtom_mass)[i] = kArMass;
        f.span(kAtom_tag)[i] = i + 1;
        f.span(kAtom_x)[i] = 0.5 * kBoxL;
        f.span(kAtom_y)[i] = 0.5 * kBoxL;
        f.span(kAtom_vx)[i] = 0.0;
        f.span(kAtom_vy)[i] = 0.0;
        f.span(kAtom_vz)[i] = 0.0;
    }
    f.span(kAtom_z)[0] = 0.5 * kBoxL - 0.5 * r;
    f.span(kAtom_z)[1] = 0.5 * kBoxL + 0.5 * r;
    f.box = cpu::Box::cubic(kBoxL);
    return f;
}

/// @brief The method under test: RHF/def2-SVP with the D3(BJ) correction on.
/// @return A configured `RHF` (parameters are members — no config bag).
cpu::scf::RHF method() {
    cpu::scf::RHF rhf;
    rhf.basis = cpu::scf::BasisSet::Def2SVP;
    rhf.d3bj = true;
    rhf.tol = 1e-10; // Ha — FD of the total energy needs a tight SCF
    rhf.max_iter = 200;
    return rhf;
}

/// @brief Ar–Ar separation along z [Å] (the dimer never crosses a face).
/// @param f Frame.
/// @return |z₁ − z₀| [Å].
double sep(const cpu::Frame& f) {
    return std::abs(
        static_cast<double>(f.span(kAtom_z)[1] - f.span(kAtom_z)[0]));
}

/// @brief Total energy [eV] and forces [eV/Å] through the public Evaluate
///        driver, thread-pinned (2 atoms: fork/join on a fat node dwarfs the
///        physics, and the FD below needs bit-comparable evaluations).
/// @param f Geometry; forces/energy are written onto it.
/// @param forces Out: 2×3 forces [eV/Å].
/// @return Total energy [eV] (includes E_disp).
double evaluate_total(cpu::Frame& f, double forces[2][3]) {
    cpu::scf::RHF rhf = method();
    cpu::Evaluate eval;
    eval.on(evaluate::Phase::evaluate).add_ref(rhf, 0);
    eval.set_threads(1);
    eval.build();
    eval.run(f);
    for (int a = 0; a < 2; ++a) {
        forces[a][0] = static_cast<double>(f.span(kAtom_fx)[a]);
        forces[a][1] = static_cast<double>(f.span(kAtom_fy)[a]);
        forces[a][2] = static_cast<double>(f.span(kAtom_fz)[a]);
    }
    return f.meta.get(kEnergy);
}

/// @brief Move one Cartesian component of one atom to `v` [Å].
/// @param f Frame to mutate.
/// @param a Atom index.
/// @param d Direction 0/1/2.
/// @param v New coordinate [Å].
void set_coord(cpu::Frame& f, int a, int d, double v) {
    if (d == 0)
        f.span(kAtom_x)[a] = v;
    else if (d == 1)
        f.span(kAtom_y)[a] = v;
    else
        f.span(kAtom_z)[a] = v;
}

/// @brief Read one Cartesian component [Å].
/// @param f Frame.
/// @param a Atom index.
/// @param d Direction 0/1/2.
/// @return The coordinate [Å].
double get_coord(const cpu::Frame& f, int a, int d) {
    return static_cast<double>(d == 0   ? f.span(kAtom_x)[a]
                               : d == 1 ? f.span(kAtom_y)[a]
                                        : f.span(kAtom_z)[a]);
}

} // namespace

int main() {
    // ── Frozen values ────────────────────────────────────────────────────
    /// Start separation of the trajectory [Å] — inside the repulsive wall, so
    /// the dimer swings out to the outer turning point of the dispersion well.
    constexpr double r_md = 3.40;
    /// Geometry of the force gates [Å] — near the D3(BJ)-bound minimum, where
    /// the dispersion force and the HF force are the same size and opposite.
    constexpr double r_fd = 3.60;

    // simple-dftd3 (pyscf-dispersion 1.5.0, xc=pbe, version=d3bj), Ar₂ at
    // 3.40 Å. See REFERENCE.md for the exact command and JSON.
    constexpr double e_disp_ref_ha = -0.0005527440863375172;
    /// Measured agreement is 4e-19 Ha (double precision on a 5.5e-4 number);
    /// 1e-12 leaves seven orders for libm/ULP differences across platforms
    /// while still being a real gate on the tables and the damping form.
    constexpr double e_disp_tol_ha = 1e-12;
    /// |∂E_disp/∂R_z| on each atom at 3.40 Å [Ha/Bohr], same oracle run.
    /// This is the **external** gate on the leg this case was written for:
    /// ATV's analytic D3(BJ) gradient vs simple-dftd3's own (measured 5e-20).
    constexpr double f_disp_ref_ha_bohr = 2.0714714378968497e-4;
    constexpr double f_disp_tol_ha_bohr = 1e-12;

    /// Central-difference half-step for the total-energy gradient [Å].
    constexpr double fd_h = 0.005;
    /// Worst |F_analytic − F_fd| allowed [eV/Å]. Measured 4.3e-6, which is
    /// pure O(h²) truncation of the central difference (it lands on 1.8e-5 at
    /// h = 0.01, exactly 4×) — NOT a gradient error. The missing dispersion
    /// leg was worth 1.09e-2 eV/Å here, 250× this window.
    constexpr double fd_tol_ev_ang = 5e-5;

    constexpr double dt_fs = 8.0; // ~1/77 of the ~615 fs well period
    constexpr int n_steps = 80;   // 640 fs ≈ one full oscillation
    /// NVE envelope on |E_tot − E_tot(step 1)| [eV]; measured 1.56e-5, and it
    /// scales as dt² (3.9e-6 at 4 fs), i.e. it is Velocity-Verlet
    /// discretization, not force/energy inconsistency.
    constexpr double drift_tol_ev = 1e-4;
    /// The dimer must stay bound. Without the dispersion force the HF surface
    /// pushes it from 3.40 Å straight past 5 Å and out of the well for good.
    constexpr double r_lo = 3.30, r_hi = 4.60;
    /// The dispersion term must be MATERIAL along the trajectory, or the
    /// conservation gate above would be satisfied by a leg of zero.
    constexpr double min_disp_span_ev = 5e-3;
    /// …and the drift must be small *compared with it*: this ratio is the
    /// witness. Measured 620.
    constexpr double min_span_over_drift = 100.0;

    // ── Leg 0: the dispersion term itself, vs the external oracle ────────
    {
        const cpu::Frame f = ar2(r_md);
        const cpu::scf::D3BJ d3;
        const double e_disp = d3.energy(f);
        const auto fd = d3.forces(f);
        const double f0 = fd[0][2];
        const double f1 = fd[1][2];
        std::printf("D3(BJ) at r=%.2f A: E_disp=%.15g Ha (ref %.15g, "
                    "d=%.2e)\n",
                    r_md, e_disp, e_disp_ref_ha, e_disp - e_disp_ref_ha);
        std::printf("                    F_z = %+.12e / %+.12e Ha/Bohr "
                    "(|ref|=%.12e, d=%.2e/%.2e)\n",
                    f0, f1, f_disp_ref_ha_bohr, f0 - f_disp_ref_ha_bohr,
                    f1 + f_disp_ref_ha_bohr);
        if (!std::isfinite(e_disp)
            || std::abs(e_disp - e_disp_ref_ha) > e_disp_tol_ha) {
            std::printf("FAIL: E_disp %.15g Ha vs simple-dftd3 %.15g Ha "
                        "(tol %.1e)\n",
                        e_disp, e_disp_ref_ha, e_disp_tol_ha);
            return 1;
        }
        // Attractive: atom 0 (lower z) is pulled toward +z, atom 1 toward −z.
        if (std::abs(f0 - f_disp_ref_ha_bohr) > f_disp_tol_ha_bohr
            || std::abs(f1 + f_disp_ref_ha_bohr) > f_disp_tol_ha_bohr) {
            std::printf("FAIL: dispersion force (%+.9e, %+.9e) Ha/Bohr vs "
                        "simple-dftd3 gradient +-%.9e (tol %.1e)\n",
                        f0, f1, f_disp_ref_ha_bohr, f_disp_tol_ha_bohr);
            return 1;
        }
    }

    // ── Leg 1: the reported force IS the gradient of the reported energy ──
    double worst_fd = 0.0;
    {
        cpu::Frame f = ar2(r_fd);
        double fa[2][3];
        const double e0 = evaluate_total(f, fa);
        const auto fdisp = cpu::scf::D3BJ{}.forces(f);
        const double disp_ev_ang = std::abs(fdisp[1][2]) * kHaBohrInEvAng;
        std::printf("Total-energy FD at r=%.2f A (h=%.4f A): E=%.9f eV, "
                    "|F_disp|=%.4e eV/A\n",
                    r_fd, fd_h, e0, disp_ev_ang);
        for (int a = 0; a < 2; ++a) {
            for (int d = 0; d < 3; ++d) {
                const double save = get_coord(f, a, d);
                double scratch[2][3];
                set_coord(f, a, d, save + fd_h);
                const double ep = evaluate_total(f, scratch);
                set_coord(f, a, d, save - fd_h);
                const double em = evaluate_total(f, scratch);
                set_coord(f, a, d, save);
                const double fd_force = -(ep - em) / (2.0 * fd_h);
                const double res = std::abs(fd_force - fa[a][d]);
                std::printf("  atom %d dir %d: analytic %+.6e  fd %+.6e  "
                            "res %.3e eV/A\n",
                            a, d, fa[a][d], fd_force, res);
                worst_fd = std::max(worst_fd, res);
            }
        }
        if (!(disp_ev_ang > 1e-3)) {
            std::printf("FAIL: the dispersion force is %.3e eV/A — too small "
                        "for this geometry to gate anything\n",
                        disp_ev_ang);
            return 1;
        }
        if (!(worst_fd <= fd_tol_ev_ang)) {
            std::printf("FAIL: worst |F_analytic - dE_total/dR| = %.3e eV/A "
                        "exceeds %.1e — the reported force is not the "
                        "gradient of the reported energy (a missing "
                        "dispersion leg is worth %.3e eV/A here)\n",
                        worst_fd, fd_tol_ev_ang, disp_ev_ang);
            return 1;
        }
    }

    // ── Leg 2: NVE Born–Oppenheimer MD in the dispersion well ────────────
    cpu::scf::RHF rhf = method();
    cpu::Frame f = ar2(r_md);
    cpu::md::MD sim;
    sim.set_integrator(cpu::md::VelocityVerlet{dt_fs});
    sim.set_evaluator_ref(rhf); // borrow: energy_disp() stays readable
    sim.set_threads(1);         // 2 atoms — fork/join dwarfs the physics
    sim.enable_timing(false);
    sim.build();

    double e_first = 0.0, worst_drift = 0.0;
    double r_min = 1e30, r_max = 0.0;
    double ed_min = 1e30, ed_max = -1e30;
    for (int s = 0; s < n_steps; ++s) {
        if (sim.run(f, 1) != 1) {
            std::printf("FAIL: MD step %d did not run\n", s + 1);
            return 1;
        }
        if (!rhf.converged()) {
            std::printf("FAIL: SCF did not converge at step %d (%s)\n", s + 1,
                        rhf.stop_reason().c_str());
            return 1;
        }
        const double pe = f.meta.get(kEnergy);
        const double ke = f.meta.get(kKinetic);
        if (!std::isfinite(pe) || !std::isfinite(ke) || !(ke >= 0.0)) {
            std::printf("FAIL: non-finite energy at step %d (PE=%g KE=%g)\n",
                        s + 1, pe, ke);
            return 1;
        }
        if (s == 0)
            e_first = pe + ke;
        worst_drift = std::max(worst_drift, std::abs(pe + ke - e_first));
        const double r = sep(f);
        r_min = std::min(r_min, r);
        r_max = std::max(r_max, r);
        const double ed_ev = rhf.energy_disp() * units::kHartreeInEv;
        ed_min = std::min(ed_min, ed_ev);
        ed_max = std::max(ed_max, ed_ev);
    }
    const double disp_span = ed_max - ed_min;
    const double ratio = disp_span / std::max(worst_drift, 1e-30);

    std::printf("NVE BOMD RHF/def2-SVP+D3(BJ) Ar2: %d steps x %.1f fs = "
                "%.0f fs\n",
                n_steps, dt_fs, n_steps * dt_fs);
    std::printf("  r in [%.4f, %.4f] A   E_disp in [%.6e, %.6e] eV "
                "(span %.4e eV)\n",
                r_min, r_max, ed_min, ed_max, disp_span);
    std::printf("  worst |E_tot - E_1| = %.4e eV (tol %.1e)   "
                "span/drift = %.1f (min %.0f)\n",
                worst_drift, drift_tol_ev, ratio, min_span_over_drift);

    if (!(r_min >= r_lo) || !(r_max <= r_hi)) {
        std::printf("FAIL: separation left [%.2f, %.2f] A — the dimer is not "
                    "bound. Only dispersion binds it at this level, so this "
                    "is what a missing dispersion force looks like\n",
                    r_lo, r_hi);
        return 1;
    }
    if (!(disp_span >= min_disp_span_ev)) {
        std::printf("FAIL: E_disp varies by only %.3e eV along the "
                    "trajectory (need %.1e) — this case no longer probes the "
                    "dispersion leg\n",
                    disp_span, min_disp_span_ev);
        return 1;
    }
    if (!(worst_drift <= drift_tol_ev)) {
        std::printf("FAIL: NVE drift %.4e eV exceeds %.1e — the force is not "
                    "the gradient of the energy along the trajectory\n",
                    worst_drift, drift_tol_ev);
        return 1;
    }
    if (!(ratio >= min_span_over_drift)) {
        std::printf("FAIL: the conserved energy holds to only %.0fx the size "
                    "of the dispersion term it must account for (need %.0fx)\n",
                    ratio, min_span_over_drift);
        return 1;
    }

    std::printf("PASS: bomd-d3bj-ar2-nve (D3(BJ) energy+gradient vs "
                "simple-dftd3; analytic force = dE_total/dR to %.2e eV/A; "
                "NVE drift %.2e eV over %.0f fs while E_disp swings %.2e eV)\n",
                worst_fd, worst_drift, n_steps * dt_fs, disp_span);
    return 0;
}
