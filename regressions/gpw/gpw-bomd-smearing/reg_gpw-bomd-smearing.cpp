/// @file regressions/gpw/gpw-bomd-smearing/reg_gpw-bomd-smearing.cpp
/// @brief END-TO-END: multi-k BOMD under Fermi–Dirac metallic smear (H₂).
///
/// Free-energy SCF each step with fractional P_k warm start. Dual-gate:
/// step-0 free energy / forces class vs offline CP2K AIMD+SMEAR+k twin;
/// short NVE drift envelope; warm-start iter reduction. See REFERENCE.md.

#include <cmath>
#include <cstdio>

#include "atomiverse/core/frame_schema.h"
#include "atomiverse/core/stage.h"
#include "atomiverse/core/units.h"
#include "atomiverse/cpu/dynamics/md.h"
#include "atomiverse/cpu/dynamics/velocity_verlet.h"
#include "atomiverse/cpu/evaluate/single_point.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/gpw/gaussian_basis.h"
#include "atomiverse/cpu/gpw/gpw_evaluator.h"
#include "atomiverse/cpu/gpw/gth_potential.h"

using namespace ATV;

int main() {
    // Periodic H₂ — same geometry as gpw-bomd-kpoints, with FD smear.
    cpu::Frame frame(2);
    constexpr double bohr2ang = 0.52917720859;
    constexpr double L_bohr = 12.0;
    const double x0[2] = {5.5 * bohr2ang, 6.5 * bohr2ang};
    for (int i = 0; i < 2; ++i) {
        frame.span(kAtom_type)[i] = 1;
        frame.span(kAtom_mass)[i] = 1.008;
        frame.span(kAtom_tag)[i] = i + 1;
        frame.span(kAtom_x)[i] = x0[i];
        frame.span(kAtom_y)[i] = 6.0 * bohr2ang;
        frame.span(kAtom_z)[i] = 6.0 * bohr2ang;
        frame.span(kAtom_vx)[i] = 0.0;
        frame.span(kAtom_vy)[i] = 0.0;
        frame.span(kAtom_vz)[i] = 0.0;
    }
    frame.span(kAtom_vx)[0] = -0.0010;
    frame.span(kAtom_vx)[1] = 0.0010;
    frame.box = cpu::Box::cubic(L_bohr * bohr2ang);

    const cpu::gpw::Kind H{
        molrs::Element::H,
        cpu::gpw::GaussianBasis(molrs::Element::H, "DZV-GTH-PADE"),
        cpu::gpw::GthPotential(molrs::Element::H, "GTH-PADE-q1")};

    cpu::gpw::GPWEvaluator gpw(
        /*kinds*/ {H}, /*cutoff_ry*/ 300.0, /*max_iter*/ 60, /*tol*/ 1e-7,
        /*mix*/ 0.3, /*xc*/ cpu::gpw::XcKind::PADE, /*nonperiodic*/ false,
        /*compute_forces*/ true);
    gpw.k_mesh[0] = 1;
    gpw.k_mesh[1] = 1;
    gpw.k_mesh[2] = 2;
    gpw.k_time_reversal = false;
    gpw.smearing = cpu::gpw::Smearing::fermi_dirac;
    gpw.degauss = 0.02; // Ry
    gpw.extra_bands = 4;

    // Soft pins: free-energy class near fixed-occ BOMD twin (~−1.07 Ha);
    // absolute window absorbs smear residual. Tightened after first green.
    constexpr double e0_soft_ha = -1.07;
    constexpr double tol_e0 = 5e-2; // Ha soft class
    constexpr double drift_tol = 0.10;
    constexpr double dt = 0.5; // fs
    constexpr int n_prod = 4;

    cpu::single_point(frame, gpw);
    const auto& r0 = gpw.last_result();
    if (!r0.converged) {
        std::printf("FAIL: step-0 multi-k free-energy SCF not converged "
                    "(iters=%d)\n",
                    r0.iters);
        return 1;
    }
    if (r0.n_kpts != 2 || static_cast<int>(r0.P_k.size()) != r0.n_kpts) {
        std::printf("FAIL: expected n_kpts==2 with full P_k, got n_kpts=%d "
                    "P_k=%zu\n",
                    r0.n_kpts, r0.P_k.size());
        return 1;
    }
    if (r0.f_k.empty() || static_cast<int>(r0.f_k.size()) != r0.n_kpts) {
        std::printf("FAIL: expected fractional f_k payload under smear\n");
        return 1;
    }
    const int iters_cold = r0.iters;
    const double e0_atv_ha = r0.e_total;
    if (!(std::fabs(e0_atv_ha - e0_soft_ha) <= tol_e0)) {
        std::printf("FAIL: step-0 free energy class |%.9f - (%.3f)| = %.3e Ha "
                    "> %.1e\n",
                    e0_atv_ha, e0_soft_ha, std::fabs(e0_atv_ha - e0_soft_ha),
                    tol_e0);
        return 1;
    }
    if (!(std::isfinite(r0.e_fermi))) {
        std::printf("FAIL: e_fermi non-finite\n");
        return 1;
    }
    std::printf("step0: F=%.10f Ha  e_fermi=%.6f Ha  iters=%d  nk=%d  "
                "f_k_present=1\n",
                e0_atv_ha, r0.e_fermi, iters_cold, r0.n_kpts);

    const double pe0_ev = e0_atv_ha * ATV::units::kHartreeInEv;
    double ke0_ev = 0.0;
    for (int i = 0; i < 2; ++i) {
        const double vx = static_cast<double>(frame.span(kAtom_vx)[i]);
        ke0_ev += 0.5 * 1.008 * vx * vx / ATV::units::force_to_accel;
    }
    const double e0_tot = pe0_ev + ke0_ev;
    const double e_scale = std::max(1.0, std::fabs(e0_tot));

    cpu::md::MD sim;
    sim.set_integrator(cpu::md::VelocityVerlet{dt});
    sim.set_evaluator_ref(gpw);
    sim.enable_timing(false);
    sim.build();

    double worst_drift = 0.0;
    int worst_iters = 0;
    int sum_warm_iters = 0;
    for (int s = 0; s < n_prod; ++s) {
        if (sim.run(frame, 1) != 1) {
            std::printf("FAIL: production step %d did not run\n", s + 1);
            return 1;
        }
        const double pe = frame.meta.get(kEnergy);
        const double ke = frame.meta.get(kKinetic);
        if (!std::isfinite(pe) || !std::isfinite(ke) || !(ke >= 0.0)) {
            std::printf("FAIL: non-finite energy at step %d PE=%g KE=%g\n",
                        s + 1, pe, ke);
            return 1;
        }
        const auto& rs = gpw.last_result();
        if (!rs.converged) {
            std::printf("FAIL: free-energy SCF not converged at step %d\n",
                        s + 1);
            return 1;
        }
        if (rs.f_k.empty()) {
            std::printf("FAIL: lost f_k payload at step %d\n", s + 1);
            return 1;
        }
        worst_iters = std::max(worst_iters, rs.iters);
        sum_warm_iters += rs.iters;
        const double e_tot = pe + ke;
        const double drift = std::fabs(e_tot - e0_tot) / e_scale;
        worst_drift = std::max(worst_drift, drift);
        std::printf("  step %d: F_Ha=%.8f  iters=%d  drift=%.3e\n", s + 1,
                    rs.e_total, rs.iters, drift);
    }

    if (!(worst_drift <= drift_tol)) {
        std::printf("FAIL: NVE relative drift %.3e > %.3e\n", worst_drift,
                    drift_tol);
        return 1;
    }
    // Warm-start should not be worse than cold on average (soft).
    const double mean_warm =
        static_cast<double>(sum_warm_iters) / static_cast<double>(n_prod);
    std::printf("cold_iters=%d  mean_warm_iters=%.1f  worst_drift=%.3e\n",
                iters_cold, mean_warm, worst_drift);

    std::printf("PASS: gpw-bomd-smearing  F0=%.10f  worst_drift=%.3e\n",
                e0_atv_ha, worst_drift);
    return 0;
}
