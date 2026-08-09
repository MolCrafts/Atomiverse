/// @file regressions/gpw/gpw-variable-cell-molopt/reg_gpw-variable-cell-molopt.cpp
/// @brief END-TO-END: GPW CELL_OPT MOLOPT re-twin vs offline CP2K.
///
/// Same ice-like periodic water chain as `gpw-variable-cell-ice`, but with a
/// MOLOPT basis (DZVP-MOLOPT-SR-GTH) + PBE + GTH-PBE — not the compact
/// GTH-PADE pivot. Full DZVP-MOLOPT-GTH periodic image-sum collocation
/// exceeds the ≤ 60 s e2e budget even for one SCF; the short-range MOLOPT
/// table (same family used by `gpw-pbe-d3bj-ice` / `gpw-ts-ice`) is the
/// product twin that still proves CELL_OPT with a MOLOPT Gaussian basis.
/// Fixed fractional coordinates; chain axis a relaxes (CP2K CONSTRAINT YZ +
/// FIXED_ATOMS). Product analytic fixed-P stress. Public API only.
/// Requires ATV_BUILD_GPW (+ HF + PW).

#include <cmath>
#include <cstdio>

#include "atomiverse/core/frame_schema.h"
#include "atomiverse/core/stage.h"
#include "atomiverse/cpu/driver/criteria.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/gpw/gaussian_basis.h"
#include "atomiverse/cpu/gpw/gpw_evaluator.h"
#include "atomiverse/cpu/gpw/gth_potential.h"
#include "atomiverse/cpu/minimize/anisotropic_cell_relax.h"
#include "atomiverse/cpu/minimize/minimize.h"

using namespace ATV;

namespace {

// Offline CP2K 2026.1 CELL_OPT (DZVP-MOLOPT-SR-GTH / GTH-PBE / PBE, fixed
// ions, CONSTRAINT YZ, ANALYTICAL stress, from a0 = 3.328 Å). See REFERENCE.
constexpr double kA_cp2k = 3.200237;             // Å
constexpr double kE_cp2k = -17.184075549824836;  // Ha

// Twin geometry: fixed fractional coordinates (CP2K SCALED); b = c = 8 Å.
// Start slightly expanded so the Newton step is short (budget).
constexpr double kA0 = 3.24;
constexpr double kB = 8.0, kC = 8.0;
constexpr double kF1x = 0.319067;  // H1 (donor, on the chain axis)
constexpr double kF2x = -0.079900; // H2 x
constexpr double kF2z = 0.115830;  // H2 z

constexpr double kATol = 0.02; // Å on the relaxed chain axis
constexpr double kETol = 1e-3; // Ha on the final energy
constexpr double kStol = 5e-4; // eV/Å³ residual stress gate (criterion)

} // namespace

int main() {
    cpu::Frame frame(3);
    const int z[3] = {8, 1, 1};
    const double x[3] = {0.0, kF1x * kA0, kF2x * kA0};
    const double y[3] = {0.0, 0.0, 0.0};
    const double zc[3] = {0.0, 0.0, kF2z * kC};
    for (int i = 0; i < 3; ++i) {
        frame.span(kAtom_type)[i] = z[i];
        frame.span(kAtom_x)[i] = x[i];
        frame.span(kAtom_y)[i] = y[i];
        frame.span(kAtom_z)[i] = zc[i];
        frame.span(kAtom_fx)[i] = frame.span(kAtom_fy)[i] =
            frame.span(kAtom_fz)[i] = 0;
    }
    frame.box = cpu::Box::orthorhombic(kA0, kB, kC);

    const cpu::gpw::Kind O{
        molrs::Element::O,
        cpu::gpw::GaussianBasis(molrs::Element::O, "DZVP-MOLOPT-SR-GTH"),
        cpu::gpw::GthPotential(molrs::Element::O, "GTH-PBE-q6")};
    const cpu::gpw::Kind H{
        molrs::Element::H,
        cpu::gpw::GaussianBasis(molrs::Element::H, "DZVP-MOLOPT-SR-GTH"),
        cpu::gpw::GthPotential(molrs::Element::H, "GTH-PBE-q1")};

    cpu::gpw::GPWEvaluator eval(
        /*kinds*/ {O, H}, /*cutoff_ry*/ 300.0, /*max_iter*/ 200, /*tol*/ 1e-7,
        /*mix*/ 0.3, /*xc*/ cpu::gpw::XcKind::PBE, /*nonperiodic*/ false,
        /*compute_forces*/ false);
    eval.compute_stress = true;
    eval.stress_use_fd = false; // product analytic fixed-P (default)
    eval.stress_strain_h = 2e-3;
    eval.stress_axes = {true, false, false}; // only the chain axis relaxes

    cpu::minimizer::Minimize opt;
    cpu::minimizer::AnisotropicCellRelax cell(/*P_target*/ 0.0, /*alpha*/ 1.0);
    // dσ_xx/dε at a* from CP2K full-MOLOPT E(a) curvature (REFERENCE) —
    // same order for SR twin → near-Newton step.
    cell.bulk_mod = 0.20;
    cell.max_scale = 0.05;
    cell.relax_y = false;
    cell.relax_z = false;
    opt.set_optimizer_ref(cell);
    opt.add_criterion(cpu::StressConverged{0.0, kStol,
                                           /*x*/ true, /*y*/ false,
                                           /*z*/ false});
    opt.on(minimizer::Phase::force).add_ref(eval);
    opt.enable_timing(false);
    opt.build();
    opt.run(frame, 6);

    if (!eval.last_result().converged) {
        std::printf("FAIL: GPW SCF did not converge (iters=%d)\n",
                    eval.last_result().iters);
        return 1;
    }
    if (!opt.converged()) {
        std::printf("FAIL: cell relax did not converge within budget "
                    "(residual=%.3e eV/A3)\n",
                    cell.pressure_residual());
        return 1;
    }

    const double a1 = frame.box.length(0);
    const double b1 = frame.box.length(1);
    const double c1 = frame.box.length(2);
    const double e_ha = eval.last_energy_hartree();
    // kStress is the extensive Voigt virial [eV]; this gate is written
    // against the intensive stress W_xx/V [eV/A^3].
    const double sxx_ev_a3 =
        frame.meta.get(kStress)[0] / frame.box.volume();
    const double sxx_ha_bohr3 = eval.last_stress_ha_bohr3()[0];

    std::printf(
        "a: start=%.4f  relaxed=%.6f  cp2k CELL_OPT=%.6f  |da|=%.4f A\n", kA0,
        a1, kA_cp2k, std::fabs(a1 - kA_cp2k));
    std::printf("E: atv=%.12f Ha  cp2k=%.12f Ha  |dE|=%.3e Ha\n", e_ha, kE_cp2k,
                std::fabs(e_ha - kE_cp2k));
    std::printf("sigma_xx(relaxed) = %.3e eV/A3 = %.3e Ha/Bohr3  (%s)\n",
                sxx_ev_a3, sxx_ha_bohr3, opt.stop_reason().c_str());

    if (!std::isfinite(a1) || !std::isfinite(e_ha)) {
        std::printf("FAIL: non-finite result\n");
        return 1;
    }
    if (b1 != kB || c1 != kC) {
        std::printf("FAIL: constrained b/c moved: b=%.17g c=%.17g\n", b1, c1);
        return 1;
    }
    if (std::fabs(a1 - kA_cp2k) > kATol) {
        std::printf("FAIL: |a_atv - a_cp2k| = %.4f > %.3f A\n",
                    std::fabs(a1 - kA_cp2k), kATol);
        return 1;
    }
    if (std::fabs(sxx_ev_a3) > 1.2 * kStol) {
        std::printf("FAIL: residual stress %.3e eV/A3 above gate\n", sxx_ev_a3);
        return 1;
    }
    if (std::fabs(e_ha - kE_cp2k) > kETol) {
        std::printf("FAIL: |E - E_cp2k| = %.3e Ha > %g\n",
                    std::fabs(e_ha - kE_cp2k), kETol);
        return 1;
    }

    std::printf("PASS: gpw-variable-cell-molopt  a=%.5f A (cp2k %.5f)  "
                "|dE|=%.3e Ha  sigma_xx=%.2e eV/A3\n",
                a1, kA_cp2k, std::fabs(e_ha - kE_cp2k), sxx_ev_a3);
    return 0;
}
