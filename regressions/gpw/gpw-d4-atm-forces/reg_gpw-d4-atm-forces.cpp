/// @file regressions/gpw/gpw-d4-atm-forces/reg_gpw-d4-atm-forces.cpp
/// @brief END-TO-END: periodic water-dimer GPW PBE-D4+ATM (s9=1) analytic
///        ionic forces.
///
/// Hard pins: offline dftd4 3.7.0 `--grad` with ATM on (pbe-D4-ATM): E_disp
/// + −∇E + the ATM-only force increment vs the two-body `--mbdscale 0` pin
/// (5e-8 Ha/Bohr — the oracle's own periodic ATM gradient is
/// non-conservative by ~2.4e-8 vs its printed energy, see REFERENCE.md; the
/// ATM increment is ~4.1e-7, so the gate still fails if the ATM gradient is
/// dropped), and a sharp 1e-9 conservative-parity gate against the FD of the
/// oracle's printed displaced energies. Wiring: product F(d4_atm)−F(plain)
/// matches standalone D4+ATM. Analytic-vs-FD dual gate of the s9=1
/// functional lives in tests/cpu/gpw/test_cpu_gpw_dftd4.cpp (6N SCFs on this
/// cell exceed the ≤60 s budget). Public API only. See REFERENCE.md.

#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <vector>

#include "atomiverse/core/frame_schema.h"
#include "atomiverse/core/stage.h"
#include "atomiverse/core/units.h"
#include "atomiverse/cpu/evaluate/evaluate.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/gpw/gaussian_basis.h"
#include "atomiverse/cpu/gpw/gpw_evaluator.h"
#include "atomiverse/cpu/gpw/gpw_solver.h"
#include "atomiverse/cpu/gpw/gth_potential.h"
#include "atomiverse/cpu/gpw/internal/gpw_forces.h"
#include "atomiverse/cpu/scf/dftd4.h"
#include "atomiverse/molrs.hpp"

using namespace ATV;

namespace {

// dftd4 3.7.0 -f pbe (s9=1) on the periodic 12 Å POSCAR (REFERENCE.md).
constexpr double kEDispAtmRef = -9.9554475834240310e-04; // Ha
constexpr double kEDispTol = 1.0e-8;                     // Ha

// dftd4 3.7.0 --grad, ATM on, POSCAR order O O H H H H remapped to
// O H H O H H (pin_periodic_atm_grad.json). Force = −gradient.
constexpr double kGradPerAtm[6][3] = {
    {-5.3595427766884764e-05, -3.3533040430274789e-24, -4.7093130846514167e-22},
    {-5.9463596930597288e-05, -8.8497804033039618e-24, -1.0397154303254306e-05},
    {-5.9463596930597301e-05, -4.9419094613683133e-24, 1.0397154303254314e-05},
    {1.3610085695186597e-04, 1.3230988245241392e-23, -8.4467780181584667e-20},
    {1.8210882338106763e-05, -1.1201532595111321e-24, -8.1642912818385456e-06},
    {1.8210882338106668e-05, 1.8647645948199754e-23, 8.1642912818386235e-06},
};
// Two-body-only twin (dftd4 --mbdscale 0 --grad; gpw-d4-forces REFERENCE):
// the difference to kGradPerAtm is the oracle's ATM-only gradient.
constexpr double kGradPer2b[6][3] = {
    {-5.3738961271913915e-05, -1.5591535100726387e-24, -6.1987516039752966e-21},
    {-5.9550148958160894e-05, -3.0412121452621079e-25, -1.0810592129949165e-05},
    {-5.9550148958160921e-05, -5.7104654456581212e-25, 1.0810592129949182e-05},
    {1.3609823922940861e-04, 6.8060330704498828e-25, -8.4063699110321910e-20},
    {1.8370509979413634e-05, 5.7191630572539285e-25, -8.5497124425277334e-06},
    {1.8370509979413539e-05, 1.2193640572572847e-24, 8.5497124425278096e-06},
};
// Direct gradient-pin gate: above the oracle's own ~2.4e-8 periodic-ATM
// non-conservativeness, far below the 4.1e-7 ATM increment it gates.
constexpr double kForcePinTol = 5.0e-8; // Ha/Bohr vs dftd4 --grad (hard)
// Conservative parity: FD over the oracle's printed s9=1 energies with O0
// displaced x ± 0.001 Å (pin_periodic_atm_dispO0x_{plus,minus}.json).
constexpr double kEPerAtmDispO0xPlus = -9.9564614957741663e-04;  // Ha
constexpr double kEPerAtmDispO0xMinus = -9.9544350841746518e-04; // Ha
constexpr double kDispO0xH = 0.001;                              // Å
constexpr double kFdPinTol = 1.0e-9;  // Ha/Bohr (truncation ~3e-10)
constexpr double kWiringTol = 1.0e-7; // Ha/Bohr additive (SCF residual)
constexpr double kAng2Bohr = 1.8897259885789;

} // namespace

int main() {
    using clock = std::chrono::steady_clock;

    constexpr double L = 12.0; // Å
    const int z[6] = {8, 1, 1, 8, 1, 1};
    const double x[6] = {0.000000, 0.758602, 0.758602,
                         3.000000, 3.758602, 3.758602};
    const double y[6] = {0.000000, 0.000000, 0.000000,
                         0.000000, 0.000000, 0.000000};
    const double zc[6] = {0.000000, 0.504284, -0.504284,
                          0.000000, 0.504284, -0.504284};

    // ── Standalone D4+ATM forces vs offline pin ───────────────────────────
    std::vector<int> Zv(6);
    std::vector<std::array<double, 3>> pos_bohr(6);
    for (int i = 0; i < 6; ++i) {
        Zv[static_cast<std::size_t>(i)] = z[i];
        pos_bohr[static_cast<std::size_t>(i)] = {
            x[i] * kAng2Bohr, y[i] * kAng2Bohr, zc[i] * kAng2Bohr};
    }
    const double cell_b[3] = {L * kAng2Bohr, L * kAng2Bohr, L * kAng2Bohr};
    const auto d4atm = cpu::scf::dftd4(
        Zv, pos_bohr, cell_b, /*periodic=*/true, /*s6=*/1.0,
        /*s8=*/0.95948085, /*a1=*/0.38574991, /*a2=*/4.80688534, /*ga=*/3.0,
        /*gc=*/2.0, /*wf=*/6.0, /*charge=*/0.0, /*cn_cut=*/30.0,
        /*disp2_cut=*/60.0, /*s9=*/1.0);
    if (!std::isfinite(d4atm.energy)
        || std::abs(d4atm.energy - kEDispAtmRef) > kEDispTol) {
        std::printf("FAIL: |E_disp-pin|=%.6e Ha  got=%.14e  want=%.14e\n",
                    std::abs(d4atm.energy - kEDispAtmRef), d4atm.energy,
                    kEDispAtmRef);
        return 1;
    }
    double max_pin_err = 0.0;
    for (int i = 0; i < 6; ++i) {
        for (int k = 0; k < 3; ++k) {
            const double want = -kGradPerAtm[i][k];
            const double got = d4atm.forces[static_cast<std::size_t>(i)]
                                           [static_cast<std::size_t>(k)];
            max_pin_err = std::max(max_pin_err, std::abs(got - want));
        }
    }
    if (max_pin_err > kForcePinTol) {
        std::printf("FAIL: max|F_d4atm+grad_pin|=%.3e > %.1e Ha/Bohr\n",
                    max_pin_err, kForcePinTol);
        return 1;
    }

    // ── Conservative parity vs FD of the oracle's printed energies ────────
    const double f_fd_o0x = -(kEPerAtmDispO0xPlus - kEPerAtmDispO0xMinus)
                            / (2.0 * kDispO0xH * kAng2Bohr);
    const double fd_pin_err = std::abs(d4atm.forces[0][0] - f_fd_o0x);
    if (fd_pin_err > kFdPinTol) {
        std::printf("FAIL: |F_x(O0)-FD(oracle E±)|=%.3e > %.1e Ha/Bohr\n",
                    fd_pin_err, kFdPinTol);
        return 1;
    }

    // ── ATM-only increment vs the oracle's pin delta ──────────────────────
    const auto d4_2b = cpu::scf::dftd4(Zv, pos_bohr, cell_b, /*periodic=*/true);
    double max_inc = 0.0;
    double max_inc_err = 0.0;
    for (int i = 0; i < 6; ++i) {
        for (int k = 0; k < 3; ++k) {
            const double inc = d4atm.forces[static_cast<std::size_t>(i)]
                                           [static_cast<std::size_t>(k)]
                               - d4_2b.forces[static_cast<std::size_t>(i)]
                                             [static_cast<std::size_t>(k)];
            const double want = -(kGradPerAtm[i][k] - kGradPer2b[i][k]);
            max_inc = std::max(max_inc, std::abs(inc));
            max_inc_err = std::max(max_inc_err, std::abs(inc - want));
        }
    }
    if (max_inc < 1e-8) {
        std::printf("FAIL: ATM force increment vacuous (max|dF|=%.3e)\n",
                    max_inc);
        return 1;
    }
    if (max_inc_err > kForcePinTol) {
        std::printf("FAIL: ATM increment max|dF-pin|=%.3e > %.1e Ha/Bohr\n",
                    max_inc_err, kForcePinTol);
        return 1;
    }
    std::printf("D4+ATM standalone: E=%.14e  max|F+g|=%.3e  "
                "|F_x(O0)-FD_pin|=%.3e  max|dF_atm-pin|=%.3e Ha/Bohr\n",
                d4atm.energy, max_pin_err, fd_pin_err, max_inc_err);

    // ── Product GPW with d4 + d4_atm analytic forces ──────────────────────
    cpu::Frame frame(6);
    for (int i = 0; i < 6; ++i) {
        frame.span(kAtom_type)[i] = z[i];
        frame.span(kAtom_x)[i] = x[i];
        frame.span(kAtom_y)[i] = y[i];
        frame.span(kAtom_z)[i] = zc[i];
        frame.span(kAtom_fx)[i] = frame.span(kAtom_fy)[i] =
            frame.span(kAtom_fz)[i] = 0;
    }
    frame.box = cpu::Box::cubic(L);

    const cpu::gpw::Kind O{
        molrs::Element::O,
        cpu::gpw::GaussianBasis(molrs::Element::O, "DZVP-MOLOPT-SR-GTH"),
        cpu::gpw::GthPotential(molrs::Element::O, "GTH-PBE-q6")};
    const cpu::gpw::Kind H{
        molrs::Element::H,
        cpu::gpw::GaussianBasis(molrs::Element::H, "DZVP-MOLOPT-SR-GTH"),
        cpu::gpw::GthPotential(molrs::Element::H, "GTH-PBE-q1")};

    cpu::gpw::GPWEvaluator eval({O, H});
    eval.cutoff_ry = 200.0;
    eval.max_iter = 120;
    eval.tol = 1e-6;
    eval.mix = 0.4;
    eval.xc = cpu::gpw::XcKind::PBE;
    eval.nonperiodic = false;
    eval.d4 = true;
    eval.d4_atm = true;
    eval.compute_forces = true;
    eval.compute_stress = false;

    const auto t0 = clock::now();
    cpu::Evaluate driver;
    driver.on(evaluate::Phase::evaluate).add_ref(eval);
    driver.enable_timing(false);
    driver.build();
    driver.run(frame);
    const auto t1 = clock::now();
    const double wall_product_s =
        std::chrono::duration<double>(t1 - t0).count();

    if (!eval.last_result().converged) {
        std::printf("FAIL: GPW SCF did not converge (iters=%d)\n",
                    eval.last_result().iters);
        return 1;
    }
    const double e_disp = eval.last_result().e_disp;
    if (!std::isfinite(e_disp) || std::abs(e_disp - kEDispAtmRef) > kEDispTol) {
        std::printf("FAIL: product |E_disp-pin|=%.6e Ha  got=%.14e\n",
                    std::abs(e_disp - kEDispAtmRef), e_disp);
        return 1;
    }

    // Build atoms + methods for the wiring gate.
    std::vector<cpu::gpw::GpwAtom> atoms;
    atoms.reserve(6);
    for (int i = 0; i < 6; ++i) {
        const cpu::gpw::Kind& k = (z[i] == 8) ? O : H;
        const double pos[3] = {x[i] * kAng2Bohr, y[i] * kAng2Bohr,
                               zc[i] * kAng2Bohr};
        atoms.push_back(cpu::gpw::GpwAtom(pos, k.basis, k.pot, z[i]));
    }
    cpu::gpw::GpwDft method_atm;
    method_atm.cell[0] = method_atm.cell[1] = method_atm.cell[2] =
        L * kAng2Bohr;
    method_atm.cutoff_ry = 200.0;
    method_atm.n_occ = 8;
    method_atm.max_iter = 120;
    method_atm.tol = 1e-6;
    method_atm.mix = 0.4;
    method_atm.xc = cpu::gpw::XcKind::PBE;
    method_atm.nonperiodic = false;
    method_atm.local_pp_on_grid = true;
    method_atm.d4 = true;
    method_atm.d4_atm = true;
    method_atm.build(atoms);

    // Product forces from the already-converged D4+ATM SCF.
    const auto F_on =
        cpu::gpw::gpw_forces_from_scf(atoms, method_atm, eval.last_result());

    // Plain electronic forces (no dispersion) — one SCF.
    cpu::gpw::GpwDft plain;
    plain.cell[0] = plain.cell[1] = plain.cell[2] = L * kAng2Bohr;
    plain.cutoff_ry = 200.0;
    plain.n_occ = 8;
    plain.max_iter = 120;
    plain.tol = 1e-6;
    plain.mix = 0.4;
    plain.xc = cpu::gpw::XcKind::PBE;
    plain.nonperiodic = false;
    plain.local_pp_on_grid = true;
    plain.d4 = false;
    plain.build(atoms);
    const auto F_plain = cpu::gpw::gpw_forces_analytic(atoms, plain);

    double max_wire = 0.0;
    double max_f_on = 0.0;
    for (int i = 0; i < 6; ++i) {
        for (int k = 0; k < 3; ++k) {
            const double on =
                F_on[static_cast<std::size_t>(i)][static_cast<std::size_t>(k)];
            const double pl = F_plain[static_cast<std::size_t>(i)]
                                     [static_cast<std::size_t>(k)];
            const double want = d4atm.forces[static_cast<std::size_t>(i)]
                                            [static_cast<std::size_t>(k)];
            max_wire = std::max(max_wire, std::abs((on - pl) - want));
            max_f_on = std::max(max_f_on, std::abs(on));
            if (!std::isfinite(on) || !std::isfinite(pl)) {
                std::printf("FAIL: non-finite force atom %d axis %d\n", i, k);
                return 1;
            }
        }
    }
    if (max_wire > kWiringTol) {
        std::printf("FAIL: wiring max|(F_atm-F_plain)-F_standalone|=%.3e "
                    "> %.1e Ha/Bohr\n",
                    max_wire, kWiringTol);
        return 1;
    }
    if (max_f_on < 1e-8) {
        std::printf("FAIL: product forces near zero (max|F|=%.3e)\n", max_f_on);
        return 1;
    }

    const double e_ha = eval.last_energy_hartree();
    std::printf("E_atv=%.12f Ha  E_disp=%.14e Ha (pin %.14e)  iters=%d\n", e_ha,
                e_disp, kEDispAtmRef, eval.last_result().iters);
    std::printf("PASS: gpw-d4-atm-forces  max|F+g|=%.3e  max|dF_atm-pin|=%.3e"
                "  max|wiring|=%.3e Ha/Bohr  wall_prod=%.3f s\n",
                max_pin_err, max_inc_err, max_wire, wall_product_s);
    return 0;
}
