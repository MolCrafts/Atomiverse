/// @file regressions/gpw/gpw-water-pbe-mt/reg_gpw-water-pbe-mt.cpp
/// @brief END-TO-END: isolated H2O GPW PBE / DZVP-MOLOPT-GTH (MT) vs offline
///        CP2K getting-started pin. See REFERENCE.md.
///
/// Public API only. Requires ATV_BUILD_GPW (+ HF + PW). Exit non-zero on
/// mismatch with got / want / tol. <= 60 s, single CPU. No live CP2K.

#include <cmath>
#include <cstdio>

#include "atomiverse/cpu/evaluate/single_point.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/gpw/gaussian_basis.h"
#include "atomiverse/cpu/gpw/gpw_evaluator.h"
#include "atomiverse/cpu/gpw/gth_potential.h"
#include "atomiverse/molrs.hpp"

int main() {
    using namespace ATV;

    // CP2K getting-started h2o.inp geometry [Å], 10 Å cubic cell.
    cpu::Frame frame(3);
    const int z[3] = {8, 1, 1};
    const double x[3] = {5.0000, 5.7586, 4.2414};
    const double y[3] = {5.0000, 5.0000, 5.0000};
    const double zc[3] = {5.0000, 5.5043, 5.5043};
    for (int i = 0; i < 3; ++i) {
        frame.span(kAtom_type)[i] = z[i];
        frame.span(kAtom_x)[i] = x[i];
        frame.span(kAtom_y)[i] = y[i];
        frame.span(kAtom_z)[i] = zc[i];
    }
    frame.box = cpu::Box::cubic(10.0);

    const cpu::gpw::Kind O{
        molrs::Element::O,
        cpu::gpw::GaussianBasis(molrs::Element::O, "DZVP-MOLOPT-GTH"),
        cpu::gpw::GthPotential(molrs::Element::O, "GTH-PBE-q6")};
    const cpu::gpw::Kind H{
        molrs::Element::H,
        cpu::gpw::GaussianBasis(molrs::Element::H, "DZVP-MOLOPT-GTH"),
        cpu::gpw::GthPotential(molrs::Element::H, "GTH-PBE-q1")};

    cpu::gpw::GPWEvaluator eval(
        /*kinds*/ {O, H}, /*cutoff_ry*/ 400.0, /*max_iter*/ 200, /*tol*/ 1e-7,
        /*mix*/ 0.3, /*xc*/ cpu::gpw::XcKind::PBE, /*nonperiodic*/ true);
    cpu::single_point(frame, eval);

    // Offline CP2K pin (getting-started h2o.inp family). See REFERENCE.md.
    constexpr double kEnergyRef = -17.21233714458516; // Ha
    constexpr double kTol = 1.0e-3;                   // Ha — e2e band
    const double got = eval.last_energy_hartree();
    const bool converged = eval.last_result().converged;

    if (!converged) {
        std::fprintf(stderr,
                     "FAIL: GPW H2O PBE/MT did not converge (iters=%d)\n",
                     eval.last_result().iters);
        return 1;
    }
    if (!std::isfinite(got) || std::abs(got - kEnergyRef) > kTol) {
        std::fprintf(
            stderr,
            "FAIL: E_GPW(H2O,PBE,MT) got=%.10f want=%.10f tol=%.1e Ha\n", got,
            kEnergyRef, kTol);
        return 1;
    }
    if (!std::isfinite(frame.meta.get(kEnergy))) {
        std::fprintf(stderr, "FAIL: non-finite frame.meta[kEnergy]\n");
        return 1;
    }

    std::printf("PASS: E_GPW(H2O,PBE,DZVP-MOLOPT-GTH,MT) = %.10f Ha "
                "(want %.10f +/- %.1e) |dE|=%.3e Ha  iters=%d  "
                "[offline CP2K getting-started h2o.inp pin]\n",
                got, kEnergyRef, kTol, std::abs(got - kEnergyRef),
                eval.last_result().iters);
    return 0;
}
