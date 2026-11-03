/// @file regressions/pw/pbe-family/pbe-family-qe.cpp
/// @brief END-TO-END: Plane-wave PBEsol/revPBE/RPBE (Si2 RKS, H3 UKS) vs actual QE 7.6 output.
///
/// See REFERENCE.md. Pins are parsed from the independent engine's own stdout
/// (never from Atomiverse); the engine inputs are under this directory.
#include "atomiverse/core/units.h"
#include "atomiverse/cpu/planewave/pw_solver.h"
#include "atomiverse/cpu/planewave/qe_units.h"
#include "qe_pins.h"
#ifdef ATV_HAS_CUDA
#include "atomiverse/cuda/planewave/pw_solver.cuh"
#endif
#include "harness/registry.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>

namespace atv_reg_pw_pbe_family_qe {

using namespace ATV;
namespace {
void require(bool value, const std::string& message) {
    if (!value)
        throw std::runtime_error(message);
}
void close(double value, double reference, double tolerance,
           const std::string& label) {
    if (!std::isfinite(value) || !std::isfinite(reference)
        || std::abs(value - reference) > tolerance) {
        std::ostringstream error;
        error.precision(17);
        error << label << ": got " << value << " expected " << reference
              << " tolerance " << tolerance;
        throw std::runtime_error(error.str());
    }
}
cpu::Frame frame(const pbe_periodic_qe::Pin& pin) {
    cpu::Frame f(pin.nat);
    constexpr double h[3][3] = {
        {1.2, 1.1, 1}, {2.1, 1.6, 1.3}, {1.6, 2.25, 1.8}};
    constexpr double si[2][3] = {{2, 2, 2}, {3.8, 3.2, 3}};
    if (pin.spin) {
        Mat3 cell{};
        cell(0, 0) = 4.2;
        cell(0, 1) = .3;
        cell(0, 2) = .2;
        cell(1, 1) = 4;
        cell(1, 2) = .1;
        cell(2, 2) = 3.8;
        f.box.set_cell(cell);
    } else
        f.box = cpu::Box::cubic(6);
    for (int i = 0; i < pin.nat; ++i) {
        f.span(kAtom_atomic_number)[i] = pin.spin ? 1 : 14;
        f.span(kAtom_tag)[i] = i + 1;
        f.span(kAtom_mass)[i] = pin.spin ? 1.008 : 28.0855;
        f.span(kAtom_x)[i] = pin.spin ? h[i][0] : si[i][0];
        f.span(kAtom_y)[i] = pin.spin ? h[i][1] : si[i][1];
        f.span(kAtom_z)[i] = pin.spin ? h[i][2] : si[i][2];
    }
    return f;
}
cpu::xc::XCType xc(const pbe_periodic_qe::Pin& pin) {
    constexpr cpu::xc::XCType models[] = {cpu::xc::XCType::PBEsol,
                                          cpu::xc::XCType::RevPBE,
                                          cpu::xc::XCType::RPBE};
    return models[pin.functional];
}
template <class Method>
void run_case(const pbe_periodic_qe::Pin& pin, const char* backend) {
    auto f = frame(pin);
    auto potential = std::make_shared<cpu::pw::GthPseudopotential>(
        pin.spin ? molrs::Element::H : molrs::Element::Si,
        pin.spin ? "GTH-PBE-q1" : "GTH-PBE-q4");
    Method m{potential, pin.spin ? 6. : 12., 1, xc(pin)};
    m.spin_polarized = pin.spin;
    m.multiplicity = pin.spin ? 2 : 1;
    m.smearing = electronic::Smearing::fermi_dirac;
    m.degauss = .02;
    m.extra_bands = pin.spin ? 3 : 2;
    m.max_iter = pin.spin ? 250 : 200;
    m.tol = 1e-9;
    m.mix_beta = pin.spin ? .35 : .2;
    const auto result = m.forces(f);
    require(result.converged, "reference force SCF convergence");
    double energy_error = std::abs(result.energy_ha - pin.energy_ha),
           force_error = 0, stress_error = 0;
    close(result.energy_ha, pin.energy_ha, pbe_periodic_qe::energy_tolerance_ha,
          "actual QE absolute Mermin energy/Ha");
    const auto fx = f.span(kAtom_fx), fy = f.span(kAtom_fy),
               fz = f.span(kAtom_fz);
    double force_peak = 0;
    for (int i = 0; i < pin.nat; ++i)
        for (int d = 0; d < 3; ++d) {
            const double got = (d == 0   ? fx[i]
                                : d == 1 ? fy[i]
                                         : fz[i])
                               / units::kHartreePerBohrInEvPerAngstrom;
            close(got, pin.forces_ha_bohr[i][d],
                  pbe_periodic_qe::force_tolerance_ha_bohr,
                  "actual QE full ionic force/Ha/Bohr");
            force_error =
                std::max(force_error, std::abs(got - pin.forces_ha_bohr[i][d]));
            force_peak =
                std::max(force_peak, std::abs(pin.forces_ha_bohr[i][d]));
        }
    require(force_peak > 1e-3, "independent force pin must be nontrivial");
    close(f.meta.get(kEnergy), result.energy * cpu::pw::qe_units::rydberg_to_ev,
          1e-9, "public energy publication");
    close(result.n_elec_occ, pin.spin ? 3. : 8., 1e-8,
          "correct thermally occupied electron count");
    require(std::isfinite(result.e_entropy) && result.e_entropy < -1e-5,
            "nonzero Mermin entropy term");
    if (pin.spin)
        require(std::abs(result.magnetization) > .1,
                "nontrivial collinear spin state");
    const auto sigma = m.stress(f);
    require(m.last_result().converged, "stress reference convergence");
    close(m.last_result().energy_ha, result.energy_ha, 1e-8,
          "force/stress reference energy identity");
    for (int k = 0; k < 6; ++k) {
        const double target =
            pin.stress_ha_bohr3[k] * units::kHartreePerBohr3InEvPerAngstrom3;
        close(sigma[k], target, pbe_periodic_qe::stress_tolerance_ev_ang3,
              "actual QE physical stress/eV/A^3");
        close(f.meta.get(kStress)[k], sigma[k] * f.box.volume(), 1e-10,
              "public virial conversion");
        stress_error = std::max(stress_error, std::abs(sigma[k] - target));
    }
    std::printf("PASS PBE_PERIODIC_QE %s %s E_Ha=%.15g dE_Ha=%.9g "
                "dF_Ha_Bohr=%.9g dStress_eV_A3=%.9g iterations=%d\n",
                pin.id, backend, result.energy_ha, energy_error, force_error,
                stress_error, result.iterations);
    std::fflush(stdout);
}
} // namespace

int run() {
    int failed = 0;
    for (const auto& pin : pbe_periodic_qe::pins) {
        try {
            run_case<cpu::pw::PW>(pin, "CPU");
        } catch (const std::exception& e) {
            ++failed;
            std::fprintf(stderr, "FAIL PBE_PERIODIC_QE %s CPU: %s\n", pin.id,
                         e.what());
        }
#ifdef ATV_HAS_CUDA
        try {
            run_case<cuda::pw::PW>(pin, "CUDA");
        } catch (const std::exception& e) {
            ++failed;
            std::fprintf(stderr, "FAIL PBE_PERIODIC_QE %s CUDA: %s\n", pin.id,
                         e.what());
        }
#endif
    }
    return failed ? 1 : 0;
}

} // namespace atv_reg_pw_pbe_family_qe

ATV_REGRESSION("pw/pbe-family", "pbe-family-qe", atv_reg_pw_pbe_family_qe::run);
