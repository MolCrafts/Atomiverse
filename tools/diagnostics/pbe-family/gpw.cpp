// Opt-in complete GPW methods through Frame/Kind only. No independent oracle.
#include "atomiverse/core/units.h"
#include "atomiverse/cpu/gpw/gpw_solver.h"
#ifdef ATV_HAS_CUDA
#include "atomiverse/cuda/gpw/gpw_solver.cuh"
#endif
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

using namespace ATV;
namespace {
// Declared before any measurements: retain the archived GPW product gates,
// translating the force/stress gates to the public Frame units.
constexpr double kTwinEnergyHa = 5e-6;
constexpr double kTwinForceEvA = 5e-5 * units::kHartreePerBohrInEvPerAngstrom;
constexpr double kTwinStressEvA3 =
    5e-5 * units::kHartreePerBohr3InEvPerAngstrom3;
constexpr double kForceFdStepA = 1e-4;
constexpr double kForceFdToleranceEvA = 1e-3;
constexpr double kNonzeroForceEvA = 1e-4;
constexpr double kPublicationTolerance = 1e-10;
constexpr double kRepeatedEnergyHa = 1e-8;

void require(bool pass, const std::string& label) {
    if (!pass)
        throw std::runtime_error(label);
}
void close(double got, double expected, double tolerance,
           const std::string& label) {
    if (!std::isfinite(got) || !std::isfinite(expected)
        || std::abs(got - expected) > tolerance) {
        std::ostringstream message;
        message << std::setprecision(17) << label << ": got=" << got
                << " expected=" << expected << " tolerance=" << tolerance;
        throw std::runtime_error(message.str());
    }
}

cpu::Frame hydrogen() {
    cpu::Frame frame(2);
    constexpr double a = units::kBohrInAngstrom;
    frame.box = cpu::Box::orthorhombic(12 * a, 11 * a, 10 * a);
    // A compressed, tilted H-H bond in a rectangular periodic cell. All
    // Cartesian force lanes are checked; no equilibrium/symmetry assumption.
    constexpr double positions[2][3] = {{5.45, 5.15, 4.4}, {6.55, 5.65, 4.8}};
    for (int i = 0; i < 2; ++i) {
        frame.span(kAtom_type)[i] = 1;
        frame.span(kAtom_mass)[i] = 1.008;
        frame.span(kAtom_x)[i] = positions[i][0] * a;
        frame.span(kAtom_y)[i] = positions[i][1] * a;
        frame.span(kAtom_z)[i] = positions[i][2] * a;
    }
    return frame;
}
std::vector<cpu::gpw::Kind> kinds() {
    return {{molrs::Element::H,
             cpu::gpw::GaussianBasis(molrs::Element::H, "DZV-GTH-PADE"),
             cpu::gpw::GthPotential(molrs::Element::H, "GTH-PBE-q1")}};
}
template <class Method>
void configure(Method& method) {
    method.n_occ = 1;
    method.max_iter = 100;
    method.tol = 1e-10;
    method.diis_depth = 6;
    method.eps_default = 1e-10;
    method.nonperiodic = false;
    method.local_pp_on_grid = true;
    method.use_ot = false;
    method.ngrids = 1;
    method.stress_strain_h = .002;
    method.stress_axes = {true, true, true};
}
std::array<double, 6> forces(const cpu::Frame& frame) {
    std::array<double, 6> values{};
    int c = 0;
    for (auto key : {kAtom_fx, kAtom_fy, kAtom_fz})
        for (int i = 0; i < 2; ++i)
            values[c++] = frame.span(key)[i];
    return values;
}
void seed_publication(cpu::Frame& frame) {
    int c = 0;
    for (auto key : {kAtom_fx, kAtom_fy, kAtom_fz})
        for (int i = 0; i < 2; ++i)
            frame.span(key)[i] = 100.25 + c++;
    frame.meta.get(kEnergy) = 123.75;
    frame.meta.get(kEnergyElectronic) = 91.25;
    auto& stress = frame.meta.get(kStress);
    for (int c = 0; c < 6; ++c)
        stress[c] = 20.5 + c;
}
// Compare the public system inputs and result channels touched by the method.
void preserved(const cpu::Frame& got, const cpu::Frame& before,
               bool energy_may_change, const std::string& label) {
    require(got.nlocal() == before.nlocal(), label + " atom count");
    const auto columns = [&](auto key) {
        for (int i = 0; i < before.nlocal(); ++i)
            require(got.span(key)[i] == before.span(key)[i],
                    label + " atom column");
    };
    for (auto key : {kAtom_x, kAtom_y, kAtom_z})
        columns(key);
    columns(kAtom_type);
    columns(kAtom_mass);
    for (auto key : {kAtom_fx, kAtom_fy, kAtom_fz})
        columns(key);
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            require(got.box.cell()(i, j) == before.box.cell()(i, j),
                    label + " cell");
    for (int i = 0; i < 6; ++i)
        require(got.meta.get(kStress)[i] == before.meta.get(kStress)[i],
                label + " virial");
    require(got.meta.get(kEnergyElectronic)
                == before.meta.get(kEnergyElectronic),
            label + " separate electronic-energy metadata");
    if (!energy_may_change)
        require(got.meta.get(kEnergy) == before.meta.get(kEnergy),
                label + " energy");
}
struct Observation {
    double energy_ha = 0;
    std::array<double, 6> forces_eva{}, stress_eva3{};
    int iterations = 0;
};
template <class Method>
Observation observe(Method& method, const std::string& label) {
    auto frame = hydrogen();
    configure(method);
    method.build(frame);
    seed_publication(frame);
    const auto before_run = frame;
    const double run_energy = method.run(frame).e_total;
    require(method.last_result().converged, label + " run convergence");
    close(frame.meta.get(kEnergy), run_energy * units::kHartreeInEv,
          kPublicationTolerance, label + " run energy/Frame eV");
    preserved(frame, before_run, true, label + " run preservation");

    // A wrong forwarding call into the mutable overload cannot hide by
    // publishing the same physical energy: give const run a fresh sentinel.
    frame.meta.get(kEnergy) = 333.125;
    const auto before_const = frame;
    const double const_energy = method.run(std::as_const(frame)).e_total;
    require(method.last_result().converged, label + " const run convergence");
    close(const_energy, run_energy, kRepeatedEnergyHa,
          label + " const run energy");
    preserved(frame, before_const, false, label + " const run preservation");

    const auto force_result = method.forces(frame);
    require(force_result.converged, label + " forces convergence");
    Observation out;
    out.energy_ha = force_result.e_total;
    out.iterations = force_result.iters;
    out.forces_eva = forces(frame);
    close(frame.meta.get(kEnergy), out.energy_ha * units::kHartreeInEv,
          kPublicationTolerance, label + " force energy/Frame eV");
    close(out.energy_ha, run_energy, kRepeatedEnergyHa,
          label + " force reference energy");
    for (double force : out.forces_eva)
        require(std::isfinite(force), label + " finite force");
    require(std::abs(out.forces_eva[1]) > kNonzeroForceEvA,
            label + " nonzero atom-1 x force");

    out.stress_eva3 = method.stress(frame);
    require(method.last_result().converged, label + " stress convergence");
    close(method.last_result().e_total, out.energy_ha, kRepeatedEnergyHa,
          label + " stress reference energy");
    close(frame.meta.get(kEnergy),
          method.last_result().e_total * units::kHartreeInEv,
          kPublicationTolerance, label + " stress energy/Frame eV");
    for (int c = 0; c < 6; ++c) {
        close(forces(frame)[c], out.forces_eva[c], 0,
              label + " stress preserves forces");
        close(frame.meta.get(kStress)[c],
              out.stress_eva3[c] * frame.box.volume(), kPublicationTolerance,
              label + " virial publication/eV");
        if (c >= 3)
            close(out.stress_eva3[c], 0, 0,
                  label + " unsupported orthorhombic shear is zero");
    }
    return out;
}
double force_fd(cpu::gpw::GpwDft& method, double expected) {
    double energies[2]{};
    for (int sign = 0; sign < 2; ++sign) {
        auto frame = hydrogen();
        frame.span(kAtom_x)[1] += sign == 0 ? kForceFdStepA : -kForceFdStepA;
        const auto result = method.run(frame);
        require(result.converged, "CPU GPW displaced SCF convergence");
        close(frame.meta.get(kEnergy), result.e_total * units::kHartreeInEv,
              kPublicationTolerance, "CPU displaced energy publication");
        energies[sign] = frame.meta.get(kEnergy);
    }
    const double derivative =
        -(energies[0] - energies[1]) / (2 * kForceFdStepA);
    close(expected, derivative, kForceFdToleranceEvA,
          "CPU GPW independent ionic energy derivative/eV/A");
    return std::abs(expected - derivative);
}
void run_case(cpu::xc::XCType type, const char* name) {
    cpu::gpw::GpwDft reference{kinds(), 80, type};
    const auto expected = observe(reference, std::string(name) + " CPU");
    const double fd_error = force_fd(reference, expected.forces_eva[1]);
    std::printf("CPU GPW %s E_Ha=%.12g iterations=%d "
                "Fx1_eV_A=%.8g force_fd_error_eV_A=%.5g\n",
                name, expected.energy_ha, expected.iterations,
                expected.forces_eva[1], fd_error);
#ifdef ATV_HAS_CUDA
    cuda::gpw::GpwDft device{kinds(), 80, type};
    const auto got = observe(device, std::string(name) + " CUDA");
    close(got.energy_ha, expected.energy_ha, kTwinEnergyHa,
          "CPU/CUDA GPW energy/Ha");
    double force_error = 0, stress_error = 0;
    for (int c = 0; c < 6; ++c) {
        close(got.forces_eva[c], expected.forces_eva[c], kTwinForceEvA,
              "CPU/CUDA GPW force/eV/A");
        close(got.stress_eva3[c], expected.stress_eva3[c], kTwinStressEvA3,
              "CPU/CUDA GPW stress/eV/A^3");
        force_error = std::max(
            force_error, std::abs(got.forces_eva[c] - expected.forces_eva[c]));
        stress_error =
            std::max(stress_error,
                     std::abs(got.stress_eva3[c] - expected.stress_eva3[c]));
    }
    std::printf("PASS GPW %s CPU_CUDA dE_Ha=%.5g dF_eV_A=%.5g "
                "dStress_eV_A3=%.5g\n",
                name, std::abs(got.energy_ha - expected.energy_ha), force_error,
                stress_error);
#else
    std::printf("PASS GPW %s CPU; CUDA twin checks skipped (CPU-only build)\n",
                name);
#endif
    std::fflush(stdout);
}
} // namespace
int main() {
    try {
        constexpr cpu::xc::XCType types[] = {
            cpu::xc::XCType::PBE, cpu::xc::XCType::PBEsol,
            cpu::xc::XCType::RevPBE, cpu::xc::XCType::RPBE};
        constexpr const char* names[] = {"PBE", "PBEsol", "revPBE", "RPBE"};
        for (int i = 0; i < 4; ++i)
            run_case(types[i], names[i]);
    } catch (const std::exception& error) {
        std::fprintf(stderr, "FAIL GPW %s\n", error.what());
        return 1;
    }
}
