// Public PW methods: CPU energy/force consistency and optional CPU/CUDA twins.
#include "atomiverse/core/units.h"
#include "atomiverse/cpu/component/criterion.h"
#include "atomiverse/cpu/planewave/pw_solver.h"
#ifdef ATV_HAS_CUDA
#include "atomiverse/cuda/planewave/pw_solver.cuh"
#endif
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

using namespace ATV;
namespace {
void require(bool pass, const std::string& message) {
    if (!pass)
        throw std::runtime_error(message);
}
void close(double got, double expected, double tolerance,
           const std::string& label) {
    require(std::isfinite(got) && std::isfinite(expected)
                && std::abs(got - expected) <= tolerance,
            label + ": got " + std::to_string(got) + " expected "
                + std::to_string(expected));
}
cpu::Frame hydrogen(bool spin) {
    cpu::Frame frame(spin ? 3 : 2);
    Mat3 cell;
    cell(0, 0) = 4.2;
    cell(0, 1) = .3;
    cell(0, 2) = .2;
    cell(1, 0) = 0;
    cell(1, 1) = 4;
    cell(1, 2) = .1;
    cell(2, 0) = cell(2, 1) = 0;
    cell(2, 2) = 3.8;
    frame.box.set_cell(cell);
    constexpr double positions[3][3] = {
        {1.2, 1.1, 1.0}, {2.1, 1.6, 1.3}, {1.6, 2.25, 1.8}};
    for (int i = 0; i < frame.nlocal(); ++i) {
        frame.span(kAtom_type)[i] = 1;
        frame.span(kAtom_mass)[i] = 1.008;
        frame.span(kAtom_x)[i] = positions[i][0];
        frame.span(kAtom_y)[i] = positions[i][1];
        frame.span(kAtom_z)[i] = positions[i][2];
    }
    return frame;
}
template <class Method>
void configure(Method& method, bool spin, electronic::Smearing smear) {
    method.spin_polarized = spin;
    method.multiplicity = spin ? 2 : 1;
    method.extra_bands = 3;
    method.tol = 1e-9;
    method.max_iter = 400;
    method.mix_beta = .3;
    method.smearing = smear;
    method.degauss = smear == electronic::Smearing::fermi_dirac ? .02 : .04;
}
std::vector<double> forces(const cpu::Frame& frame) {
    std::vector<double> result;
    for (auto key : {kAtom_fx, kAtom_fy, kAtom_fz})
        for (int i = 0; i < frame.nlocal(); ++i)
            result.push_back(frame.span(key)[i]);
    return result;
}
// Attach after the first solve: later SCFs must compile the updated mount
// table and build the criterion before it observes actual iteration results.
class CpuPhysicalProbe final : public cpu::Criterion {
    const cpu::pw::PW& method_;
    bool live_ = false;
    int steps_ = 0;

  public:
    int builds = 0, destroys = 0;
    explicit CpuPhysicalProbe(const cpu::pw::PW& method) : method_(method) {}
    std::string name() const override { return "pw_physical_result_probe"; }
    std::vector<DefaultBinding> default_bindings() const override {
        return {make_binding(pw::Phase::check, 0)};
    }
    void build(const cpu::BuildContext&) override {
        require(!live_, "CPU criterion must not be built twice");
        live_ = true;
        steps_ = 0;
        ++builds;
    }
    void launch(const cpu::LaunchContext& context) override {
        require(live_ && method_.last_result().iterations == context.step + 1,
                "CPU attached stage must observe its current SCF iteration");
        require(std::isfinite(method_.last_result().energy_ha)
                    && std::abs(method_.last_result().energy_ha) > 1e-3,
                "CPU attached stage must observe a physical energy");
        ++steps_;
    }
    Verdict verdict(const cpu::LoopState& state) override {
        require(live_ && steps_ == state.iteration + 1,
                "CPU criterion must be built and measured before polling");
        return Verdict::Continue;
    }
    void destroy(const cpu::DestroyContext&) override {
        require(live_ && steps_ > 0,
                "CPU criterion teardown must follow measured SCF stages");
        live_ = false;
        ++destroys;
    }
};
void run_case(cpu::xc::XCType type, const char* name, bool spin,
              electronic::Smearing smear) {
    // Odd restricted H3 makes the thermal path fill genuinely fractional
    // occupations; the fixed restricted fixture remains closed-shell H2.
    auto frame = hydrogen(spin || smear != electronic::Smearing::none);
    auto pp = std::make_shared<cpu::pw::GthPseudopotential>(molrs::Element::H,
                                                            "GTH-PBE-q1");
    cpu::pw::PW reference{pp, 6.0, 1, type};
    configure(reference, spin, smear);
    const auto result = reference.forces(frame);
    require(result.converged, std::string(name) + " CPU convergence");
    if (smear != electronic::Smearing::none) {
        close(result.n_elec_occ, frame.nlocal(), 1e-8,
              "CPU thermal electron count");
        require(std::abs(result.e_entropy) > 1e-8,
                "CPU thermal fixture must exercise a free-energy correction");
    }
    const auto expected_forces = forces(frame);
    CpuPhysicalProbe physical_probe(reference);
    reference.add_criterion_ref(physical_probe);
    const auto expected_stress = reference.stress(frame);
    require(physical_probe.builds == 1 && physical_probe.destroys == 1,
            "CPU stress must execute the newly attached stage lifecycle");
    require(reference.last_result().converged, "CPU stress convergence");
    close(reference.last_result().energy_ha, result.energy_ha, 1e-8,
          "CPU energy reuse");
    for (int i = 0; i < 6; ++i)
        close(frame.meta.get(kStress)[i],
              expected_stress[i] * frame.box.volume(), 1e-10,
              "CPU virial publication");

    // Change an ionic coordinate only: the PW basis and cell remain fixed.
    // This is an independent full-SCF energy derivative, not a force helper.
    const double position = frame.span(kAtom_x)[1];
    constexpr double step = 1e-4;
    double energies[2];
    for (int sign = 0; sign < 2; ++sign) {
        frame.span(kAtom_x)[1] = position + (sign == 0 ? step : -step);
        const auto sample = reference.run(frame);
        require(sample.converged, "CPU force FD convergence");
        energies[sign] = frame.meta.get(kEnergy);
    }
    frame.span(kAtom_x)[1] = position;
    require(physical_probe.builds == 3 && physical_probe.destroys == 3,
            "CPU repeated solves must rebuild and release attached criteria");
    const double force_fd = -(energies[0] - energies[1]) / (2 * step);
    require(std::abs(expected_forces[1]) > 1e-3,
            "ionic force probe must exercise a nonzero component");
    close(expected_forces[1], force_fd, 2e-4, "CPU ionic force FD/eV/A");
#ifdef ATV_HAS_CUDA
    double energy_error = 0, force_error = 0, stress_error = 0;
    auto source = std::make_unique<cuda::pw::PW>(pp, 6.0, 1, type);
    configure(*source, spin, smear);
    int marker = 0;
    for (auto key : {kAtom_fx, kAtom_fy, kAtom_fz})
        for (int i = 0; i < frame.nlocal(); ++i)
            frame.span(key)[i] = 10.25 + marker++;
    frame.meta.get(kStress) = {1, 2, 3, 4, 5, 6};
    const auto markers = forces(frame);
    const auto virial_marker = frame.meta.get(kStress);
    const auto before_move = source->run(frame);
    require(before_move.converged, "CUDA pre-move convergence");
    require(forces(frame) == markers
                && frame.meta.get(kStress) == virial_marker,
            "CUDA run preserves forces and virial");
    cuda::pw::PW device(std::move(*source));
    source.reset();
    close(device.last_result().energy_ha, before_move.energy_ha, 0,
          "move retains the last result");
    frame.meta.get(kEnergy) = 31415.125;
    const auto const_result = device.run(std::as_const(frame));
    require(const_result.converged, "CUDA post-move const run convergence");
    require(frame.meta.get(kEnergy) == 31415.125 && forces(frame) == markers
                && frame.meta.get(kStress) == virial_marker,
            "CUDA const run preserves Frame outputs");
    const auto got = device.forces(frame);
    require(got.converged, std::string(name) + " CUDA convergence");
    energy_error = std::abs(got.energy_ha - result.energy_ha);
    close(got.energy_ha, result.energy_ha, 1e-7, "CPU/CUDA energy/Ha");
    const auto actual_forces = forces(frame);
    for (std::size_t i = 0; i < expected_forces.size(); ++i) {
        force_error = std::max(force_error,
                               std::abs(actual_forces[i] - expected_forces[i]));
        close(actual_forces[i], expected_forces[i], 2e-5,
              "CPU/CUDA force/eV/A");
    }
    const auto actual_stress = device.stress(frame);
    require(device.last_result().converged, "CUDA stress convergence");
    close(device.last_result().energy_ha, got.energy_ha, 1e-8,
          "CUDA stress reference energy");
    for (int i = 0; i < 6; ++i) {
        stress_error = std::max(
            stress_error, std::abs(actual_stress[i] - expected_stress[i]));
        close(actual_stress[i], expected_stress[i], 5e-4,
              "CPU/CUDA stress/eV/A^3");
        close(frame.meta.get(kStress)[i], actual_stress[i] * frame.box.volume(),
              1e-10, "CUDA virial publication");
    }
    if (smear != electronic::Smearing::none) {
        close(got.n_elec_occ, frame.nlocal(), 1e-8, "thermal electron count");
        require(std::abs(got.e_entropy) > 1e-8,
                "thermal fixture must exercise a free-energy correction");
        close(got.e_entropy, result.e_entropy, 1e-7,
              "CPU/CUDA free-energy correction/Ry");
    }
    // Exhaustion is a fresh, non-converged solve, including derivative
    // publication. It must never retain an earlier successful verdict.
    device.max_iter = 1;
    const auto limited = device.run(frame);
    require(!limited.converged && limited.iterations == 1,
            "one-iteration run must report exhaustion");
    const auto limited_force = device.forces(frame);
    require(!limited_force.converged && limited_force.iterations == 1,
            "force result must retain its own exhausted verdict");
    close(limited_force.energy_ha, limited.energy_ha, 1e-8,
          "exhausted force reference energy");
    device.stress(frame);
    require(!device.last_result().converged
                && device.last_result().iterations == 1,
            "stress result must retain its own exhausted verdict");
    close(device.last_result().energy_ha, limited.energy_ha, 1e-8,
          "exhausted stress reference energy");
    device.max_iter = 400;
    auto shifted = frame;
    shifted.span(kAtom_x)[1] += .03;
    const auto reused = device.run(shifted);
    cuda::pw::PW fresh{pp, 6.0, 1, type};
    configure(fresh, spin, smear);
    const auto fresh_result = fresh.run(shifted);
    require(reused.converged && fresh_result.converged,
            "changed-geometry reuse convergence");
    close(reused.energy_ha, fresh_result.energy_ha, 1e-8,
          "changed-geometry reuse agrees with a fresh method");
    std::printf("PASS %s %s E_Ha=%.12g force_fd_error=%.5g "
                "twin_energy_error=%.5g twin_force_error=%.5g "
                "twin_stress_error=%.5g\n",
                name, spin ? "UKS" : "RKS", result.energy_ha,
                std::abs(expected_forces[1] - force_fd), energy_error,
                force_error, stress_error);
#else
    std::printf("PASS %s %s E_Ha=%.12g force_fd_error=%.5g twin=skipped\n",
                name, spin ? "UKS" : "RKS", result.energy_ha,
                std::abs(expected_forces[1] - force_fd));
#endif
    std::fflush(stdout);
}
} // namespace
int main(int argc, char** argv) {
    electronic::Smearing smear = electronic::Smearing::none;
    if (argc != 1) {
        if (argc != 3 || std::strcmp(argv[1], "--smearing") != 0) {
            std::fprintf(stderr,
                         "usage: pbe-family [--smearing fd|gaussian|mp1]\n");
            return 2;
        }
        if (std::strcmp(argv[2], "fd") == 0)
            smear = electronic::Smearing::fermi_dirac;
        else if (std::strcmp(argv[2], "gaussian") == 0)
            smear = electronic::Smearing::gaussian;
        else if (std::strcmp(argv[2], "mp1") == 0)
            smear = electronic::Smearing::methfessel_paxton;
        else
            return 2;
    }
    constexpr cpu::xc::XCType types[] = {
        cpu::xc::XCType::PBE, cpu::xc::XCType::PBEsol, cpu::xc::XCType::RevPBE,
        cpu::xc::XCType::RPBE};
    constexpr const char* names[] = {"PBE", "PBEsol", "revPBE", "RPBE"};
    int failed = 0;
    for (int i = 0; i < 4; ++i)
        for (bool spin : {false, true})
            try {
                run_case(types[i], names[i], spin, smear);
            } catch (const std::exception& error) {
                ++failed;
                std::fprintf(stderr, "FAIL %s %s: %s\n", names[i],
                             spin ? "UKS" : "RKS", error.what());
            }
    return failed ? 1 : 0;
}
