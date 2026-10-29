// Independent opt-in Gamma UKS PBE-family Driver envelope; never CTest.
#include "../pbe-periodic-oracles/uks_fixture.h"
#include "cuda_observers.h"
#include "fixtures.h"
#include <algorithm>
#include <array>
#include <cstdio>
#include <memory>
#include <vector>
using namespace ATV;
namespace {
namespace old = ATV::diagnostics::gpw_driver;
namespace fixed = pbe_periodic_uks_fixture;
using namespace ATV::diagnostics::gpw_driver::cuda_observers;
using Six = std::array<double, 6>;
// Preserve the existing per-child science gates and nested publication gates.
constexpr double energy_ha = 5e-6, nested_energy_ev = 2e-7,
                 nested_force_eva = 2e-7, nested_virial_ev = 2e-6;
constexpr double force_eva = 5e-5 * units::kHartreePerBohrInEvPerAngstrom;
constexpr double stress_eva3 = 5e-5 * units::kHartreePerBohr3InEvPerAngstrom3;
struct Case {
    const char* family;
    cpu::xc::XCType xc;
    bool empty;
};
cpu::Frame geometry(const Case& c, bool shifted) {
    if (c.empty)
        return old::geometry(shifted);
    auto frame = fixed::h3();
    if (shifted)
        frame.span(kAtom_x)[2] += .017; // Angstrom, second geometry only.
    old::mark(frame);
    return frame;
}
template <class Solver>
std::unique_ptr<Solver> method(const Case& c, int child) {
    auto value = std::make_unique<Solver>(
        c.empty ? old::kinds() : fixed::kinds(), old::cutoff[child], c.xc);
    if (c.empty)
        old::configure(*value, old::cases[1]);
    else
        fixed::configure(*value);
    // Independent solves need explicit stationarity for the unchanged force gate.
    value->residual_tol = 1e-9;
    auto frame = geometry(c, false);
    value->build(frame);
    return value;
}
template <class Result>
void spin(const Result& result, const Case& c) {
    require(result.converged && result.e_entropy == 0
                && result.S.dimension() == 2
                && result.P.shape() == result.S.shape()
                && result.P_beta.shape() == result.S.shape(),
            "converged integer UKS requires accepted Pa/Pb/S");
    double na = 0, nb = 0;
    for (std::size_t i = 0; i < result.S.shape()[0]; ++i)
        for (std::size_t j = 0; j < result.S.shape()[1]; ++j) {
            na += result.P(i, j) * result.S(j, i);
            nb += result.P_beta(i, j) * result.S(j, i);
        }
    close(na, 2, 1e-8, "accepted Tr(Palpha S)");
    close(nb, c.empty ? 0 : 1, 1e-8, "accepted Tr(Pbeta S)");
}
struct Value {
    double energy = 0;
    std::vector<double> force;
    Six virial{};
};
void add(Value& sum, const Value& value) {
    sum.energy += value.energy;
    if (sum.force.empty())
        sum.force.assign(value.force.size(), 0);
    for (std::size_t i = 0; i < sum.force.size(); ++i)
        sum.force[i] += value.force[i];
    for (int k = 0; k < 6; ++k)
        sum.virial[k] += value.virial[k];
}
void unchanged(const cpu::Frame& frame, const cpu::Frame& before,
               const char* why) {
    require(old::forces(frame) == old::forces(before)
                && frame.meta.get(kStress) == before.meta.get(kStress)
                && frame.meta.get(kEnergy) == before.meta.get(kEnergy),
            why);
}
struct Reference {
    Value cpu, cuda;
};
Reference reference(const Case& c, bool shifted) {
    Reference reference;
    for (int child = 0; child < 2; ++child) {
        auto host = method<cpu::gpw::GpwDft>(c, child);
        auto twin = geometry(c, shifted);
        const auto accepted = host->forces(twin);
        spin(accepted, c);
        Value expected{twin.meta.get(kEnergy), old::forces(twin), {}};
        const auto cpu_sigma = host->stress(twin);
        spin(host->last_result(), c);
        expected.virial = twin.meta.get(kStress);
        auto gpu = method<Method>(c, child);
        Science science;
        Measurement measurement(science, *gpu);
        ScientificGuard guard(science);
        gpu->on(gpw::Phase::check).add_ref(measurement);
        gpu->add_criterion_ref(guard);
        auto frame = geometry(c, shifted);
        const auto before = frame;
        gpu->run(frame);
        spin(gpu->last_result(), c);
        lifecycle(science, 1, gpu->last_result().iters);
        close(frame.meta.get(kEnergy), expected.energy,
              energy_ha * units::kHartreeInEv, "public run CPU energy");
        require(old::forces(frame) == old::forces(before)
                    && frame.meta.get(kStress) == before.meta.get(kStress),
                "run preserves derivatives");
        gpu->forces(frame);
        spin(gpu->last_result(), c);
        lifecycle(science, 2, gpu->last_result().iters);
        Value actual{frame.meta.get(kEnergy), old::forces(frame), {}};
        close(actual.energy, expected.energy, energy_ha * units::kHartreeInEv,
              "public forces CPU energy");
        for (std::size_t i = 0; i < actual.force.size(); ++i)
            close(actual.force[i], expected.force[i], force_eva,
                  "public full UKS force");
        const auto sigma = gpu->stress(frame);
        spin(gpu->last_result(), c);
        lifecycle(science, 3, gpu->last_result().iters);
        actual.virial = frame.meta.get(kStress);
        require(old::forces(frame) == actual.force,
                "public stress preserves force");
        for (int k = 0; k < 6; ++k) {
            close(sigma[k], cpu_sigma[k], stress_eva3, "public UKS stress");
            close(actual.virial[k], sigma[k] * frame.box.volume(), 1e-10,
                  "Frame virial units");
        }
        add(reference.cpu, expected);
        add(reference.cuda, actual);
    }
    return reference;
}
// Compare inside the child SCF, where its LoopServices intentionally differ
// from the parent's. The execution owner, frame and borrowed stream must match.
class BorrowedStream final : public cuda::Component {
    ParentProbe& parent_;

  public:
    explicit BorrowedStream(ParentProbe& parent) : parent_(parent) {}
    std::string name() const override { return "uks_borrowed_stream"; }
    void launch(const cuda::LaunchContext& c) override {
        require(parent_.stream == c.runtime.stream && parent_.frame == &c.frame
                    && parent_.exec.backend == c.runtime.exec.backend,
                "both UKS children must use the exact parent execution owner");
    }
};
void run(const Case& c) {
    const std::array<Reference, 2> refs = {reference(c, false),
                                           reference(c, true)};
    require(std::abs(refs[0].cpu.energy - refs[1].cpu.energy) > 1e-6,
            "geometry refresh changes energy");
    for (int request = 0; request < 3; ++request) {
        const bool forces = request > 0, stress = request == 2, inject = stress;
        auto first = method<Method>(c, 0), second = method<Method>(c, 1);
        Science a, b;
        Measurement ma(a, *first), mb(b, *second);
        ScientificGuard ga(a), gb(b);
        first->on(gpw::Phase::check).add_ref(ma);
        first->add_criterion_ref(ga);
        second->on(gpw::Phase::check).add_ref(mb);
        second->add_criterion_ref(gb);
        ParentProbe probe;
        Capture capture(probe);
        BorrowedStream stream_a(probe), stream_b(probe);
        first->on(gpw::Phase::check).add_ref(stream_a, 10);
        second->on(gpw::Phase::check).add_ref(stream_b, 10);
        AfterChild after_a(probe, *first, a, 0), after_b(probe, *second, b, 1);
        cuda::Evaluate parent(forces, stress);
        parent.on(evaluate::Phase::evaluate).add_ref(capture, -10);
        parent.on(evaluate::Phase::evaluate).add_ref(*first, 0);
        parent.on(evaluate::Phase::evaluate).add_ref(after_a, 10);
        parent.on(evaluate::Phase::evaluate).add_ref(*second, 20);
        parent.on(evaluate::Phase::evaluate).add_ref(after_b, 30);
        parent.build();
        if (inject) {
            auto frame = geometry(c, false);
            const auto before = frame;
            b.inject = true;
            bool failed = false;
            try {
                parent.run(frame);
            } catch (const std::runtime_error& e) {
                failed = std::string(e.what())
                         == "intentional GPW SCF stage failure";
            }
            require(failed && b.injected && probe.teardowns == 1,
                    "second child failure restores parent");
            unchanged(
                frame, before,
                "failed parent preserves all caller outputs after first child");
            lifecycle(a, 1, first->last_result().iters);
            require(!b.stage_live && !b.criterion_live && b.builds == 1
                        && b.destroys == 1 && b.criterion_builds == 1
                        && b.criterion_destroys == 1 && b.measurements == 1
                        && b.polls == 0,
                    "failed second child fully tears down");
        }
        for (int shifted = 0; shifted < 2; ++shifted) {
            auto frame = geometry(c, shifted != 0);
            const auto before = frame;
            parent.run(frame);
            spin(first->last_result(), c);
            spin(second->last_result(), c);
            const auto& expected = refs[shifted];
            close(frame.meta.get(kEnergy), expected.cuda.energy,
                  nested_energy_ev, "nested standalone energy sum");
            close(frame.meta.get(kEnergy), expected.cpu.energy,
                  2 * energy_ha * units::kHartreeInEv, "nested CPU energy sum");
            const auto f = old::forces(frame);
            if (!forces)
                require(f == old::forces(before),
                        "energy request preserves forces");
            else
                for (std::size_t i = 0; i < f.size(); ++i) {
                    close(f[i], expected.cuda.force[i], nested_force_eva,
                          "nested standalone force sum");
                    close(f[i], expected.cpu.force[i], 2 * force_eva,
                          "nested CPU force sum");
                }
            if (!stress)
                require(frame.meta.get(kStress) == before.meta.get(kStress),
                        "unrequested virial preserved");
            else
                for (int k = 0; k < 6; ++k) {
                    close(frame.meta.get(kStress)[k], expected.cuda.virial[k],
                          nested_virial_ev, "nested standalone virial sum");
                    close(frame.meta.get(kStress)[k], expected.cpu.virial[k],
                          2 * stress_eva3 * frame.box.volume(),
                          "nested CPU virial sum");
                }
            lifecycle(a, shifted + 1 + inject, first->last_result().iters);
            lifecycle(b, shifted + 1 + inject, second->last_result().iters);
        }
        require(probe.moves == 4 + inject && probe.teardowns == 2 + inject,
                "both child visits and parent retries observed");
    }
    std::printf("PASS GPW_CUDA_UKS_PBE_DRIVER %s-%s modes=3 geometries=2 "
                "children=2 cutoffs=80,100 failure=second-child-retry\n",
                c.family, c.empty ? "h2-emptybeta" : "h3");
    std::fflush(stdout);
}
} // namespace
int main(int argc, char** argv) {
    std::string selected;
    if (argc == 3 && std::string(argv[1]) == "--case")
        selected = argv[2];
    else if (argc != 1)
        return 2;
    bool matched = false;
    constexpr const char* names[] = {"PBE", "PBEsol", "revPBE", "RPBE"};
    constexpr cpu::xc::XCType types[] = {
        cpu::xc::XCType::PBE, cpu::xc::XCType::PBEsol, cpu::xc::XCType::RevPBE,
        cpu::xc::XCType::RPBE};
    for (int family = 0; family < 4; ++family)
        for (bool empty : {false, true}) {
            const std::string id =
                std::string(names[family]) + (empty ? "-h2-emptybeta" : "-h3");
            if (!selected.empty() && selected != id)
                continue;
            matched = true;
            try {
                run({names[family], types[family], empty});
            } catch (const std::exception& e) {
                std::fprintf(stderr, "FAIL GPW_CUDA_UKS_PBE_DRIVER %s-%s %s\n",
                             names[family], empty ? "h2-emptybeta" : "h3",
                             e.what());
                return 1;
            }
        }
    return matched ? 0 : 2;
}
