/// @brief Complete CUDA AM1/RM1 energy/charge/analytic-force solves against executed MOPAC pins.
/// Sources/tolerances: existing am1-water-mopac-energy and
/// tools/tb_oracle/inputs/mopac/live-23.2.5/README.md (OpenMOPAC 23.2.5).
#include "am1-water-mopac-energy.h"
#include "atomiverse/core/frame_schema.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/semiempirical/am1_solver.h"
#include "atomiverse/cpu/semiempirical/rm1_solver.h"
#include "atomiverse/cuda/device/error.cuh"
#include "atomiverse/cuda/evaluate/evaluate.cuh"
#include "atomiverse/cuda/frame/precision.cuh"
#include "atomiverse/cuda/semiempirical/am1_solver.cuh"
#include "atomiverse/cuda/semiempirical/rm1_solver.cuh"
#include "harness/registry.h"
#ifdef ATV_HAS_MD
#include "atomiverse/cpu/dynamics/md.h"
#include "atomiverse/cpu/dynamics/velocity_verlet.h"
#include "atomiverse/cuda/dynamics/md.cuh"
#include "atomiverse/cuda/dynamics/velocity_verlet.cuh"
#endif
#ifdef ATV_HAS_MINIMIZER
#include "atomiverse/cuda/minimize/lbfgs.cuh"
#include "atomiverse/cuda/minimize/minimize.cuh"
#include "atomiverse/cuda/minimize/sd.cuh"
#endif

#include <array>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace atv_reg_semiempirical_cuda_native_energy {
using namespace ATV;
namespace {
using namespace atv_reg_semiempirical_am1_water_mopac_energy;
cpu::Frame molecule(const ModernMopacPin& pin) {
    cpu::Frame frame(pin.count);
    for (int i = 0; i < pin.count; ++i) {
        frame.span(kAtom_type)[i] = pin.species[i];
        frame.span(kAtom_x)[i] = pin.xyz[i][0];
        frame.span(kAtom_y)[i] = pin.xyz[i][1];
        frame.span(kAtom_z)[i] = pin.xyz[i][2];
        frame.span(kAtom_fx)[i] = .125;
        frame.span(kAtom_fy)[i] = -.25;
        frame.span(kAtom_fz)[i] = .5;
    }
    return frame;
}
void near(const char* label, double got, double want, double tol) {
    if (!std::isfinite(got) || std::abs(got - want) > tol) {
        std::fprintf(stderr, "FAIL %s got=%.16g want=%.16g tol=%.3g\n", label,
                     got, want, tol);
        throw std::runtime_error(label);
    }
}
void require(bool condition, const char* label) {
    if (!condition) {
        std::fprintf(stderr, "FAIL %s got=0 want=1 tol=0\n", label);
        throw std::runtime_error(label);
    }
}
template <class F>
void rejects(F&& fn, const char* label, const char* message = nullptr) {
    bool rejected = false;
    try {
        fn();
    } catch (const std::exception& error) {
        if (message)
            require(std::strcmp(error.what(), message) == 0,
                    "original failure message preserved");
        rejected = true;
    }
    require(rejected, label);
}
void forces_preserved(const cpu::Frame& frame) {
    for (int i = 0; i < frame.nlocal(); ++i) {
        near("force x preserved", frame.span(kAtom_fx)[i], .125, 0);
        near("force y preserved", frame.span(kAtom_fy)[i], -.25, 0);
        near("force z preserved", frame.span(kAtom_fz)[i], .5, 0);
    }
}
constexpr double kExtraForce[3] = {.125, -.25, .5};
__global__ void extra_contribution(cuda::Frame frame, int n) {
    const int i = threadIdx.x;
    if (i == 0)
        frame.scalars->energy += 1.25;
    if (i < n) {
        cuda::add_force(frame.atoms.fx, i, .125, frame.force_fault);
        cuda::add_force(frame.atoms.fy, i, -.25, frame.force_fault);
        cuda::add_force(frame.atoms.fz, i, .5, frame.force_fault);
    }
}
/// Uses the default parent's energy/force request; only execution history flags
/// are varied. Contributions precede the method to detect accidental overwrite.
class ContributionProbe final : public cuda::Component {
  public:
    bool initial = false, trial = false;
    std::string name() const override { return "nddo_contribution_probe"; }
    bool provides_energy() const override { return true; }
    bool provides_force() const override { return true; }
    void launch(const cuda::LaunchContext& ctx) override {
        ctx.runtime.exec.initializing_forces = initial;
        ctx.runtime.exec.trial_evaluation = trial;
        extra_contribution<<<1, 32, 0,
                             static_cast<cudaStream_t>(ctx.runtime.stream)>>>(
            ctx.frame, ctx.frame.n_atoms());
        ATV_CUDA_CHECK(cudaGetLastError());
    }
};
void check_force_frame(const cpu::Frame& frame,
                       const std::vector<double>& expected, bool extra = false,
                       double tolerance = 1e-8) {
    require(expected.size() == 3 * static_cast<std::size_t>(frame.nlocal()),
            "full force vector present");
    const auto fx = frame.span(kAtom_fx), fy = frame.span(kAtom_fy),
               fz = frame.span(kAtom_fz);
    for (int i = 0; i < frame.nlocal(); ++i) {
        const double actual[3] = {fx[i], fy[i], fz[i]};
        for (int d = 0; d < 3; ++d)
            near("Frame force", actual[d],
                 expected[3 * i + d] + (extra ? kExtraForce[d] : 0.0),
                 tolerance);
    }
}
class SolveFailure final : public cuda::Component {
  public:
    bool fail = false;
    std::string name() const override { return "nddo_solve_failure_probe"; }
    void launch(const cuda::LaunchContext&) override {
        if (fail) {
            fail = false;
            throw std::runtime_error("injected force solve failure");
        }
    }
};
class LifecycleProbe final : public cuda::Component {
  public:
    int builds = 0, launches = 0, destroys = 0;
    bool ready = false;
    std::string name() const override { return "nddo_lifecycle_probe"; }
    void build(const cuda::BuildContext&) override {
        ready = true;
        ++builds;
    }
    void launch(const cuda::LaunchContext&) override {
        if (!ready)
            throw std::runtime_error("NDDO component was not built");
        ++launches;
    }
    void destroy(const cuda::DestroyContext&) override {
        ready = false;
        ++destroys;
    }
};
template <class Gpu>
class MountedMoveProbe final : public cuda::Component {
  public:
    explicit MountedMoveProbe(Gpu& method) : method_(method) {}
    std::string name() const override { return "mounted_move_probe"; }
    void launch(const cuda::LaunchContext&) override {
        rejects([&] { Gpu moved(std::move(method_)); },
                "mounted move rejected");
    }

  private:
    Gpu& method_;
};
class BuildFailure final : public cuda::Component {
  public:
    bool fail = true;
    int builds = 0, destroys = 0;
    std::string name() const override { return "build_failure_probe"; }
    void build(const cuda::BuildContext&) override {
        ++builds;
        if (fail) {
            fail = false;
            throw std::runtime_error("injected build failure");
        }
    }
    void launch(const cuda::LaunchContext&) override {}
    void destroy(const cuda::DestroyContext&) override { ++destroys; }
};
class DestroyFailure final : public cuda::Component {
  public:
    bool fail = true, ready = false;
    int destroys = 0;
    std::string name() const override { return "destroy_failure_probe"; }
    void build(const cuda::BuildContext&) override { ready = true; }
    void launch(const cuda::LaunchContext&) override {
        require(ready, "destroy retry stage built");
    }
    void destroy(const cuda::DestroyContext&) override {
        ready = false;
        ++destroys;
        if (fail) {
            fail = false;
            throw std::runtime_error(
                "injected completed-stage destruction failure");
        }
    }
};
template <class Gpu>
void nested(Gpu& method, cpu::Frame& frame, bool initial = false,
            bool trial = false, int budget = -1) {
    ContributionProbe request;
    request.initial = initial;
    request.trial = trial;
    cuda::Evaluate outer;
    outer.on(evaluate::Phase::prepare).add_ref(request);
    outer.on(evaluate::Phase::evaluate).add_ref(method, 0, budget);
    outer.build();
    outer.run(frame);
}

#if defined(ATV_HAS_MD) || defined(ATV_HAS_MINIMIZER)
cpu::Frame dynamic_molecule(const ModernMopacPin& pin) {
    auto frame = molecule(pin);
    for (int i = 0; i < pin.count; ++i) {
        frame.span(kAtom_tag)[i] = i + 1;
        frame.span(kAtom_mass)[i] =
            pin.species[i] == 1
                ? 1.008
                : (pin.species[i] == 8
                       ? 15.999
                       : (pin.species[i] == 7 ? 14.007 : 12.011));
        frame.span(kAtom_vx)[i] = 1e-4 * (i - 1);
        frame.span(kAtom_vy)[i] = -2e-4 * (i % 2);
        frame.span(kAtom_vz)[i] = 1e-4 * (i % 3);
    }
    return frame;
}
#endif
void check_observables(const cpu::Frame& frame,
                       const ATV::semiempirical::NDDOResult& result) {
    near("composition energy", frame.meta.get(kEnergy), result.total_energy_ev,
         1e-8);
    check_force_frame(frame, result.forces_ev_a, false, 1e-7);
    for (int i = 0; i < frame.nlocal(); ++i)
        near("composition charge", frame.span(kAtom_charge)[i],
             result.mulliken_charges[i], 1e-8);
}
#ifdef ATV_HAS_MD
void compare_trajectory(const cpu::Frame& a, const cpu::Frame& b) {
    near("four-step potential energy", a.meta.get(kEnergy), b.meta.get(kEnergy),
         1e-8);
    const auto ap =
        std::array{a.span(kAtom_x), a.span(kAtom_y), a.span(kAtom_z)};
    const auto bp =
        std::array{b.span(kAtom_x), b.span(kAtom_y), b.span(kAtom_z)};
    const auto av =
        std::array{a.span(kAtom_vx), a.span(kAtom_vy), a.span(kAtom_vz)};
    const auto bv =
        std::array{b.span(kAtom_vx), b.span(kAtom_vy), b.span(kAtom_vz)};
    const auto af =
        std::array{a.span(kAtom_fx), a.span(kAtom_fy), a.span(kAtom_fz)};
    const auto bf =
        std::array{b.span(kAtom_fx), b.span(kAtom_fy), b.span(kAtom_fz)};
    for (int i = 0; i < a.nlocal(); ++i) {
        near("four-step charge", a.span(kAtom_charge)[i],
             b.span(kAtom_charge)[i], 1e-8);
        for (int d = 0; d < 3; ++d) {
            near("four-step coordinate", ap[d][i], bp[d][i], 1e-10);
            near("four-step velocity", av[d][i], bv[d][i], 1e-10);
            near("four-step force", af[d][i], bf[d][i], 1e-7);
        }
    }
}
template <class Cpu, class Gpu>
void short_md(const ModernMopacPin& pin) {
    constexpr double dt =
        .01; // fs, four deterministic steps, no long-trajectory claim.
    auto split_frame = dynamic_molecule(pin),
         whole_frame = dynamic_molecule(pin), cpu_frame = dynamic_molecule(pin);
    Gpu split_method{0, 1, 1000, 1e-10, 1e-12},
        whole_method{0, 1, 1000, 1e-10, 1e-12},
        direct{0, 1, 1000, 1e-10, 1e-12};
    Cpu cpu_method{0, 1, 1000, 1e-10, 1e-12};
    auto reference_frame = dynamic_molecule(pin);
    const auto reference = direct.forces(reference_frame);
    cuda::md::MD split, whole;
    cpu::md::MD cpu_md;
    split.set_integrator(cuda::md::VelocityVerlet{dt});
    whole.set_integrator(cuda::md::VelocityVerlet{dt});
    cpu_md.set_integrator(cpu::md::VelocityVerlet{dt});
    split.on(md::Phase::force).add_ref(split_method);
    whole.on(md::Phase::force).add_ref(whole_method);
    cpu_md.on(md::Phase::force).add_ref(cpu_method);
    split.build();
    whole.build();
    cpu_md.build();
    require(split.run(split_frame, 0) == 0, "MD zero-step budget");
    check_observables(split_frame, reference);
    require(!split_method.has_density(),
            "MD initial force does not commit guess");
    cpu_md.run(cpu_frame, 0);
    compare_trajectory(split_frame, cpu_frame);
    require(split.run(split_frame, 2) == 2, "MD first chunk");
    require(split.run(split_frame, 2) == 2, "MD second chunk");
    require(whole.run(whole_frame, 4) == 4, "MD unchunked");
    require(cpu_md.run(cpu_frame, 4) == 4, "CPU four-step MD");
    compare_trajectory(split_frame, whole_frame);
    compare_trajectory(whole_frame, cpu_frame);
    // A host edit between chunks must be seen before the first half kick.
    // Compare with a new driver starting from the edited state, including the
    // old output force: it must prepare a new force rather than reuse that one.
    split_frame.span(kAtom_x)[1] += .013;
    split_frame.span(kAtom_vz)[0] -= .0003;
    auto edited_reference = split_frame;
    Gpu edited_method{0, 1, 1000, 1e-10, 1e-12};
    cuda::md::MD edited;
    edited.set_integrator(cuda::md::VelocityVerlet{dt});
    edited.on(md::Phase::force).add_ref(edited_method);
    edited.build();
    require(split.run(split_frame, 1) == 1, "MD edited chunk");
    require(edited.run(edited_reference, 1) == 1, "MD edited fresh reference");
    compare_trajectory(split_frame, edited_reference);
    // A rejected box edit must not silently continue from the old device box.
    const double edited_x = split_frame.span(kAtom_x)[1];
    split_frame.box = cpu::Box::cubic(20);
    rejects([&] { split.run(split_frame, 0); },
            "MD edited periodic box rejected");
    near("MD rejected box preserves host coordinate",
         split_frame.span(kAtom_x)[1], edited_x, 0);
    split_frame.box = cpu::Box{};
    split.run(split_frame, 0);
    check_observables(split_frame, direct.forces(edited_reference));
    split_frame = dynamic_molecule(kModernPins[pin.rm1 ? 4 : 1]);
    split.run(split_frame, 0);
    std::swap(split_frame.span(kAtom_type)[0], split_frame.span(kAtom_type)[1]);
    std::swap(split_frame.span(kAtom_mass)[0], split_frame.span(kAtom_mass)[1]);
    edited_reference = split_frame;
    split.run(split_frame, 0);
    check_observables(split_frame, direct.forces(edited_reference));
    // Replacing the same host Frame with another atom count rebuilds mappings.
    split_frame = dynamic_molecule(kModernPins[pin.rm1 ? 3 : 0]);
    edited_reference = split_frame;
    split.run(split_frame, 0);
    check_observables(split_frame, direct.forces(edited_reference));
    std::printf("PASS CUDA %s MD initial force, 2+2 versus 4, CPU four-step "
                "dt=%.3g fs\n",
                pin.rm1 ? "RM1" : "AM1", dt);
}
#endif
#ifdef ATV_HAS_MINIMIZER
__global__ void trial_energy_penalty(cuda::Frame frame) {
    frame.scalars->energy += 1e6;
}
class RejectTrial final : public cuda::Component {
  public:
    bool enabled = true;
    int trials = 0;
    std::string name() const override { return "reject_nddo_trial"; }
    bool provides_energy() const override { return true; }
    void launch(const cuda::LaunchContext& ctx) override {
        if (enabled && ctx.runtime.exec.trial_evaluation) {
            ++trials;
            trial_energy_penalty<<<
                1, 1, 0, static_cast<cudaStream_t>(ctx.runtime.stream)>>>(
                ctx.frame);
            ATV_CUDA_CHECK(cudaGetLastError());
        }
    }
};
template <class Gpu>
void line_search_transaction(const ModernMopacPin& pin) {
    auto frame = dynamic_molecule(pin), reference = dynamic_molecule(pin);
    Gpu method{0, 1, 1000, 1e-10, 1e-12}, direct{0, 1, 1000, 1e-10, 1e-12};
    const auto initial = direct.forces(reference);
    RejectTrial reject;
    cuda::minimizer::LBFGS optimizer{5, .001, 3};
    optimizer.line_search();
    cuda::minimizer::Minimize opt;
    opt.set_optimizer_ref(optimizer);
    opt.add_criterion(cuda::ForceConverged{1e-12, 1e-12, 1e-12, 1e-12, 1e-14});
    opt.on(minimizer::Phase::force).add_ref(method);
    opt.on(minimizer::Phase::force).add_ref(reject, 100);
    opt.build();
    rejects([&] { opt.run(frame, 1); }, "exhausted Armijo search rejects");
    require(reject.trials > 0 && reject.trials <= 3,
            "line search obeys its per-step budget");
    require(!method.has_density(), "rejected trials leave no accepted guess");
    for (int i = 0; i < pin.count; ++i) {
        near("rejected search restores x", frame.span(kAtom_x)[i],
             pin.xyz[i][0], 0);
        near("rejected search restores y", frame.span(kAtom_y)[i],
             pin.xyz[i][1], 0);
        near("rejected search restores z", frame.span(kAtom_z)[i],
             pin.xyz[i][2], 0);
    }
    reject.enabled = false;
    require(opt.run(frame, 1) == 1,
            "same minimizer retries after search failure");
    require(method.has_density(),
            "final accepted step commits without next step");
    require(frame.meta.get(kEnergy) < initial.total_energy_ev,
            "accepted one-step search lowers energy");
    const auto final = method.last_result();
    check_observables(frame, final);
    reference = frame;
    check_observables(frame, direct.forces(reference));
    std::printf("PASS CUDA %s bounded real line search, rejection, retry and "
                "final-step commit\n",
                pin.rm1 ? "RM1" : "AM1");
}
template <class Gpu>
void short_minimize(const ModernMopacPin& pin) {
    auto frame = dynamic_molecule(pin), reference_frame = dynamic_molecule(pin);
    Gpu method{0, 1, 1000, 1e-10, 1e-12}, direct{0, 1, 1000, 1e-10, 1e-12};
    const auto initial = direct.forces(reference_frame);
    cuda::minimizer::Minimize opt;
    opt.set_optimizer(cuda::minimizer::SteepestDescent{.01});
    opt.add_criterion(cuda::ForceConverged{1e-12, 1e-12, 1e-12, 1e-12, 1e-14});
    opt.on(minimizer::Phase::force).add_ref(method);
    opt.build();
    require(opt.run(frame, 0) == 0, "Minimize zero-step budget");
    check_observables(frame, initial);
    require(!method.has_density(),
            "Minimize initial force does not commit guess");
    require(opt.run(frame, 3) == 3, "three small steepest-descent steps");
    const double final_energy = frame.meta.get(kEnergy);
    require(std::isfinite(final_energy)
                && final_energy < initial.total_energy_ev - 1e-8,
            "small SD steps lower energy");
    const auto final_result = method.last_result();
    check_observables(frame, final_result);
    // Preserve the actual minimizer output for comparison before standalone
    // forces overwrites the frame at this identical final geometry.
    const auto independent = direct.forces(frame);
    near("minimizer final standalone energy", final_energy,
         independent.total_energy_ev, 1e-8);
    require(final_result.forces_ev_a.size() == independent.forces_ev_a.size(),
            "minimizer full final force");
    for (std::size_t i = 0; i < independent.forces_ev_a.size(); ++i)
        near("minimizer final standalone force", final_result.forces_ev_a[i],
             independent.forces_ev_a[i], 1e-7);
    std::printf("PASS CUDA %s Minimize initial force and 3 SD steps E %.12f -> "
                "%.12f eV (no convergence claim)\n",
                pin.rm1 ? "RM1" : "AM1", initial.total_energy_ev, final_energy);
}
#endif

template <class Cpu, class Gpu>
void grade(int m) {
    for (const auto& pin : kModernPins) {
        if (pin.rm1 != static_cast<bool>(m))
            continue;
        auto frame = molecule(pin);
        Cpu cpu_method{0, 1, 1000, 1e-10, 1e-12};
        // CPU is the independently pinned twin for the backend comparison.
        auto cpu_frame = molecule(pin);
        const auto cpu_result = cpu_method.forces(cpu_frame);
        LifecycleProbe lifecycle;
        Gpu method{0, 1, 1000, 1e-10, 1e-12};
        method.on(nddo::Phase::check).add_ref(lifecycle, 100);
        method.build();
        const auto result = method.run(frame);
        require(lifecycle.builds == 1 && lifecycle.launches > 0
                    && lifecycle.destroys == 0,
                "standalone stages built once before launch");
        require(result.converged && method.converged(), "CUDA convergence");
        near("CUDA external E", result.total_energy_ev, pin.energy_ev, 1e-6);
        near("CUDA external HOF", result.heat_of_formation_kcal_mol,
             pin.hof_kcal, 1e-6);
        near("CUDA CPU E", result.total_energy_ev, cpu_result.total_energy_ev,
             1e-8);
        for (int i = 0; i < pin.count; ++i) {
            near("CUDA external q", result.mulliken_charges[i], pin.charges[i],
                 1e-7);
            near("CUDA CPU q", result.mulliken_charges[i],
                 cpu_result.mulliken_charges[i], 1e-8);
            near("CUDA Frame q", frame.span(kAtom_charge)[i],
                 result.mulliken_charges[i], 0);
        }
        forces_preserved(frame);
        // Const entry must leave both result channels and force columns untouched.
        frame.meta.get(kEnergy) = 123.0;
        frame.meta.get(kEnergyElectronic) = 234.0;
        frame.span(kAtom_charge)[0] = .456;
        method.run(static_cast<const cpu::Frame&>(frame));
        near("const E", frame.meta.get(kEnergy), 123, 0);
        near("const electronic E", frame.meta.get(kEnergyElectronic), 234, 0);
        near("const q", frame.span(kAtom_charge)[0], .456, 0);
        forces_preserved(frame);
        const auto force_result = method.forces(frame);
        require(force_result.forces_ev_a.size()
                    == 3 * static_cast<std::size_t>(pin.count),
                "CUDA force result size");
        near("forces external E", force_result.total_energy_ev, pin.energy_ev,
             1e-6);
        near("forces Frame E", frame.meta.get(kEnergy),
             force_result.total_energy_ev, 0);
        near("forces external HOF", force_result.heat_of_formation_kcal_mol,
             pin.hof_kcal, 1e-6);
        for (int i = 0; i < pin.count; ++i)
            near("forces Frame external q", frame.span(kAtom_charge)[i],
                 pin.charges[i], 1e-7);
        for (int i = 0; i < pin.count; ++i)
            for (int d = 0; d < 3; ++d) {
                near("CUDA external full force",
                     force_result.forces_ev_a[3 * i + d], pin.forces[i][d],
                     1e-5);
                near("CUDA CPU full force", force_result.forces_ev_a[3 * i + d],
                     cpu_result.forces_ev_a[3 * i + d], 1e-7);
            }
        check_force_frame(frame, force_result.forces_ev_a);
        method.run(frame);
        require(method.last_result().forces_ev_a.empty(),
                "energy run clears method force result");
        check_force_frame(frame, force_result.forces_ev_a);
        method.run(static_cast<const cpu::Frame&>(frame));
        require(method.last_result().forces_ev_a.empty(),
                "const energy run clears method force result");
        check_force_frame(frame, force_result.forces_ev_a);
        // Rotation and translation preserve the same Hamiltonian energy/charges.
        for (int i = 0; i < pin.count; ++i) {
            const double x = pin.xyz[i][0], y = pin.xyz[i][1];
            frame.span(kAtom_x)[i] = .8 * x - .6 * y + .37;
            frame.span(kAtom_y)[i] = .6 * x + .8 * y - .21;
            frame.span(kAtom_z)[i] = pin.xyz[i][2] + .18;
        }
        const auto rotated = method.forces(frame);
        near("rotation/translation E", rotated.total_energy_ev,
             result.total_energy_ev, 1e-8);
        for (int i = 0; i < pin.count; ++i) {
            const double fx = force_result.forces_ev_a[3 * i],
                         fy = force_result.forces_ev_a[3 * i + 1];
            near("rotation force x", rotated.forces_ev_a[3 * i],
                 .8 * fx - .6 * fy, 1e-7);
            near("rotation force y", rotated.forces_ev_a[3 * i + 1],
                 .6 * fx + .8 * fy, 1e-7);
            near("rotation force z", rotated.forces_ev_a[3 * i + 2],
                 force_result.forces_ev_a[3 * i + 2], 1e-7);
        }
        check_force_frame(frame, rotated.forces_ev_a);
        for (int i = 0; i < pin.count; ++i)
            near("rotation/translation q",
                 method.last_result().mulliken_charges[i],
                 result.mulliken_charges[i], 1e-8);
        // Host coordinate changes are uploaded before every pass.
        frame.span(kAtom_x)[1] += .012;
        const auto moved =
            cpu_method.run(static_cast<const cpu::Frame&>(frame));
        near("host coordinate edit", method.run(frame).total_energy_ev,
             moved.total_energy_ev, 1e-8);
        // Failure is transactional with respect to the caller's Frame and guess.
        const double before = frame.meta.get(kEnergy);
        const double oldx = frame.span(kAtom_x)[1];
        frame.span(kAtom_x)[1] = std::numeric_limits<double>::quiet_NaN();
        rejects([&] { method.forces(frame); },
                "force NaN coordinate rejection");
        check_force_frame(frame, rotated.forces_ev_a);
        require(!method.converged() && !method.last_result().converged,
                "failed result invalidated");
        near("failure preserves E", frame.meta.get(kEnergy), before, 0);
        frame.span(kAtom_x)[1] = oldx;
        near("retry E", method.run(frame).total_energy_ev,
             moved.total_energy_ev, 1e-8);
        frame.span(kAtom_type)[0] = 9;
        rejects([&] { method.run(frame); },
                "same-size unsupported identity rejected");
        frame.span(kAtom_type)[0] = pin.species[0];
        near("identity retry", method.run(frame).total_energy_ev,
             moved.total_energy_ev, 1e-8);
        frame.box = cpu::Box::cubic(20);
        rejects([&] { method.run(frame); }, "periodic cell rejected");
        frame.box = cpu::Box{};
        method.run(frame);
        method.forces(frame);
        require(!method.provides_system_stress(),
                "molecular stress capability absent");
        if (pin.count == 2) {
            std::swap(frame.span(kAtom_type)[0], frame.span(kAtom_type)[1]);
            const auto swapped =
                cpu_method.run(static_cast<const cpu::Frame&>(frame));
            const auto actual = method.run(frame);
            near("same-count supported identity E", actual.total_energy_ev,
                 swapped.total_energy_ev, 1e-8);
            for (int i = 0; i < pin.count; ++i)
                near("same-count supported identity q",
                     actual.mulliken_charges[i], swapped.mulliken_charges[i],
                     1e-8);
            std::swap(frame.span(kAtom_type)[0], frame.span(kAtom_type)[1]);
            method.run(frame);
        }
        // A cold finite budget fails; controls changed after failure take effect.
        Gpu limited{0, 1, 1, 1e-10, 1e-12};
        rejects([&] { limited.run(frame); }, "finite budget failure");
        require(!limited.has_density(), "failed guess not committed");
        limited.max_iter = 1000;
        near("finite budget retry", limited.run(frame).total_energy_ev,
             moved.total_energy_ev, 1e-8);
        method.shutdown();
        require(lifecycle.builds == 1 && lifecycle.destroys == 1,
                "persistent standalone lifecycle teardown");
        std::printf("PASS CUDA %s %s E=%.12f iterations=%d\n",
                    m ? "RM1" : "AM1", pin.name, result.total_energy_ev,
                    result.iterations);
    }
    auto frame = molecule(kModernPins[m ? 5 : 2]);
    rejects([&] { Gpu charged{1}; }, "charged constructor rejected");
    rejects([&] { Gpu spin{0, 3}; }, "open-shell constructor rejected");
    {
        Gpu first_failure{0, 1, 1000, 1e-10, 1e-12};
        frame.span(kAtom_type)[0] = 9;
        rejects([&] { first_failure.run(frame); },
                "first invalid identity build");
        require(!first_failure.has_density(), "invalid build retains no guess");
        frame.span(kAtom_type)[0] = 8;
        first_failure.run(frame);
        require(first_failure.converged(), "retry after first build failure");
    }
    {
        Gpu original{0, 1, 1000, 1e-10, 1e-12};
        const double energy = original.run(frame).total_energy_ev;
        Gpu moved(std::move(original));
        require(moved.has_density() && !original.has_density(),
                "idle move preserves accepted density");
        near("idle move result", moved.last_result().total_energy_ev, energy,
             0);
        near("idle move retry", moved.run(frame).total_energy_ev, energy, 1e-8);
    }
    {
        BuildFailure failure;
        Gpu retry{0, 1, 1000, 1e-10, 1e-12};
        retry.on(nddo::Phase::check).add_ref(failure, 100);
        retry.build();
        rejects([&] { retry.run(frame); }, "user component build failure");
        require(failure.builds == 1 && failure.destroys == 1,
                "failed build rollback");
        retry.run(frame);
        require(retry.converged() && failure.builds == 2, "failed build retry");
        retry.shutdown();
        require(failure.destroys == 2, "retried build teardown");
    }
    {
        const auto& pin = kModernPins[m ? 5 : 2];
        auto retry_frame = molecule(pin);
        DestroyFailure failure;
        Gpu retry{0, 1, 1000, 1e-10, 1e-12};
        retry.on(nddo::Phase::check).add_ref(failure, 100);
        {
            cuda::Evaluate parent;
            parent.on(evaluate::Phase::evaluate).add_ref(retry);
            parent.build();
            rejects([&] { parent.run(retry_frame); },
                    "completed stage destruction failure",
                    "injected completed-stage destruction failure");
        }
        require(failure.destroys == 1 && !failure.ready,
                "failed destroy does not repeat freed stage");
        const auto check_retry = [&] {
            const auto& actual = retry.last_result();
            require(actual.converged, "destroy failure retry converged");
            near("destroy failure retry external E", actual.total_energy_ev,
                 pin.energy_ev, 1e-6);
            check_force_frame(retry_frame, actual.forces_ev_a);
            for (int i = 0; i < pin.count; ++i) {
                near("destroy failure retry external q",
                     retry_frame.span(kAtom_charge)[i], pin.charges[i], 1e-7);
                for (int d = 0; d < 3; ++d)
                    near("destroy failure retry external force",
                         actual.forces_ev_a[3 * i + d], pin.forces[i][d], 1e-5);
            }
        };
        retry.forces(retry_frame);
        check_retry();
        retry.shutdown();
        require(failure.destroys == 2,
                "standalone destroy retry releases stage");
        {
            cuda::Evaluate parent;
            parent.on(evaluate::Phase::evaluate).add_ref(retry);
            parent.build();
            parent.run(retry_frame);
            check_retry();
        }
        require(failure.destroys == 3 && !failure.ready,
                "new parent destroys recovered stage once");
        // The standalone owner must finish cleanup before propagating the
        // original failure, permitting this same object to acquire a new owner.
        retry.forces(retry_frame);
        failure.fail = true;
        rejects([&] { retry.shutdown(); },
                "standalone shutdown reports destroy failure",
                "injected completed-stage destruction failure");
        require(failure.destroys == 4 && !failure.ready,
                "failed standalone shutdown releases stage once");
        retry.forces(retry_frame);
        check_retry();
        retry.shutdown();
        require(failure.destroys == 5 && !failure.ready,
                "standalone shutdown failure recovered");
        // A destructor cannot propagate the user callback exception. The
        // callback outlives the method so its completed cleanup is observable.
        DestroyFailure destructor_failure;
        {
            Gpu scoped{0, 1, 1000, 1e-10, 1e-12};
            scoped.on(nddo::Phase::check).add_ref(destructor_failure, 100);
            scoped.forces(retry_frame);
        }
        require(
            destructor_failure.destroys == 1 && !destructor_failure.ready,
            "method destructor survives completed-stage destruction failure");
    }
    LifecycleProbe lifecycle;
    Gpu method{0, 1, 1000, 1e-10, 1e-12};
    {
        // Retry the same parent and child, so a stale parent lifecycle ledger
        // cannot be hidden by constructing a fresh Evaluate after failure.
        ContributionProbe intent;
        Gpu limited{0, 1, 1, 1e-10, 1e-12};
        cuda::Evaluate parent;
        parent.on(evaluate::Phase::prepare).add_ref(intent);
        parent.on(evaluate::Phase::evaluate).add_ref(limited);
        parent.build();
        frame.meta.get(kEnergy) = 123.0;
        frame.span(kAtom_charge)[0] = .456;
        const auto before_forces =
            std::array{frame.span(kAtom_fx)[0], frame.span(kAtom_fy)[0],
                       frame.span(kAtom_fz)[0]};
        rejects([&] { parent.run(frame); },
                "same-parent finite budget failure");
        near("failed parent force x", frame.span(kAtom_fx)[0], before_forces[0],
             0);
        near("failed parent force y", frame.span(kAtom_fy)[0], before_forces[1],
             0);
        near("failed parent force z", frame.span(kAtom_fz)[0], before_forces[2],
             0);
        require(!limited.has_density(), "failed nested guess not committed");
        near("failed parent preserves E", frame.meta.get(kEnergy), 123, 0);
        near("failed parent preserves q", frame.span(kAtom_charge)[0], .456, 0);
        limited.max_iter = 1000;
        parent.run(frame);
        const auto& pin = kModernPins[m ? 5 : 2];
        near("same-parent retry external E", frame.meta.get(kEnergy),
             pin.energy_ev + 1.25, 1e-6);
        for (int i = 0; i < pin.count; ++i)
            near("same-parent retry external q", frame.span(kAtom_charge)[i],
                 pin.charges[i], 1e-7);
    }
    method.on(nddo::Phase::check).add_ref(lifecycle, 100);
    method.build();
    {
        ContributionProbe intent;
        MountedMoveProbe<Gpu> move_probe(method);
        cuda::Evaluate parent;
        parent.on(evaluate::Phase::prepare).add_ref(intent);
        parent.on(evaluate::Phase::prepare).add_ref(move_probe, 1);
        parent.on(evaluate::Phase::evaluate).add_ref(method);
        parent.build();
        parent.run(frame);
    }
    // Reset workspace after the move probe's normal accepted solve.
    method.shutdown();
    lifecycle.builds = lifecycle.launches = lifecycle.destroys = 0;
    nested(method, frame, true, false);
    require(lifecycle.builds == 1 && lifecycle.destroys == 1
                && lifecycle.launches > 0,
            "nested child plan lifecycle");
    require(!method.has_density(), "initial solve leaves history untouched");
    const double expected = method.last_result().total_energy_ev;
    near("nested additional E", frame.meta.get(kEnergy), expected + 1.25,
         1e-10);
    check_force_frame(frame, method.last_result().forces_ev_a, true);
    for (int i = 0; i < frame.nlocal(); ++i)
        near("nested q", frame.span(kAtom_charge)[i],
             method.last_result().mulliken_charges[i], 0);
    nested(method, frame, false, true);
    require(!method.has_density(), "trial solve leaves history untouched");
    nested(method, frame);
    require(method.has_density(), "normal solve commits history");
    rejects([&] { nested(method, frame, false, false, 0); },
            "explicit zero budget");
    require(!method.converged(), "zero budget clears verdict");
    nested(method, frame);
    near("zero budget retry", method.last_result().total_energy_ev, expected,
         1e-8);
    {
        cuda::Evaluate outer;
        outer.on(evaluate::Phase::evaluate).add_ref(method);
        outer.build();
        outer.run(frame);
        const auto direct = method.last_result();
        near("default Evaluate E", frame.meta.get(kEnergy), expected, 1e-8);
        check_force_frame(frame, direct.forces_ev_a);
    }
    {
        cuda::Evaluate owned;
        owned.on(evaluate::Phase::prepare).add(ContributionProbe{});
        owned.on(evaluate::Phase::evaluate).add(Gpu{0, 1, 1000, 1e-10, 1e-12});
        owned.build();
        owned.run(frame);
        near("owned nested method", frame.meta.get(kEnergy), expected + 1.25,
             1e-8);
        check_force_frame(frame, method.last_result().forces_ev_a, true, 1e-7);
    }
    {
        const auto& pin = kModernPins[m ? 5 : 2];
        auto probed_frame = molecule(pin), control_frame = molecule(pin);
        SolveFailure failure;
        Gpu probed{0, 1, 1000, 1e-10, 1e-12}, control{0, 1, 1000, 1e-10, 1e-12};
        probed.on(nddo::Phase::check).add_ref(failure, 100);
        ContributionProbe flags;
        cuda::Evaluate parent;
        parent.on(evaluate::Phase::prepare).add_ref(flags);
        parent.on(evaluate::Phase::evaluate).add_ref(probed);
        parent.build();
        parent.run(probed_frame);
        control.forces(control_frame);
        require(probed.has_density(), "accepted history established");
        for (int step = 0; step < 3; ++step) {
            const double accepted_x = probed_frame.span(kAtom_x)[1];
            probed_frame.span(kAtom_x)[1] = accepted_x + .017;
            flags.initial = true;
            parent.run(probed_frame);
            flags.initial = false;
            flags.trial = true;
            parent.run(probed_frame);
            flags.trial = false;
            probed_frame.span(kAtom_x)[1] = accepted_x;
            if (step == 1) {
                const auto old = probed.last_result();
                const double olde = probed_frame.meta.get(kEnergy),
                             oldq = probed_frame.span(kAtom_charge)[0];
                failure.fail = true;
                rejects([&] { parent.run(probed_frame); },
                        "force solve failure after accepted history");
                require(
                    probed.has_density() && !probed.converged(),
                    "failure preserves accepted history but clears verdict");
                near("force failure E untouched",
                     probed_frame.meta.get(kEnergy), olde, 0);
                near("force failure q untouched",
                     probed_frame.span(kAtom_charge)[0], oldq, 0);
                check_force_frame(probed_frame, old.forces_ev_a, true);
            }
            probed_frame.span(kAtom_x)[1] += .002;
            control_frame.span(kAtom_x)[1] += .002;
            parent.run(probed_frame);
            const auto a = probed.last_result(),
                       b = control.forces(control_frame);
            require(a.iterations == b.iterations,
                    "initial/trial/failure did not advance accepted guess");
            near("history twin E", a.total_energy_ev, b.total_energy_ev, 1e-8);
            for (std::size_t i = 0; i < a.forces_ev_a.size(); ++i)
                near("history twin force", a.forces_ev_a[i], b.forces_ev_a[i],
                     1e-7);
            check_force_frame(probed_frame, a.forces_ev_a, true);
        }
    }
    {
        // The child can finish successfully while a later parent component
        // fails. Its provisional density must not become the next guess.
        const auto& pin = kModernPins[m ? 5 : 2];
        auto failed_frame = molecule(pin), control_frame = molecule(pin);
        Gpu probed{0, 1, 1000, 1e-10, 1e-12}, control{0, 1, 1000, 1e-10, 1e-12};
        probed.forces(failed_frame);
        control.forces(control_frame);
        SolveFailure later;
        cuda::Evaluate parent;
        parent.on(evaluate::Phase::evaluate).add_ref(probed);
        parent.on(evaluate::Phase::evaluate).add_ref(later, 100);
        parent.build();
        failed_frame.span(kAtom_x)[1] += .04;
        later.fail = true;
        rejects([&] { parent.run(failed_frame); },
                "later parent component failure rolls back electronic history");
        failed_frame.span(kAtom_x)[1] = control_frame.span(kAtom_x)[1] + .002;
        control_frame.span(kAtom_x)[1] += .002;
        parent.run(failed_frame);
        const auto a = probed.last_result(), b = control.forces(control_frame);
        require(a.iterations == b.iterations,
                "later parent failure preserves previous accepted guess");
        check_observables(failed_frame, b);
    }
    Cpu changed_cpu{0, 1, 1000, 1e-10, 1e-12};
    auto methane = cpu::Frame(5);
    methane.span(kAtom_type)[0] = 6;
    const double a = .63;
    for (int i = 1; i < 5; ++i) {
        methane.span(kAtom_type)[i] = 1;
        methane.span(kAtom_x)[i] = (i == 1 || i == 2) ? a : -a;
        methane.span(kAtom_y)[i] = (i == 1 || i == 3) ? a : -a;
        methane.span(kAtom_z)[i] = (i == 1 || i == 4) ? a : -a;
    }
    const auto methane_cpu =
        changed_cpu.run(static_cast<const cpu::Frame&>(methane));
    near("atom-count rebuild CH4", method.run(methane).total_energy_ev,
         methane_cpu.total_energy_ev, 1e-8);
#ifdef ATV_HAS_MD
    short_md<Cpu, Gpu>(kModernPins[m ? 5 : 2]);
#else
    std::printf("SKIP CUDA %s MD composition: ATV_HAS_MD disabled\n",
                m ? "RM1" : "AM1");
#endif
#ifdef ATV_HAS_MINIMIZER
    short_minimize<Gpu>(kModernPins[m ? 5 : 2]);
    line_search_transaction<Gpu>(kModernPins[m ? 5 : 2]);
#else
    std::printf(
        "SKIP CUDA %s Minimize composition: ATV_HAS_MINIMIZER disabled\n",
        m ? "RM1" : "AM1");
#endif
}
} // namespace
int run(int argc, char** argv) {
    std::string device = "both";
    for (int i = 1; i < argc; ++i) {
        if (std::strncmp(argv[i], "--device=", 9) == 0)
            device = argv[i] + 9;
        else if (std::strcmp(argv[i], "--device") == 0 && i + 1 < argc)
            device = argv[++i];
    }
    if (device != "cpu" && device != "cuda" && device != "both")
        return 1;
    try {
        if (device != "cuda"
            && atv_reg_semiempirical_am1_water_mopac_energy::run_cpu() != 0)
            return 1;
        if (device != "cpu") {
            grade<cpu::semiempirical::AM1, cuda::semiempirical::AM1>(0);
            grade<cpu::semiempirical::RM1, cuda::semiempirical::RM1>(1);
        }
    } catch (const std::exception& error) {
        std::fprintf(stderr, "FAIL CUDA NDDO: %s\n", error.what());
        return 1;
    }
    return 0;
}
} // namespace atv_reg_semiempirical_cuda_native_energy
ATV_REGRESSION("semiempirical/am1", "am1-water-mopac-energy",
               atv_reg_semiempirical_cuda_native_energy::run);
