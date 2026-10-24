/// CUDA NDDO method qualification shared by the MNDO and PM3 oracle sections.
/// Public method/Frame boundaries only; no internal workspace inspection.
#pragma once
#include "atomiverse/core/frame_schema.h"
#include "atomiverse/cuda/device/error.cuh"
#include "atomiverse/cuda/evaluate/evaluate.cuh"
#include "atomiverse/cuda/frame/precision.cuh"
#include "nddo_mopac_fixtures.h"
#ifdef ATV_HAS_MD
#include "atomiverse/cuda/dynamics/md.cuh"
#include "atomiverse/cuda/dynamics/velocity_verlet.cuh"
#endif
#ifdef ATV_HAS_MINIMIZER
#include "atomiverse/cuda/minimize/lbfgs.cuh"
#include "atomiverse/cuda/minimize/minimize.cuh"
#endif
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <utility>

namespace atv_reg_nddo_mopac_cuda {
using namespace ATV;
using atv_reg_nddo_mopac::make_frame;
using atv_reg_nddo_mopac::Pin;
inline void near(const std::string& label, double got, double want,
                 double tol) {
    if (!std::isfinite(got) || std::abs(got - want) > tol) {
        std::fprintf(stderr, "FAIL %s got=%.16g want=%.16g tol=%.3g\n",
                     label.c_str(), got, want, tol);
        throw std::runtime_error(label);
    }
}
inline void require(bool ok, const std::string& label) {
    near(label, ok ? 1 : 0, 1, 0);
}
template <class F>
void rejects(F&& operation, const std::string& label) {
    bool rejected = false;
    try {
        operation();
    } catch (const std::exception&) {
        rejected = true;
    }
    require(rejected, label);
}
inline void compare(const semiempirical::NDDOResult& a,
                    const semiempirical::NDDOResult& b,
                    const std::string& label) {
    require(a.converged && b.converged, label + " converged");
    near(label + " E", a.total_energy_ev, b.total_energy_ev, 1e-8);
    near(label + " HOF", a.heat_of_formation_kcal_mol,
         b.heat_of_formation_kcal_mol, 1e-6);
    require(a.mulliken_charges.size() == b.mulliken_charges.size(),
            label + " q size");
    require(a.forces_ev_a.size() == b.forces_ev_a.size(),
            label + " force size");
    for (std::size_t i = 0; i < a.mulliken_charges.size(); ++i)
        near(label + " q atom " + std::to_string(i), a.mulliken_charges[i],
             b.mulliken_charges[i], 1e-8);
    for (std::size_t i = 0; i < a.forces_ev_a.size(); ++i)
        near(label + " force component " + std::to_string(i), a.forces_ev_a[i],
             b.forces_ev_a[i], 1e-7);
}
inline void outputs(const cpu::Frame& frame, const semiempirical::NDDOResult& r,
                    const std::string& label, bool extra = false) {
    near(label + " Frame E", frame.meta.get(kEnergy),
         r.total_energy_ev + (extra ? 1.25 : 0), 1e-8);
    require(r.forces_ev_a.size() == 3 * std::size_t(frame.nlocal()),
            label + " full force");
    const auto fx = frame.span(kAtom_fx), fy = frame.span(kAtom_fy),
               fz = frame.span(kAtom_fz);
    for (int i = 0; i < frame.nlocal(); ++i) {
        near(label + " Frame q " + std::to_string(i),
             frame.span(kAtom_charge)[i], r.mulliken_charges[i], 1e-8);
        const double f[3] = {fx[i], fy[i], fz[i]};
        const double added[3] = {.125, -.25, .5};
        for (int d = 0; d < 3; ++d)
            near(label + " Frame F " + std::to_string(3 * i + d), f[d],
                 r.forces_ev_a[3 * i + d] + (extra ? added[d] : 0), 1e-7);
    }
}
namespace {
__global__ void contribute(cuda::Frame frame, int atom_count) {
    const int i = threadIdx.x;
    if (i == 0)
        frame.scalars->energy += 1.25;
    if (i < atom_count) {
        cuda::add_force(frame.atoms.fx, i, .125, frame.force_fault);
        cuda::add_force(frame.atoms.fy, i, -.25, frame.force_fault);
        cuda::add_force(frame.atoms.fz, i, .5, frame.force_fault);
    }
}
} // namespace
class Intent final : public cuda::Component {
  public:
    bool initial = false, trial = false, extra = false;
    std::string name() const override { return "nddo_oracle_intent"; }
    void launch(const cuda::LaunchContext& ctx) override {
        ctx.runtime.exec.initializing_forces = initial;
        ctx.runtime.exec.trial_evaluation = trial;
        if (extra) {
            contribute<<<1, 32, 0,
                         static_cast<cudaStream_t>(ctx.runtime.stream)>>>(
                ctx.frame, ctx.frame.n_atoms());
            ATV_CUDA_CHECK(cudaGetLastError());
        }
    }
};
class FailOnce final : public cuda::Component {
  public:
    bool fail = false;
    std::string name() const override { return "nddo_oracle_fail_once"; }
    void launch(const cuda::LaunchContext&) override {
        if (fail) {
            fail = false;
            throw std::runtime_error("injected parent failure");
        }
    }
};

template <class Cpu, class Gpu>
void history_and_composition(const Pin& pin, const std::string& label) {
    auto frame = make_frame(pin), control_frame = make_frame(pin);
    Gpu method{0, 1, 1000, 1e-10, 1e-12}, control{0, 1, 1000, 1e-10, 1e-12};
    Intent intent;
    FailOnce failure;
    cuda::Evaluate parent;
    parent.on(evaluate::Phase::prepare).add_ref(intent, 100);
    parent.on(evaluate::Phase::evaluate).add_ref(method);
    parent.on(evaluate::Phase::finalize).add_ref(failure);
    parent.build();
    intent.initial = true;
    parent.run(frame);
    require(!method.has_density(), label + " initial guess not committed");
    intent.initial = false;
    intent.trial = true;
    parent.run(frame);
    require(!method.has_density(), label + " trial guess not committed");
    intent.trial = false;
    intent.extra = true;
    parent.run(frame);
    require(method.has_density(), label + " accepted guess committed");
    outputs(frame, method.last_result(), label + " additive", true);
    compare(method.last_result(), control.forces(control_frame),
            label + " normal twin");
    // A genuinely different trial must not seed the next accepted calculation.
    frame.span(kAtom_x)[1] += .13;
    intent.trial = true;
    parent.run(frame);
    intent.trial = false;
    failure.fail = true;
    frame.span(kAtom_x)[1] -= .06;
    frame.meta.get(kEnergy) = 4321;
    frame.span(kAtom_fx)[0] = .8125;
    rejects([&] { parent.run(frame); }, label + " trailing parent failure");
    near(label + " failed parent E unchanged", frame.meta.get(kEnergy), 4321,
         0);
    near(label + " failed parent force unchanged", frame.span(kAtom_fx)[0],
         .8125, 0);
    require(method.has_density(), label + " old accepted guess retained");
    frame = make_frame(pin);
    control_frame = make_frame(pin);
    frame.span(kAtom_z)[1] += .023;
    control_frame.span(kAtom_z)[1] += .023;
    parent.run(frame);
    const auto actual = method.last_result();
    const auto expected = control.forces(control_frame);
    compare(actual, expected, label + " accepted-history twin");
    near(label + " history iteration count", actual.iterations,
         expected.iterations, 0);
    outputs(frame, actual, label + " accepted retry", true);
    Cpu external_twin{0, 1, 1000, 1e-10, 1e-12};
    auto cpu_frame = frame;
    compare(actual, external_twin.forces(cpu_frame),
            label + " changed geometry CPU");
}

#ifdef ATV_HAS_MD
template <class Cpu, class Gpu, std::size_t N>
void host_edits(const Pin (&pins)[N], const std::string& label) {
    auto frame = make_frame(pins[0]);
    Gpu method{0, 1, 1000, 1e-10, 1e-12};
    cuda::md::MD md;
    md.set_integrator(cuda::md::VelocityVerlet{.001});
    md.on(ATV::md::Phase::force).add_ref(method);
    md.build();
    md.run(frame, 1);
    frame.span(kAtom_x)[1] += .037;
    frame.span(kAtom_vy)[0] += .0001;
    auto reference = frame;
    Gpu fresh_method{0, 1, 1000, 1e-10, 1e-12};
    cuda::md::MD fresh;
    fresh.set_integrator(cuda::md::VelocityVerlet{.001});
    fresh.on(ATV::md::Phase::force).add_ref(fresh_method);
    fresh.build();
    fresh.run(reference, 1);
    md.run(frame, 1);
    compare(method.last_result(), fresh_method.last_result(),
            label + " host x/v continuation");
    for (int i = 0; i < frame.nlocal(); ++i) {
        near(label + " edited position " + std::to_string(i),
             frame.span(kAtom_x)[i], reference.span(kAtom_x)[i], 1e-9);
        near(label + " edited velocity " + std::to_string(i),
             frame.span(kAtom_vy)[i], reference.span(kAtom_vy)[i], 1e-9);
    }
    frame.box = cpu::Box::cubic(20);
    rejects([&] { md.run(frame, 0); }, label + " edited box rejected");
    frame.box = cpu::Box{};
    md.run(frame, 0);
    Cpu cpu_method{0, 1, 1000, 1e-10, 1e-12};
    reference = frame;
    compare(method.last_result(), cpu_method.forces(reference),
            label + " box correction retry");
    for (const auto& pin : pins) {
        frame = make_frame(pin);
        md.run(frame, 0);
        require(atv_reg_nddo_mopac::check_result(method.last_result(), pin,
                                                 label.c_str()),
                label + " population external");
        outputs(frame, method.last_result(), label + " population Frame");
        if (pin.count == 2) {
            std::swap(frame.span(kAtom_type)[0], frame.span(kAtom_type)[1]);
            md.run(frame, 0);
            reference = frame;
            compare(method.last_result(), cpu_method.forces(reference),
                    label + " same-N type edit");
        }
    }
}
#endif

#ifdef ATV_HAS_MINIMIZER
template <class Gpu>
void accepted_minimum_step(const Pin& pin, const std::string& label) {
    auto frame = make_frame(pin);
    Gpu method{0, 1, 1000, 1e-10, 1e-12}, direct{0, 1, 1000, 1e-10, 1e-12};
    const double initial =
        direct.run(static_cast<const cpu::Frame&>(frame)).total_energy_ev;
    auto algorithm = cuda::minimizer::LBFGS{5, .001, 8};
    algorithm.line_search();
    cuda::minimizer::Minimize min;
    min.set_optimizer_ref(algorithm);
    min.on(minimizer::Phase::force).add_ref(method);
    min.build();
    min.run(frame, 1);
    require(method.has_density(), label + " final accepted search commits");
    require(frame.meta.get(kEnergy) < initial,
            label + " accepted search lowers E");
    outputs(frame, method.last_result(), label + " search outputs");
}
#endif

template <class Cpu, class Gpu, std::size_t N>
void grade(const Pin (&pins)[N], const char* method_name) {
    for (const auto& pin : pins) {
        const std::string label = std::string(method_name) + " " + pin.name;
        auto frame = make_frame(pin), cpu_frame = make_frame(pin);
        Cpu cpu_method{0, 1, 1000, 1e-10, 1e-12};
        Gpu method{0, 1, 1000, 1e-10, 1e-12};
        for (int i = 0; i < pin.count; ++i) {
            frame.span(kAtom_fx)[i] = .125;
            frame.span(kAtom_fy)[i] = -.25;
            frame.span(kAtom_fz)[i] = .5;
        }
        const auto energy = method.run(frame);
        near(label + " run external E", energy.total_energy_ev, pin.energy_ev,
             1e-6);
        for (int i = 0; i < pin.count; ++i) {
            near(label + " run force x preserved", frame.span(kAtom_fx)[i],
                 .125, 0);
            near(label + " run force y preserved", frame.span(kAtom_fy)[i],
                 -.25, 0);
            near(label + " run force z preserved", frame.span(kAtom_fz)[i], .5,
                 0);
            near(label + " run external q " + std::to_string(i),
                 frame.span(kAtom_charge)[i], pin.charges[i], 1e-7);
        }
        frame.meta.get(kEnergy) = 123;
        frame.span(kAtom_charge)[0] = .456;
        method.run(static_cast<const cpu::Frame&>(frame));
        near(label + " const energy preserved", frame.meta.get(kEnergy), 123,
             0);
        near(label + " const charge preserved", frame.span(kAtom_charge)[0],
             .456, 0);
        for (int i = 0; i < pin.count; ++i) {
            near(label + " const force x preserved", frame.span(kAtom_fx)[i],
                 .125, 0);
            near(label + " const force y preserved", frame.span(kAtom_fy)[i],
                 -.25, 0);
            near(label + " const force z preserved", frame.span(kAtom_fz)[i],
                 .5, 0);
        }
        const auto result = method.forces(frame);
        require(atv_reg_nddo_mopac::check_result(result, pin, label.c_str()),
                label + " external result");
        compare(result, cpu_method.forces(cpu_frame), label + " CPU twin");
        outputs(frame, result, label);
        auto rotated = make_frame(pin);
        for (int i = 0; i < pin.count; ++i) {
            rotated.span(kAtom_x)[i] = -pin.xyz[i][1] + .25;
            rotated.span(kAtom_y)[i] = pin.xyz[i][0] - .5;
            rotated.span(kAtom_z)[i] = pin.xyz[i][2] + .125;
        }
        const auto covariant = method.forces(rotated);
        require(atv_reg_nddo_mopac::check_result(covariant, pin, label.c_str(),
                                                 false),
                label + " transformed scalars");
        for (int i = 0; i < pin.count; ++i) {
            near(label + " rotated force x " + std::to_string(i),
                 covariant.forces_ev_a[3 * i], -pin.forces[i][1], 1e-5);
            near(label + " rotated force y " + std::to_string(i),
                 covariant.forces_ev_a[3 * i + 1], pin.forces[i][0], 1e-5);
            near(label + " rotated force z " + std::to_string(i),
                 covariant.forces_ev_a[3 * i + 2], pin.forces[i][2], 1e-5);
        }
        Gpu limited{0, 1, 1, 1e-10, 1e-12};
        cuda::Evaluate retry;
        retry.on(evaluate::Phase::evaluate).add_ref(limited);
        retry.build();
        frame.meta.get(kEnergy) = 987;
        rejects([&] { retry.run(frame); },
                label + " same-parent SCF budget failure");
        require(!limited.has_density(),
                label + " failed cold guess not committed");
        near(label + " budget failure E preserved", frame.meta.get(kEnergy),
             987, 0);
        limited.max_iter = 1000;
        retry.run(frame);
        require(atv_reg_nddo_mopac::check_result(limited.last_result(), pin,
                                                 label.c_str()),
                label + " retry external");
        outputs(frame, limited.last_result(), label + " same-parent retry");
        std::printf("PASS CUDA %s E/q/HOF/full force and same-parent retry\n",
                    label.c_str());
    }
    rejects([] { Gpu charged{1}; },
            std::string(method_name) + " charged unsupported");
    rejects([] { Gpu open_shell{0, 3}; },
            std::string(method_name) + " spin unsupported");
    for (const auto& pin : pins)
        if (pin.count == 3) {
            auto invalid = make_frame(pin);
            Gpu method{0, 1, 1000, 1e-10, 1e-12};
            invalid.span(kAtom_type)[0] = 9;
            rejects([&] { method.run(invalid); },
                    std::string(method_name) + " unsupported element");
            require(!method.has_density(),
                    std::string(method_name) + " invalid first build no guess");
            invalid.span(kAtom_type)[0] = pin.species[0];
            invalid.box = cpu::Box::cubic(20);
            rejects([&] { method.forces(invalid); },
                    std::string(method_name) + " unsupported periodic cell");
            invalid.box = cpu::Box{};
            require(atv_reg_nddo_mopac::check_result(method.forces(invalid),
                                                     pin, method_name),
                    std::string(method_name)
                        + " invalid input correction retry");
            history_and_composition<Cpu, Gpu>(pin, method_name);
#ifdef ATV_HAS_MINIMIZER
            accepted_minimum_step<Gpu>(pin, method_name);
#endif
        }
#ifdef ATV_HAS_MD
    host_edits<Cpu, Gpu>(pins, method_name);
#endif
}
} // namespace atv_reg_nddo_mopac_cuda
