/// @file regressions/md/cpu-velocity-verlet-initial-force/reg_cpu-velocity-verlet-initial-force.cpp
/// @brief Initial half-kick and continued CPU Velocity Verlet trajectory.
#include "atomiverse/core/frame_schema.h"
#include "atomiverse/core/units.h"
#include "atomiverse/cpu/dynamics/md.h"
#include "atomiverse/cpu/dynamics/velocity_verlet.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/potentials/bond_harmonic.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <type_traits>
#include <vector>

using namespace ATV;

namespace {

class CountLaunches : public cpu::Component {
  public:
    explicit CountLaunches(std::vector<int>& steps) : steps_(steps) {}
    std::string name() const override { return "count_launches"; }
    void launch(const cpu::LaunchContext& ctx) override {
        steps_.push_back(ctx.step);
    }

  private:
    std::vector<int>& steps_;
};

} // namespace

int main() try {
    constexpr double spring = 7.0, equilibrium = 1.2, dt = 0.5;
    constexpr double masses[2] = {2.0, 3.0};
    cpu::Frame whole(2), chunked(2);
    for (cpu::Frame* frame : {&whole, &chunked}) {
        frame->box = cpu::Box::cubic(30.0);
        for (int i = 0; i < 2; ++i) {
            frame->span(kAtom_x)[i] = i == 0 ? 4.0 : 6.4;
            frame->span(kAtom_y)[i] = 3.0;
            frame->span(kAtom_z)[i] = 3.0;
            frame->span(kAtom_vx)[i] = 0.0;
            frame->span(kAtom_vy)[i] = 0.0;
            frame->span(kAtom_vz)[i] = 0.0;
            frame->span(kAtom_mass)[i] = masses[i];
            frame->span(kAtom_tag)[i] = i;
        }
        frame->rebuild_tag_index(0, 2);
        frame->add_bond(0, 1, 0);
    }

    // Swope et al., J. Chem. Phys. 76, 637 (1982), velocity Verlet:
    // x' = x + dt*v + dt^2*a(x)/2, v' = v + dt*(a(x)+a(x'))/2.
    // This two-body harmonic force has the analytic derivative of K(r-r0)^2.
    double expected_x[2] = {4.0, 6.4}, expected_v[2] = {};
    auto analytic_step = [&] {
        const double force =
            2 * spring * (expected_x[1] - expected_x[0] - equilibrium);
        const double old_a[2] = {force * units::force_to_accel / masses[0],
                                 -force * units::force_to_accel / masses[1]};
        for (int i = 0; i < 2; ++i)
            expected_x[i] += dt * expected_v[i] + 0.5 * dt * dt * old_a[i];
        const double new_force =
            2 * spring * (expected_x[1] - expected_x[0] - equilibrium);
        for (int i = 0; i < 2; ++i) {
            const double new_a = (i == 0 ? new_force : -new_force)
                                 * units::force_to_accel / masses[i];
            expected_v[i] += 0.5 * dt * (old_a[i] + new_a);
        }
    };
    auto trajectory_error = [&](const cpu::Frame& frame) {
        double error = 0;
        for (int i = 0; i < 2; ++i) {
            const double residuals[] = {
                double(frame.span(kAtom_x)[i]) - expected_x[i],
                double(frame.span(kAtom_vx)[i]) - expected_v[i],
                double(frame.span(kAtom_y)[i]) - 3.0,
                double(frame.span(kAtom_z)[i]) - 3.0,
                double(frame.span(kAtom_vy)[i]),
                double(frame.span(kAtom_vz)[i])};
            for (double residual : residuals) {
                if (!std::isfinite(residual))
                    throw std::runtime_error("non-finite trajectory");
                error = std::max(error, std::abs(residual));
            }
        }
        return error;
    };

    ATV::cpu::md::MD whole_sim, chunked_sim;
    std::vector<int> force_steps, initial_steps, end_steps;
    chunked_sim.on(md::Phase::post_force).add(CountLaunches{force_steps});
    chunked_sim.on(md::Phase::initial_integrate)
        .add(CountLaunches{initial_steps});
    chunked_sim.on(md::Phase::end_of_step).add(CountLaunches{end_steps});
    for (auto* sim : {&whole_sim, &chunked_sim}) {
        sim->set_evaluator(ATV::cpu::BondHarmonic{{spring}, {equilibrium}});
        sim->set_integrator(ATV::cpu::md::VelocityVerlet{dt});
        sim->build();
    }
    // A zero-step run performs setup and the initial force evaluation, but
    // changes neither positions nor velocities and consumes no trajectory step.
    chunked_sim.run(chunked, 0);
    const double initial_error = trajectory_error(chunked);
    const double initial_force = 2 * spring * (6.4 - 4.0 - equilibrium);
    const double initial_force_error =
        std::abs(double(chunked.span(kAtom_fx)[0]) - initial_force);
    const double initial_energy_error =
        std::abs(chunked.meta.get(kEnergy)
                 - spring * std::pow(6.4 - 4.0 - equilibrium, 2));
    if (chunked.meta.get(kKinetic) != 0.0)
        throw std::runtime_error("zero-step kinetic energy is not input KE");
    chunked_sim.run(chunked, 0);
    const double repeated_zero_error = trajectory_error(chunked);
    analytic_step();
    chunked_sim.run(chunked, 1);
    const double first_error = trajectory_error(chunked);
    chunked_sim.run(chunked, 0);
    const double continued_zero_error = trajectory_error(chunked);
    analytic_step();
    chunked_sim.run(chunked, 1);
    const double continued_error = trajectory_error(chunked);
    whole_sim.run(whole, 2);
    const double whole_error = trajectory_error(whole);
    double chunk_error = 0;
    for (int i = 0; i < 2; ++i) {
        chunk_error =
            std::max(chunk_error, std::abs(double(whole.span(kAtom_x)[i])
                                           - double(chunked.span(kAtom_x)[i])));
        chunk_error = std::max(chunk_error,
                               std::abs(double(whole.span(kAtom_vx)[i])
                                        - double(chunked.span(kAtom_vx)[i])));
    }
    std::printf("CPU Velocity Verlet: first_error=%.6g continued_error=%.6g "
                "whole_error=%.6g chunk_error=%.6g\n",
                first_error, continued_error, whole_error, chunk_error);
    constexpr double tolerance = (std::is_same_v<ATV::StorageReal, float>
                                  || std::is_same_v<ATV::WorkReal, float>)
                                     ? 3e-6
                                     : 1e-11;
    const double max_error =
        std::max({initial_error, initial_force_error, initial_energy_error,
                  repeated_zero_error, continued_zero_error, first_error,
                  continued_error, whole_error, chunk_error});
    std::printf("CPU zero-step/setup maximum error=%.6g\n", max_error);
    if (force_steps != std::vector<int>{0, 0, 1}
        || initial_steps != std::vector<int>{0, 1}
        || end_steps != std::vector<int>{0, 1})
        throw std::runtime_error(
            "zero-step/continuation changed phase cadence");
    return max_error < tolerance ? 0 : 1;
} catch (const std::exception& e) {
    std::fprintf(stderr, "cpu-velocity-verlet-initial-force: %s\n", e.what());
    return 1;
}
