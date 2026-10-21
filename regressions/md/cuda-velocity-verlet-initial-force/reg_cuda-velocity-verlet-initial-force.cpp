/// @file regressions/md/cuda-velocity-verlet-initial-force/reg_cuda-velocity-verlet-initial-force.cpp
/// @brief Initial half-kick and continued CUDA Velocity Verlet trajectory.
#include "atomiverse/core/frame_schema.h"
#include "atomiverse/core/units.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cuda/dynamics/md.cuh"
#include "atomiverse/cuda/dynamics/velocity_verlet.cuh"
#include "atomiverse/cuda/potentials/bond_harmonic.cuh"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <type_traits>

using namespace ATV;

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

    ATV::cuda::md::MD whole_sim, chunked_sim;
    for (auto* sim : {&whole_sim, &chunked_sim}) {
        sim->set_bond(ATV::cuda::BondHarmonic{{spring}, {equilibrium}});
        sim->set_integrator(ATV::cuda::md::VelocityVerlet{dt});
        sim->build();
    }
    analytic_step();
    chunked_sim.run(chunked, 1);
    const double first_error = trajectory_error(chunked);
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
    std::printf("CUDA Velocity Verlet: first_error=%.6g continued_error=%.6g "
                "whole_error=%.6g chunk_error=%.6g\n",
                first_error, continued_error, whole_error, chunk_error);
    constexpr double tolerance = (std::is_same_v<ATV::cuda::real_t, float>
                                  || std::is_same_v<ATV::cuda::work_t, float>)
                                     ? 3e-6
                                     : 1e-11;
    return std::max({first_error, continued_error, whole_error, chunk_error})
                   < tolerance
               ? 0
               : 1;
} catch (const std::exception& e) {
    std::fprintf(stderr, "cuda-velocity-verlet-initial-force: %s\n", e.what());
    return 1;
}
