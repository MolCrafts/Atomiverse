/// @file regressions/semiempirical/nddo_nve_cuda.cuh
/// @brief Native CUDA NDDO NVE qualification through public MD/Frame APIs.
#pragma once

#include "atomiverse/cuda/dynamics/md.cuh"
#include "atomiverse/cuda/dynamics/velocity_verlet.cuh"
#include "nddo_nve_cpu.h"
#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <utility>

namespace atv_reg_nddo_nve_cuda {
using namespace ATV;
using atv_reg_nddo_nve::Trace;

inline void compare_value(const std::string& label, double got, double want,
                          double tolerance) {
    if (!std::isfinite(got) || !std::isfinite(want)
        || std::abs(got - want) > tolerance) {
        std::fprintf(stderr, "FAIL %s got=%.16g want=%.16g tol=%.3g\n",
                     label.c_str(), got, want, tolerance);
        throw std::runtime_error(label);
    }
}

template <class Method, class Pin>
Trace run_cuda(const Pin& water, double dt) {
    auto frame = atv_reg_nddo_nve::initial_frame(water);
    Trace trace;
    const std::string label =
        std::string(water.name) + " CUDA NVE dt=" + std::to_string(dt);
    {
        Method method{0, 1, 400, 1e-10, 1e-12};
        cuda::md::MD simulation;
        simulation.set_integrator(cuda::md::VelocityVerlet{dt});
        simulation.on(md::Phase::force).add_ref(method);
        simulation.build();
        simulation.run(frame, 0);
        compare_value(label + " initial external potential",
                      frame.meta.get(kEnergy), water.energy_ev, 1e-6);
        atv_reg_nddo_nve::record_sample(trace, frame, 0, label.c_str());
        const int chunk = static_cast<int>(
            std::lround(atv_reg_nddo_nve::kSampleInterval / dt));
        compare_value(label + " exact sample duration", chunk * dt,
                      atv_reg_nddo_nve::kSampleInterval, 1e-14);
        for (int sample = 1; sample <= atv_reg_nddo_nve::kSamples; ++sample) {
            const auto steps = simulation.run(frame, chunk);
            compare_value(
                label + " completed steps sample=" + std::to_string(sample),
                static_cast<double>(steps), chunk, 0);
            atv_reg_nddo_nve::record_sample(trace, frame, sample,
                                            label.c_str());
        }
    }
    atv_reg_nddo_nve::finish_trace(trace, std::move(frame), label.c_str());
    return trace;
}

inline void compare_backends(const Trace& gpu, const Trace& cpu,
                             const std::string& label) {
    // This is a fixed 10 fs water trajectory, with the same SCF tolerances and
    // integrator. These checks do not prescribe agreement of long chaotic paths.
    for (int sample = 0; sample <= atv_reg_nddo_nve::kSamples; ++sample) {
        const auto where = label + " sample=" + std::to_string(sample);
        compare_value(where + " total energy", gpu.total_energy[sample],
                      cpu.total_energy[sample], 1e-6);
        compare_value(where + " kinetic energy", gpu.kinetic[sample],
                      cpu.kinetic[sample], 1e-6);
    }
    const auto& a = gpu.final_frame;
    const auto& b = cpu.final_frame;
    compare_value(label + " atom count", a.nlocal(), b.nlocal(), 0);
    compare_value(label + " final potential", a.meta.get(kEnergy),
                  b.meta.get(kEnergy), 1e-6);
    for (int i = 0; i < a.nlocal(); ++i) {
        const auto where = label + " atom=" + std::to_string(i);
        compare_value(where + " charge", a.span(kAtom_charge)[i],
                      b.span(kAtom_charge)[i], 1e-7);
        const double xyz_a[] = {a.span(kAtom_x)[i], a.span(kAtom_y)[i],
                                a.span(kAtom_z)[i]};
        const double xyz_b[] = {b.span(kAtom_x)[i], b.span(kAtom_y)[i],
                                b.span(kAtom_z)[i]};
        const double v_a[] = {a.span(kAtom_vx)[i], a.span(kAtom_vy)[i],
                              a.span(kAtom_vz)[i]};
        const double v_b[] = {b.span(kAtom_vx)[i], b.span(kAtom_vy)[i],
                              b.span(kAtom_vz)[i]};
        const double f_a[] = {a.span(kAtom_fx)[i], a.span(kAtom_fy)[i],
                              a.span(kAtom_fz)[i]};
        const double f_b[] = {b.span(kAtom_fx)[i], b.span(kAtom_fy)[i],
                              b.span(kAtom_fz)[i]};
        for (int d = 0; d < 3; ++d) {
            const auto axis = where + " axis=" + std::to_string(d);
            compare_value(axis + " position", xyz_a[d], xyz_b[d], 1e-8);
            compare_value(axis + " velocity", v_a[d], v_b[d], 1e-9);
            compare_value(axis + " force", f_a[d], f_b[d], 1e-5);
        }
    }
}

template <class Cpu, class Gpu, class Pin>
int grade(const Pin& water, const char* method, bool both) {
    const auto coarse = run_cuda<Gpu>(water, atv_reg_nddo_nve::kDt);
    const auto fine = run_cuda<Gpu>(water, atv_reg_nddo_nve::kDt / 2);
    const std::string label = std::string(method) + " CUDA";
    if (!atv_reg_nddo_nve::check_halving(coarse, fine, label.c_str()))
        return 1;
    if (both) {
        const auto cpu_coarse =
            atv_reg_nddo_nve::run_cpu<Cpu>(water, atv_reg_nddo_nve::kDt);
        const auto cpu_fine =
            atv_reg_nddo_nve::run_cpu<Cpu>(water, atv_reg_nddo_nve::kDt / 2);
        const std::string cpu_label = std::string(method) + " CPU";
        if (!atv_reg_nddo_nve::check_halving(cpu_coarse, cpu_fine,
                                             cpu_label.c_str()))
            return 1;
        compare_backends(coarse, cpu_coarse, label + " coarse vs CPU");
        compare_backends(fine, cpu_fine, label + " fine vs CPU");
    }
    std::printf("PASS %s NVE energy conservation and timestep halving\n",
                method);
    return 0;
}
} // namespace atv_reg_nddo_nve_cuda
