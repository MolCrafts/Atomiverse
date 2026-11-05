/// @file regressions/semiempirical/nddo_nve_cpu.h
/// @brief Fixed-duration NVE and time-step convergence for externally pinned NDDO water.
#pragma once
#include "atomiverse/core/frame_schema.h"
#include "atomiverse/core/units.h"
#include "atomiverse/cpu/dynamics/md.h"
#include "atomiverse/cpu/dynamics/velocity_verlet.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

namespace atv_reg_nddo_nve {
inline constexpr double kDt = .1;
inline constexpr double kDuration = 10.;
inline constexpr double kSampleInterval = .5;
inline constexpr int kSamples = 20;
// A priori acceptance: 1 meV/molecule absolute sampled energy deviation;
// halving dt should reduce the O(dt^2) Verlet error (ideal ratio 1/4).
// The 0.4 ceiling allows higher-order terms, without a fitted noise floor.
inline constexpr double kMaxDriftEv = 1e-3;
inline constexpr double kHalvingRatio = .4;

struct Trace {
    ATV::cpu::Frame final_frame;
    std::array<double, kSamples + 1> total_energy{};
    std::array<double, kSamples + 1> kinetic{};
    double initial_energy = 0;
    double max_drift = 0;
    double rms_drift = 0;
    double final_drift = 0;
};

template <class Pin>
ATV::cpu::Frame initial_frame(const Pin& water) {
    using namespace ATV;
    if (water.count != 3 || water.species[0] != 8 || water.species[1] != 1
        || water.species[2] != 1)
        throw std::invalid_argument(
            "NDDO NVE requires the archived O,H,H water fixture");
    cpu::Frame frame(3);
    for (int i = 0; i < 3; ++i) {
        frame.span(kAtom_atomic_number)[i] = water.species[i];
        frame.span(kAtom_tag)[i] = i + 1;
        frame.span(kAtom_mass)[i] = i == 0 ? 16 : 1;
        frame.span(kAtom_x)[i] = water.xyz[i][0];
        frame.span(kAtom_y)[i] = water.xyz[i][1];
        frame.span(kAtom_z)[i] = water.xyz[i][2];
    }
    // Fixed velocities [Angstrom/fs], with exactly specified zero total COM
    // momentum. No thermostat, random draw, COM-removal fix, or periodic cell.
    frame.span(kAtom_vx)[1] = .01;
    frame.span(kAtom_vy)[1] = .002;
    frame.span(kAtom_vz)[1] = 0;
    frame.span(kAtom_vx)[2] = -.003;
    frame.span(kAtom_vy)[2] = .008;
    frame.span(kAtom_vz)[2] = .001;
    frame.span(kAtom_vx)[0] = -(.01 - .003) / 16;
    frame.span(kAtom_vy)[0] = -(.002 + .008) / 16;
    frame.span(kAtom_vz)[0] = -.001 / 16;
    return frame;
}

inline void record_sample(Trace& trace, const ATV::cpu::Frame& frame,
                          int sample, const char* label) {
    using namespace ATV;
    if (sample < 0 || sample > kSamples)
        throw std::out_of_range("NDDO NVE sample index");
    const auto mass = frame.span(kAtom_mass);
    const auto vx = frame.span(kAtom_vx), vy = frame.span(kAtom_vy),
               vz = frame.span(kAtom_vz);
    double sum_mv2 = 0;
    for (int i = 0; i < frame.nlocal(); ++i)
        sum_mv2 += mass[i] * (vx[i] * vx[i] + vy[i] * vy[i] + vz[i] * vz[i]);
    const double expected_ke = .5 * units::md_mvv2e * sum_mv2;
    const double ke = frame.meta.get(kKinetic);
    const double potential = frame.meta.get(kEnergy);
    if (!std::isfinite(ke) || !std::isfinite(expected_ke)
        || !std::isfinite(potential) || std::abs(ke - expected_ke) > 1e-12) {
        std::fprintf(stderr,
                     "FAIL %s sample=%d time=%.3g fs potential=%.17g KE "
                     "got=%.17g want=%.17g tol=1e-12 eV\n",
                     label, sample, sample * kSampleInterval, potential, ke,
                     expected_ke);
        throw std::runtime_error(
            "NDDO NVE nonfinite energy or inconsistent kinetic energy");
    }
    trace.total_energy[sample] = potential + ke;
    trace.kinetic[sample] = ke;
    if (sample == 0)
        trace.initial_energy = potential + ke;
}

inline void finish_trace(Trace& trace, ATV::cpu::Frame frame,
                         const char* label) {
    trace.max_drift = trace.rms_drift = 0;
    for (int sample = 1; sample <= kSamples; ++sample) {
        const double drift = trace.total_energy[sample] - trace.initial_energy;
        if (!std::isfinite(drift))
            throw std::runtime_error("NDDO NVE nonfinite energy drift");
        trace.max_drift = std::max(trace.max_drift, std::abs(drift));
        trace.rms_drift += drift * drift;
    }
    trace.rms_drift = std::sqrt(trace.rms_drift / kSamples);
    trace.final_drift = trace.total_energy[kSamples] - trace.initial_energy;
    trace.final_frame = std::move(frame);
    std::printf("NVE %s T=%.1f fs samples=%d E0=%.14g K0=%.14g Efinal=%.14g "
                "Kfinal=%.14g eV; sampled max|deltaH|=%.9g RMS=%.9g "
                "endpoint=%+.9g eV\n",
                label, kDuration, kSamples,
                trace.total_energy[0] - trace.kinetic[0], trace.kinetic[0],
                trace.total_energy[kSamples] - trace.kinetic[kSamples],
                trace.kinetic[kSamples], trace.max_drift, trace.rms_drift,
                trace.final_drift);
}

inline bool check_halving(const Trace& coarse, const Trace& fine,
                          const char* label) {
    const double ratio = coarse.max_drift > 0
                             ? fine.max_drift / coarse.max_drift
                             : std::numeric_limits<double>::quiet_NaN();
    std::printf("NVE %s dt=%.3g/%.3g fs sampled max coarse=%.9g fine=%.9g eV "
                "ratio=%.9g; limits %.3g eV and ratio<=%.3g\n",
                label, kDt, kDt / 2, coarse.max_drift, fine.max_drift, ratio,
                kMaxDriftEv, kHalvingRatio);
    if (!(coarse.max_drift > 0) || !std::isfinite(coarse.max_drift)
        || !std::isfinite(fine.max_drift) || coarse.max_drift > kMaxDriftEv
        || fine.max_drift > kMaxDriftEv
        || !(fine.max_drift <= kHalvingRatio * coarse.max_drift)) {
        std::fprintf(stderr,
                     "FAIL %s NVE sampled energy "
                     "conservation/time-step-halving limits\n",
                     label);
        return false;
    }
    return true;
}

template <class Method, class Pin>
Trace run_cpu(const Pin& water, double dt) {
    using namespace ATV;
    const int chunk = static_cast<int>(std::lround(kSampleInterval / dt));
    if (chunk < 1 || std::abs(chunk * dt - kSampleInterval) > 1e-14)
        throw std::invalid_argument(
            "NDDO NVE sampling interval must be an integer number of steps");
    auto frame = initial_frame(water);
    Method method{0, 1, 400, 1e-10, 1e-12};
    Trace trace;
    const std::string label =
        std::string(water.name) + " CPU dt=" + std::to_string(dt);
    std::printf("NVE %s SCF max_iter=400 density_tol=1e-10 energy_tol=1e-12 eV "
                "steps=%d\n",
                label.c_str(), chunk * kSamples);
    {
        cpu::md::MD md;
        md.enable_timing(false);
        md.set_integrator(cpu::md::VelocityVerlet{dt});
        md.on(ATV::md::Phase::force).add_ref(method);
        md.build();
        md.run(frame, 0);
        const double energy = frame.meta.get(kEnergy);
        if (!std::isfinite(energy)
            || std::abs(energy - water.energy_ev) > 1e-6) {
            std::fprintf(
                stderr,
                "FAIL %s external initial E got=%.17g want=%.17g tol=1e-6 eV\n",
                label.c_str(), energy, water.energy_ev);
            throw std::runtime_error(
                "NDDO NVE external initial energy mismatch");
        }
        record_sample(trace, frame, 0, label.c_str());
        for (int sample = 1; sample <= kSamples; ++sample) {
            const auto advanced = md.run(frame, chunk);
            if (advanced != chunk || !method.last_result().converged) {
                std::fprintf(
                    stderr,
                    "FAIL %s sample=%d steps got=%lld want=%d converged=%d\n",
                    label.c_str(), sample, static_cast<long long>(advanced),
                    chunk, method.last_result().converged);
                throw std::runtime_error(
                    "NDDO NVE incomplete or unconverged sample");
            }
            record_sample(trace, frame, sample, label.c_str());
        }
    } // Release the live MD binding before moving its Frame.
    finish_trace(trace, std::move(frame), label.c_str());
    return trace;
}

template <class Method, class Pin>
int run_cpu_case(const Pin& water) {
    try {
        const auto coarse = run_cpu<Method>(water, kDt);
        const auto fine = run_cpu<Method>(water, kDt / 2);
        return check_halving(coarse, fine, water.name) ? 0 : 1;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "FAIL %s CPU NVE: %s\n", water.name, error.what());
        return 1;
    }
}
} // namespace atv_reg_nddo_nve
