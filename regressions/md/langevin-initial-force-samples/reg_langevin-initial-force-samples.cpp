/// @file regressions/md/langevin-initial-force-samples/reg_langevin-initial-force-samples.cpp
/// @brief One-step Langevin white-noise second moment, oracle: literature.
#include "atomiverse/core/frame_schema.h"
#include "atomiverse/core/units.h"
#include "atomiverse/cpu/dynamics/fix_langevin.h"
#include "atomiverse/cpu/dynamics/md.h"
#include "atomiverse/cpu/dynamics/velocity_verlet.h"
#include "atomiverse/cpu/frame/frame.h"

#ifdef ATV_HAS_CUDA
#include "atomiverse/cuda/dynamics/fix_langevin.cuh"
#include "atomiverse/cuda/dynamics/md.cuh"
#include "atomiverse/cuda/dynamics/velocity_verlet.cuh"
#endif

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <initializer_list>
#include <stdexcept>
#include <string>

using namespace ATV;

namespace {

constexpr int kAtoms = 16384;
constexpr double kMass = 12.0;         // amu
constexpr double kTemperature = 300.0; // K
constexpr double kDamp = 100.0;        // fs
constexpr double kDt = 0.5;            // fs
constexpr std::uint32_t kSeed = 173;
constexpr double kRatioTolerance = 0.04;

cpu::Frame initial_atoms() {
    cpu::Frame frame(kAtoms);
    frame.box = cpu::Box::cubic(200.0);
    auto x = frame.span(kAtom_x);
    auto y = frame.span(kAtom_y);
    auto z = frame.span(kAtom_z);
    auto vx = frame.span(kAtom_vx);
    auto vy = frame.span(kAtom_vy);
    auto vz = frame.span(kAtom_vz);
    auto mass = frame.span(kAtom_mass);
    auto tag = frame.span(kAtom_tag);
    for (int i = 0; i < kAtoms; ++i) {
        x[i] = 2.5 + 5.0 * (i % 32);
        y[i] = 2.5 + 5.0 * ((i / 32) % 32);
        z[i] = 2.5 + 5.0 * (i / 1024);
        vx[i] = vy[i] = vz[i] = 0.0;
        mass[i] = kMass;
        tag[i] = i;
    }
    return frame;
}

void check_second_moment(const cpu::Frame& frame, const char* backend) {
    const auto vx = frame.span(kAtom_vx);
    const auto vy = frame.span(kAtom_vy);
    const auto vz = frame.span(kAtom_vz);
    double sum_squares = 0.0;
    for (int i = 0; i < kAtoms; ++i) {
        for (double v : {double(vx[i]), double(vy[i]), double(vz[i])}) {
            if (!std::isfinite(v))
                throw std::runtime_error(std::string(backend)
                                         + ": non-finite velocity");
            sum_squares += v * v;
        }
    }
    // Einstein fluctuation-dissipation amplitude + two VV half-kicks:
    // v1 = c*((1-a)*xi0 + xi1), E[xi0*xi1]=0, E[xi0^2]=E[xi1^2]=1.
    // See REFERENCE.md: this is an initial-step pin, not an equilibrium claim.
    const double c_squared = kDt * units::kB_ev_per_k * kTemperature
                             / (2.0 * kMass * units::mass_v2_to_energy * kDamp);
    const double a = kDt / (2.0 * kDamp);
    const double expected = c_squared * ((1.0 - a) * (1.0 - a) + 1.0);
    const double observed = sum_squares / (3.0 * kAtoms);
    const double ratio = observed / expected;
    std::printf("%s Langevin first-step samples=%d second_moment=%.16g "
                "expected=%.16g ratio=%.9g allowed=[0.96,1.04]\n",
                backend, 3 * kAtoms, observed, expected, ratio);
    if (!std::isfinite(ratio) || std::abs(ratio - 1.0) > kRatioTolerance)
        throw std::runtime_error(std::string(backend)
                                 + ": initial Langevin second moment failed");
}

} // namespace

int main() try {
    {
        auto frame = initial_atoms();
        cpu::md::MD simulation;
        simulation.set_integrator(cpu::md::VelocityVerlet{kDt});
        simulation.set_thermostat(
            cpu::md::FixLangevin{kTemperature, -1.0, kDamp, kDt, kSeed});
        simulation.enable_timing(false);
        simulation.build();
        if (simulation.run(frame, 1) != 1)
            throw std::runtime_error("CPU did not execute exactly one step");
        check_second_moment(frame, "CPU");
    }
#ifdef ATV_HAS_CUDA
    {
        auto frame = initial_atoms();
        cuda::md::MD simulation;
        simulation.set_integrator(cuda::md::VelocityVerlet{kDt});
        simulation.set_thermostat(
            cuda::md::FixLangevin{kTemperature, -1.0, kDamp, kDt, kSeed});
        simulation.build();
        simulation.run(frame, 1);
        check_second_moment(frame, "CUDA");
    }
#endif
    return 0;
} catch (const std::exception& error) {
    std::fprintf(stderr, "langevin-initial-force-samples: %s\n", error.what());
    return 1;
}
