/// @file regressions/forcefield/cuda-pme-nacl/reg_cuda-pme-nacl.cu
/// @brief Validation of CUDA Particle-Mesh Ewald (CoulLong + KSpacePME)
///        against the NaCl Madelung constant. Needs ATV_BUILD_KSPACE.

#include <cmath>
#include <cstdio>
#include <exception>

#include "atomiverse/core/mat3.h"
#include "atomiverse/core/units.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cuda/dynamics/md.cuh"
#include "atomiverse/cuda/dynamics/velocity_verlet.cuh"
#include "atomiverse/cuda/longrange/pme.cuh"
#include "atomiverse/cuda/neighbor/neighbor_list.cuh"
#include "atomiverse/cuda/potentials/coul_long.cuh"

using namespace ATV;

namespace {

constexpr double kMadelungNaCl = 1.747564594633182;

cpu::Frame make_rocksalt(double d, int n) {
    const int N = n * n * n;
    cpu::Frame frame(N);
    int idx = 0;
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            for (int k = 0; k < n; ++k) {
                frame.span(kAtom_x)[idx] = i * d;
                frame.span(kAtom_y)[idx] = j * d;
                frame.span(kAtom_z)[idx] = k * d;
                frame.span(kAtom_charge)[idx] =
                    ((i + j + k) % 2 == 0) ? 1.0 : -1.0;
                frame.span(kAtom_mass)[idx] = 1.0;
                frame.span(kAtom_type)[idx] = 0;
                frame.span(kAtom_tag)[idx] = idx;
                ++idx;
            }
    frame.box = cpu::Box::cubic(n * d);
    return frame;
}

} // namespace

int main() try {
    const double d = 2.82;
    const int n = 4;
    auto frame = make_rocksalt(d, n);

    const double cutoff = 1.9 * d;
    const double alpha =
        3.5 / cutoff; // moderate α so the grid resolves k-space

    cuda::md::MD sim;
    sim.set_neighbor(cuda::NeighborList{cutoff});
    sim.set_evaluator(cuda::CoulLong{alpha, cutoff});
    sim.set_integrator(cuda::md::VelocityVerlet{0.001});
    sim.on(md::Phase::force).add(cuda::KSpacePME{alpha, 0.5, 6}, 100);
    sim.build();
    sim.run(frame, 1);

    const int N = n * n * n;
    const double e_expected =
        -0.5 * N * kMadelungNaCl * ATV::units::coulomb / d;
    const double e = static_cast<double>(frame.meta.get(kEnergy));
    const double error = std::abs(e / e_expected - 1.0);
    std::printf(
        "CUDA PME NaCl: E=%.12f eV expected=%.12f eV relative_error=%.6g\n", e,
        e_expected, error);
    return error <= 3e-3 ? 0 : 1;
} catch (const std::exception& error) {
    std::fprintf(stderr, "cuda-pme-nacl: %s\n", error.what());
    return 1;
}
