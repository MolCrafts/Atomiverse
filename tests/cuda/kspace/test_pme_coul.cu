/// @file tests/cuda/kspace/test_pme_coul.cu
/// @brief Validation of CUDA Particle-Mesh Ewald (CoulLong + KSpacePME)
///        against the NaCl Madelung constant. Needs ATV_BUILD_KSPACE.

#include <cmath>
#include <gtest/gtest.h>

#include "atomiverse/core/mat3.h"
#include "atomiverse/core/units.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cuda/kspace/pme.cuh"
#include "atomiverse/cuda/md/md.cuh"
#include "atomiverse/cuda/md/velocity_verlet.cuh"
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
                frame.atoms.x[idx] = i * d;
                frame.atoms.y[idx] = j * d;
                frame.atoms.z[idx] = k * d;
                frame.atoms.charge[idx] = ((i + j + k) % 2 == 0) ? 1.0 : -1.0;
                frame.atoms.mass[idx] = 1.0;
                frame.atoms.type[idx] = 0;
                frame.atoms.tag[idx] = idx;
                ++idx;
            }
    frame.box = Mat3::cubic(n * d);
    frame.atoms.map_clear();
    frame.atoms.map_set(0, frame.atoms.nlocal);
    return frame;
}

} // namespace

TEST(CudaPmeCoul, MadelungNaCl) {
    const double d = 2.82;
    const int n = 4;
    auto frame = make_rocksalt(d, n);

    const double cutoff = 1.9 * d;
    const double alpha =
        3.5 / cutoff; // moderate α so the grid resolves k-space

    cuda::md::MD sim;
    sim.set_neighbor(cuda::NeighborList{cutoff});
    sim.add_potential(cuda::CoulLong{alpha, cutoff});
    sim.set_integrator(cuda::md::VelocityVerlet{0.001});
    sim.on(md::Phase::force).add(cuda::KSpacePME{alpha, 0.5, 6}, 100);
    sim.build();
    sim.run(frame, 1);

    const int N = n * n * n;
    const double e_expected = -0.5 * N * kMadelungNaCl * ATV::kCoulomb / d;
    const double e = static_cast<double>(frame.scalars.energy);
    EXPECT_NEAR(e / e_expected, 1.0, 3e-3)
        << "E=" << e << " expected=" << e_expected;
}
