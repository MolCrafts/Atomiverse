/// @file tests/cuda/pw/test_pw_si.cu
/// @brief CPU/CUDA parity test for the plane-wave SCF on toy Si.
#include "atomiverse/core/mat3.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/pw/pw_scf.h"
#include "atomiverse/cuda/pw/pw_scf.cuh"
#include <cmath>
#include <gtest/gtest.h>
#include <string>

using namespace ATV;

namespace {
cpu::Frame make_si_fcc(double a) {
    cpu::Frame f(2);
    f.atoms.type[0] = 14;
    f.atoms.type[1] = 14;
    f.atoms.mass[0] = 28.0855;
    f.atoms.mass[1] = 28.0855;
    f.atoms.x[0] = 0;
    f.atoms.y[0] = 0;
    f.atoms.z[0] = 0;
    f.atoms.x[1] = a / 4.0;
    f.atoms.y[1] = a / 4.0;
    f.atoms.z[1] = a / 4.0;
    double h = a / 2.0;
    Mat3 cell;
    cell(0, 0) = 0;
    cell(0, 1) = h;
    cell(0, 2) = h;
    cell(1, 0) = h;
    cell(1, 1) = 0;
    cell(1, 2) = h;
    cell(2, 0) = h;
    cell(2, 1) = h;
    cell(2, 2) = 0;
    f.box = cell;
    return f;
}
} // namespace

TEST(CudaPWScf, RunsAndFinite) {
    auto frame = make_si_fcc(5.43);
    auto r = cuda::pw::pw_scf(frame, "", 10.0, 2, 1e-6, 30);
    EXPECT_TRUE(std::isfinite(r.energy))
        << "Energy should be finite, got: " << r.energy;
    EXPECT_GT(r.iterations, 0);
}

// Parity runs on the real ONCV pseudopotential: the toy potential has a
// near-degenerate band-4/5 manifold at every k (gaps down to ~4 mHa), so
// which states the occupied cut selects — and hence the SCF fixed point —
// is rounding-sensitive and not comparable across backends. The LDA Si
// pseudo has a clean gap at the sampled k-points.
TEST(CudaPWScf, MatchesCpu) {
    auto frame = make_si_fcc(5.43);
    const std::string upf = ATV_PW_UPF_PATH;
    auto r_cpu = cpu::pw::pw_scf(frame, upf, 12.0, 2, 1e-6, 30);
    auto r_cuda = cuda::pw::pw_scf(frame, upf, 12.0, 2, 1e-6, 30);
    ASSERT_TRUE(r_cpu.converged);
    ASSERT_TRUE(r_cuda.converged);
    // Same SCF fixed point: backends differ only in Davidson arithmetic
    // order and the random 1e-4 startingwfc perturbation.
    EXPECT_NEAR(r_cpu.energy, r_cuda.energy, 1e-3)
        << "CPU " << r_cpu.energy << " Ry vs CUDA " << r_cuda.energy << " Ry";
}
