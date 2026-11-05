/// @file tests/cuda/longrange/test_kspace_pppm.cu
/// @brief cuda::KSpacePPPM (Hockney–Eastwood ik) reciprocal energy, forces
///        and virial on a small charged periodic cell against the direct
///        reciprocal Ewald sum (cuda::KSpaceEwald at 1e-12), orthorhombic and
///        triclinic, neutral and net-charged.
///
/// Both solvers are driven directly with the same α, the same device Frame
/// layout and a published device Box, so the self energy and the
/// neutralizing-background term are common; what differs is only the mesh
/// discretisation, which at α·h ≈ 0.11 and order 7 is ~1e-10 relative in
/// FP64 (see the tolerance below).
#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <type_traits>
#include <vector>

#include <cuda_runtime.h>

#include "atomiverse/core/frame_schema.h"
#include "atomiverse/core/system_scalars.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/frame/transfer.h"
#include "atomiverse/cuda/component/context.cuh"
#include "atomiverse/cuda/frame/precision.cuh"
#include "atomiverse/cuda/frame/transfer.cuh"
#include "atomiverse/cuda/longrange/ewald.cuh"
#include "atomiverse/cuda/longrange/pme.cuh"

using namespace ATV;

namespace {

constexpr double kAlpha = 0.32;   // 1/Å
constexpr double kSpacing = 0.35; // Å
constexpr int kOrder = 7;
constexpr int kAtoms = 12;
constexpr bool kFloatMesh = std::is_same_v<cuda::force_t, float>;

struct Result {
    double energy = 0;
    double virial[6] = {};
    std::vector<double> force; ///< x, y, z planes.
};

/// @brief Evaluate one reciprocal solver once on a fixed charged cell.
template <class Solver>
Result evaluate(Solver solver, const std::array<double, 9>& cell,
                bool neutral) {
    cpu::Frame host(kAtoms);
    Mat3 h;
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            h(i, j) = cell[3 * i + j];
    host.box.set_cell(h);
    for (int i = 0; i < kAtoms; ++i) {
        const double s[3] = {std::fmod(0.137 + 0.618034 * i, 1.0),
                             std::fmod(0.291 + 0.414214 * i, 1.0),
                             std::fmod(0.533 + 0.732051 * i, 1.0)};
        const ColKey<StorageReal>* keys[3] = {&kAtom_x, &kAtom_y, &kAtom_z};
        for (int a = 0; a < 3; ++a) {
            double r = 0;
            for (int k = 0; k < 3; ++k)
                r += cell[3 * a + k] * s[k];
            host.span(*keys[a])[i] = r;
        }
        host.span(kAtom_charge)[i] =
            (i % 2 ? -0.8 : 0.8) + (neutral ? 0.0 : 0.05 * (i % 3));
        host.span(kAtom_mass)[i] = 1;
        host.span(kAtom_tag)[i] = i;
        host.span(kAtom_type)[i] = 0;
    }
    host.rebuild_tag_index(0, kAtoms);

    cuda::Frame device = cpu::to_cuda(host);
    cudaStream_t stream = nullptr;
    EXPECT_EQ(cudaStreamCreate(&stream), cudaSuccess);
    cuda::Box* resident = nullptr;
    EXPECT_EQ(cudaMalloc(&resident, sizeof(cuda::Box)), cudaSuccess);
    cuda::Context ctx{};
    ctx.stream = stream;
    cuda::Box box{};
    box.assign_cell(cell.data());
    box.revision = 1;
    EXPECT_EQ(cudaMemcpy(resident, &box, sizeof(box), cudaMemcpyHostToDevice),
              cudaSuccess);
    box.resident = resident;
    ctx.host_box = box;

    solver.build({ctx, device});
    EXPECT_EQ(cudaMemset(device.scalars, 0, sizeof(SystemScalars)),
              cudaSuccess);
    for (auto* f : {device.atoms.fx, device.atoms.fy, device.atoms.fz})
        EXPECT_EQ(cudaMemset(f, 0, kAtoms * sizeof(cuda::fixed_t)),
                  cudaSuccess);
    solver.launch({ctx, device, 0, 0});
    EXPECT_EQ(cudaStreamSynchronize(stream), cudaSuccess);

    Result r;
    SystemScalars scalars{};
    EXPECT_EQ(cudaMemcpy(&scalars, device.scalars, sizeof(scalars),
                         cudaMemcpyDeviceToHost),
              cudaSuccess);
    r.energy = scalars.energy;
    for (int k = 0; k < 6; ++k)
        r.virial[k] = scalars.stress[k];
    std::vector<cuda::fixed_t> words(kAtoms);
    for (auto* f : {device.atoms.fx, device.atoms.fy, device.atoms.fz}) {
        EXPECT_EQ(cudaMemcpy(words.data(), f, kAtoms * sizeof(cuda::fixed_t),
                             cudaMemcpyDeviceToHost),
                  cudaSuccess);
        for (auto w : words)
            r.force.push_back(cuda::decode_force<double>(w));
    }
    solver.destroy({ctx, device});
    cudaFree(resident);
    cudaStreamDestroy(stream);
    cuda::destroy(device);
    return r;
}

void check_against_ewald(const std::array<double, 9>& cell, bool neutral,
                         const char* label) {
    const Result ewald =
        evaluate(cuda::KSpaceEwald{kAlpha, 1.0e-12}, cell, neutral);
    const Result pppm =
        evaluate(cuda::KSpacePPPM{kAlpha, kSpacing, kOrder}, cell, neutral);

    double fmax = 0, wmax = 0, fdiff = 0, wdiff = 0;
    for (std::size_t i = 0; i < ewald.force.size(); ++i) {
        fmax = std::max(fmax, std::abs(ewald.force[i]));
        fdiff = std::max(fdiff, std::abs(pppm.force[i] - ewald.force[i]));
    }
    for (int k = 0; k < 6; ++k) {
        wmax = std::max(wmax, std::abs(ewald.virial[k]));
        wdiff = std::max(wdiff, std::abs(pppm.virial[k] - ewald.virial[k]));
    }
    std::printf("%s E_ewald=%.12g E_pppm=%.12g |dF|max=%.3g (|F|max=%.3g) "
                "|dW|max=%.3g (|W|max=%.3g)\n",
                label, ewald.energy, pppm.energy, fdiff, fmax, wdiff, wmax);

    // Not vacuous: the reference has a sizeable energy, force and virial.
    ASSERT_GT(std::abs(ewald.energy), 1e-2) << label;
    ASSERT_GT(fmax, 1e-2) << label;
    ASSERT_GT(wmax, 1e-2) << label;

    // FP64: mesh discretisation only — measured on GH200 (CUDA 13.2) at
    // ≤ 5e-11 relative in energy and ≤ 5e-10 of the largest force / virial
    // component on all three cells, so 1e-8 leaves a 20× margin. FP32
    // mesh/spline: single-precision spreading and FFT over ~10⁵ points.
    const double rel = kFloatMesh ? 2.0e-4 : 1.0e-8;
    EXPECT_NEAR(pppm.energy, ewald.energy, rel * std::abs(ewald.energy))
        << label;
    for (std::size_t i = 0; i < ewald.force.size(); ++i)
        EXPECT_NEAR(pppm.force[i], ewald.force[i], rel * fmax)
            << label << " force component " << i;
    for (int k = 0; k < 6; ++k)
        EXPECT_NEAR(pppm.virial[k], ewald.virial[k], rel * wmax)
            << label << " virial component " << k;
}

const std::array<double, 9> kOrtho = {11.5, 0, 0, 0, 11.5, 0, 0, 0, 15.5};
const std::array<double, 9> kTriclinic = {14.5, 0,    0,    1.25, 12.5,
                                          0,    0.75, -1.0, 14.5};

} // namespace

TEST(CudaKSpacePppm, OrthorhombicNeutralMatchesEwald) {
    check_against_ewald(kOrtho, true, "pppm-ortho-neutral");
}

TEST(CudaKSpacePppm, OrthorhombicChargedMatchesEwald) {
    check_against_ewald(kOrtho, false, "pppm-ortho-charged");
}

TEST(CudaKSpacePppm, TriclinicChargedMatchesEwald) {
    check_against_ewald(kTriclinic, false, "pppm-triclinic-charged");
}
