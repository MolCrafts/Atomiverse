/// @file tests/cuda/longrange/test_pppm_stencil.cu
/// @brief cuda::KSpacePPPM energy, forces and virial against the literal
///        LAMMPS make_rho / poisson_ik / fieldforce_ik reference
///        (tests/core/pppm_lammps_reference.h) for odd and even orders: the
///        centred charge-assignment stencil and, on even meshes, the Nyquist
///        planes of LAMMPS's full complex transform.
///
/// Same systems, meshes and reference as the CPU `PppmStencil` tests in
/// tests/cpu/longrange/test_pppm_coul.cpp, so both backends are pinned to one
/// placement. The odd 9×9×15 mesh has no Nyquist planes; the even 8×8×12 one
/// has all three.
#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <string>
#include <type_traits>
#include <vector>

#include <cuda_runtime.h>

#include "atomiverse/core/frame_schema.h"
#include "atomiverse/core/longrange_math.h"
#include "atomiverse/core/system_scalars.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/frame/transfer.h"
#include "atomiverse/cuda/component/context.cuh"
#include "atomiverse/cuda/frame/precision.cuh"
#include "atomiverse/cuda/frame/transfer.cuh"
#ifdef ATV_HAS_KSPACE
#include "atomiverse/cuda/longrange/pme.cuh"
#endif
#include "core/pppm_lammps_reference.h"

#ifdef ATV_HAS_KSPACE
using namespace ATV;
namespace ref = ATV::test::lammps_pppm;

namespace {

constexpr int kAtoms = 10;
constexpr double kAlpha = 0.9;
constexpr bool kFloatMesh = std::is_same_v<cuda::force_t, float>;

struct Run {
    double energy = 0.0;       ///< reciprocal energy, self energy removed [eV]
    std::vector<double> force; ///< 3·n atom-major [eV/Å]
    double virial[6] = {};     ///< xx yy zz xy xz yz [eV]
    std::array<int, 3> mesh{};
};

Run run_cuda(const std::array<double, 9>& cell, const std::vector<double>& frac,
             const std::vector<double>& q, int order, double spacing) {
    cpu::Frame host(kAtoms);
    host.set_nlocal(kAtoms);
    Mat3 h;
    for (int r = 0; r < 3; ++r)
        for (int c = 0; c < 3; ++c)
            h(r, c) = cell[3 * r + c];
    host.box.set_cell(h);
    const ColKey<StorageReal>* axis[3] = {&kAtom_x, &kAtom_y, &kAtom_z};
    for (int i = 0; i < kAtoms; ++i) {
        for (int r = 0; r < 3; ++r) {
            double x = 0.0;
            for (int c = 0; c < 3; ++c)
                x += cell[3 * r + c] * frac[3 * i + c];
            host.span(*axis[r])[i] = x;
        }
        host.span(kAtom_charge)[i] = q[i];
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

    cuda::KSpacePPPM solver(kAlpha, spacing, order);
    solver.build({ctx, device});
    EXPECT_EQ(cudaMemset(device.scalars, 0, sizeof(SystemScalars)),
              cudaSuccess);
    for (auto* f : {device.atoms.fx, device.atoms.fy, device.atoms.fz})
        EXPECT_EQ(cudaMemset(f, 0, kAtoms * sizeof(cuda::fixed_t)),
                  cudaSuccess);
    solver.launch({ctx, device, 0, 0});
    EXPECT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
    SystemScalars scalars{};
    EXPECT_EQ(cudaMemcpy(&scalars, device.scalars, sizeof(scalars),
                         cudaMemcpyDeviceToHost),
              cudaSuccess);
    // The device holds force_t charges; remove the self energy it added.
    std::vector<cuda::force_t> charge(kAtoms);
    EXPECT_EQ(cudaMemcpy(charge.data(), device.atoms.charge,
                         kAtoms * sizeof(cuda::force_t),
                         cudaMemcpyDeviceToHost),
              cudaSuccess);
    double qsq = 0.0;
    for (const cuda::force_t c : charge)
        qsq += double(c) * double(c);
    Run out;
    out.energy =
        scalars.energy - longrange::self_energy_prefactor(kAlpha, 1.0) * qsq;
    out.mesh = solver.mesh();
    for (int c = 0; c < 6; ++c)
        out.virial[c] = scalars.stress[c];
    std::vector<cuda::fixed_t> words(kAtoms);
    std::vector<double> planes[3];
    cuda::fixed_t* cols[3] = {device.atoms.fx, device.atoms.fy,
                              device.atoms.fz};
    for (int a = 0; a < 3; ++a) {
        EXPECT_EQ(cudaMemcpy(words.data(), cols[a],
                             kAtoms * sizeof(cuda::fixed_t),
                             cudaMemcpyDeviceToHost),
                  cudaSuccess);
        for (auto w : words)
            planes[a].push_back(cuda::decode_force<double>(w));
    }
    out.force.resize(3 * kAtoms);
    for (int i = 0; i < kAtoms; ++i)
        for (int a = 0; a < 3; ++a)
            out.force[3 * i + a] = planes[a][i];
    solver.destroy({ctx, device});
    cudaFree(resident);
    cudaStreamDestroy(stream);
    cuda::destroy(device);
    return out;
}

void check_stencil(const std::array<double, 9>& cell, int order, double spacing,
                   const std::array<int, 3>& mesh) {
    SCOPED_TRACE("order " + std::to_string(order) + ", mesh "
                 + std::to_string(mesh[0]) + "x" + std::to_string(mesh[1]) + "x"
                 + std::to_string(mesh[2]));
    std::vector<double> frac(3 * kAtoms), q(kAtoms);
    for (int i = 0; i < kAtoms; ++i) {
        frac[3 * i + 0] = std::fmod(0.137 + 0.618034 * i, 1.0);
        frac[3 * i + 1] = std::fmod(0.291 + 0.414214 * i, 1.0);
        frac[3 * i + 2] = std::fmod(0.533 + 0.732051 * i, 1.0);
        q[i] = i % 2 ? -0.8 : 0.8;
    }
    const Run gpu = run_cuda(cell, frac, q, order, spacing);
    ASSERT_EQ(gpu.mesh, mesh);
    const int K[3] = {mesh[0], mesh[1], mesh[2]};
    const ref::Result lammps =
        ref::solve(cell.data(), frac, q, K, kAlpha, order);
    // FP64: fixed-point force columns and atomics; FP32 mesh: float splines.
    const double e_tol = kFloatMesh ? 2e-5 : 1e-11;
    const double f_tol = kFloatMesh ? 2e-5 : 1e-10;
    double fmax = 0.0, dfmax = 0.0;
    for (int k = 0; k < 3 * kAtoms; ++k) {
        fmax = std::max(fmax, std::abs(lammps.force[k]));
        dfmax = std::max(dfmax, std::abs(gpu.force[k] - lammps.force[k]));
    }
    EXPECT_NEAR(gpu.energy, lammps.energy, e_tol * std::abs(lammps.energy));
    EXPECT_LT(dfmax, f_tol * fmax);
    double wmax = 0.0, dwmax = 0.0;
    for (int c = 0; c < 6; ++c) {
        wmax = std::max(wmax, std::abs(lammps.virial[c]));
        dwmax = std::max(dwmax, std::abs(gpu.virial[c] - lammps.virial[c]));
    }
    EXPECT_GT(wmax, 0.0);
    EXPECT_LT(dwmax, e_tol * wmax);
    // The one-sided smooth-PME placement is ≥ 5e-4 away for odd orders here,
    // so the tolerances above separate the two stencils on either precision.
    if (order % 2) {
        const ref::Result one_sided =
            ref::solve(cell.data(), frac, q, K, kAlpha, order, 0.5 * order);
        EXPECT_GT(std::abs(one_sided.energy - lammps.energy),
                  20 * e_tol * std::abs(lammps.energy));
    }
}

} // namespace

TEST(CudaPppmStencil, OrthorhombicMatchesLammpsMakeRhoAndFieldforce) {
    const std::array<double, 9> cell = {6.3, 0, 0, 0, 6.3, 0, 0, 0, 10.4};
    for (int order : {3, 4, 5, 6, 7})
        check_stencil(cell, order, 0.7, {9, 9, 15});
}

TEST(CudaPppmStencil, TriclinicMatchesLammpsMakeRhoAndFieldforce) {
    const std::array<double, 9> cell = {6.3, 0.7, -0.6, 0,   6.25,
                                        0.9, 0,   0,    10.4};
    for (int order : {4, 5, 7})
        check_stencil(cell, order, 0.7, {9, 9, 15});
}

// LAMMPS's ik derivative along a Nyquist index cancels in the real part of
// its complex transform; the folded r2c tables reproduce that and the
// conjugate pair's energy and virial.
TEST(CudaPppmStencil, EvenMeshNyquistPlanesMatchLammpsPoissonIk) {
    const std::array<double, 9> cell = {6.4, 0, 0, 0, 6.4, 0, 0, 0, 9.6};
    for (int order : {3, 4, 5, 6, 7})
        check_stencil(cell, order, 0.8, {8, 8, 12});
}

TEST(CudaPppmStencil, EvenMeshTriclinicNyquistPlanesMatchLammpsPoissonIk) {
    const std::array<double, 9> cell = {6.4, 0.7, -0.6, 0,  6.35,
                                        0.9, 0,   0,    9.5};
    for (int order : {4, 5, 6})
        check_stencil(cell, order, 0.8, {8, 8, 12});
}

TEST(CudaPppmStencil, ExplicitMeshAndAccuracyConstructionMatchTheCpuRule) {
    // kspace_modify mesh 20 21 22 → 20 24 24 (2·3·5 sizes), as on the CPU.
    longrange::PppmAccuracyRequest req;
    req.accuracy = 1e-4;
    req.cutoff = 6.0;
    req.order = 6;
    const double cell[9] = {12, 0, 0, 0, 13, 0, 0, 0, 14};
    for (int k = 0; k < 9; ++k)
        req.cell[k] = cell[k];
    req.natoms = 40;
    req.qsqsum = 40.0;
    const auto chosen = longrange::pppm_from_accuracy(req);
    // LAMMPS PPPM::init for this request (test_pppm_accuracy.cpp ortho_o6).
    EXPECT_EQ(chosen.mesh[0], 12);
    EXPECT_EQ(chosen.mesh[1], 12);
    EXPECT_EQ(chosen.mesh[2], 12);
    cuda::KSpacePPPM solver = cuda::KSpacePPPM::from_accuracy(req);
    EXPECT_EQ(solver.long_range_contract().alpha, AccReal(chosen.alpha));
    EXPECT_THROW((cuda::KSpacePPPM{0.3, std::array<int, 3>{8, 0, 8}}),
                 std::invalid_argument);
}
#endif
