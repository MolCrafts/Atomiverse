/// @file tests/cuda/planewave/test_pw_cell_stress.cu
/// @brief Unit gates for CUDA plane-wave Nielsen–Martin cell stress.
///
/// UNIT SCOPE: empty-frame / open-shell reject; CUDA↔CPU parity on a cheap
/// closed-shell H₂ cell. Full Si dual-gate lives in
/// regressions/pw/cuda-pw-stress — never under ctest as an e2e product.

#include "atomiverse/core/frame_schema.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/planewave/pw_solver.h"
#include "atomiverse/cpu/scf/xc_functional.h"
#include "atomiverse/cuda/planewave/pw_solver.cuh"

#include <cmath>
#include <gtest/gtest.h>
#include <stdexcept>

using namespace ATV;

namespace {

cpu::Frame make_h_cubic(double L) {
    cpu::Frame f(1);
    f.span(kAtom_type)[0] = 1;
    f.span(kAtom_mass)[0] = 1.008;
    f.span(kAtom_x)[0] = 0.0;
    f.span(kAtom_y)[0] = 0.0;
    f.span(kAtom_z)[0] = 0.0;
    f.box = cpu::Box::cubic(L);
    return f;
}

/// Closed-shell H₂ (2× GTH-PBE-q1 → 2 valence e) in a cubic box.
cpu::Frame make_h2_cubic(double L) {
    cpu::Frame f(2);
    for (int i = 0; i < 2; ++i) {
        f.span(kAtom_type)[i] = 1;
        f.span(kAtom_mass)[i] = 1.008;
    }
    f.span(kAtom_x)[0] = 0.0;
    f.span(kAtom_y)[0] = 0.0;
    f.span(kAtom_z)[0] = 0.0;
    f.span(kAtom_x)[1] = 0.75; // Å — short bond, nonzero stress
    f.span(kAtom_y)[1] = 0.0;
    f.span(kAtom_z)[1] = 0.0;
    f.box = cpu::Box::cubic(L);
    return f;
}

void fill_cuda_method(cuda::pw::PW& m) {
    m.gth_name = "GTH-PBE-q1";
    m.ecut_ry = 5.0;
    m.k_mesh = 1;
    m.tol = 1e-4;
    m.max_iter = 12;
    m.mix_beta = 0.5;
    m.xc = cpu::scf::XCType::LSDA;
}

void fill_cpu_method(cpu::pw::PW& m) {
    m.gth_name = "GTH-PBE-q1";
    m.ecut_ry = 5.0;
    m.k_mesh = 1;
    m.tol = 1e-4;
    m.max_iter = 12;
    m.mix_beta = 0.5;
    m.xc = cpu::scf::XCType::LSDA;
}

} // namespace

TEST(CudaPWCellStress, EmptyFrameThrows) {
    cpu::Frame f(0);
    f.box = cpu::Box::cubic(6.0);
    cuda::pw::PW method;
    fill_cuda_method(method);
    EXPECT_THROW(cuda::pw::pw_stress(f, method), std::invalid_argument);
}

TEST(CudaPWCellStress, OpenShellThrows) {
    // Single H: 1 valence e — CUDA density path is closed-shell only.
    auto f = make_h_cubic(6.0);
    cuda::pw::PW method;
    fill_cuda_method(method);
    method.build(f);
    EXPECT_THROW(cuda::pw::pw_stress(f, method), std::invalid_argument);
}

TEST(CudaPWCellStress, H2ClosedShellCpuTwin) {
    auto frame = make_h2_cubic(6.0);

    cpu::pw::PW cpu_m;
    fill_cpu_method(cpu_m);
    cpu_m.build(frame);
    const auto s_cpu = cpu::pw::pw_stress(frame, cpu_m);

    cuda::pw::PW cuda_m;
    fill_cuda_method(cuda_m);
    cuda_m.build(frame);
    double e_ry = 0.0;
    const auto s_cuda = cuda::pw::pw_stress(frame, cuda_m, &e_ry);

    for (int k = 0; k < 6; ++k) {
        EXPECT_TRUE(std::isfinite(s_cuda[static_cast<std::size_t>(k)])) << k;
        EXPECT_TRUE(std::isfinite(s_cpu[static_cast<std::size_t>(k)])) << k;
    }
    EXPECT_TRUE(std::isfinite(e_ry));

    // Dual-gate vs same-settings CPU analytic stress.
    // Cheap ecut/tol: design-time 5e-2 eV/Å³ (matches Si regression window).
    constexpr double kTol = 5.0e-2;
    double max_abs = 0.0;
    for (int k = 0; k < 6; ++k) {
        max_abs = std::max(
            max_abs, std::abs(s_cuda[static_cast<std::size_t>(k)]
                              - s_cpu[static_cast<std::size_t>(k)]));
    }
    EXPECT_LT(max_abs, kTol) << "max|σ_cuda-σ_cpu|=" << max_abs
                             << " cpu=(" << s_cpu[0] << "," << s_cpu[1] << ","
                             << s_cpu[2] << ") cuda=(" << s_cuda[0] << ","
                             << s_cuda[1] << "," << s_cuda[2] << ")";
}
