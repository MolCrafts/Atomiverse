// Complete SCF baselines migrated from the unit suite without changing gates.
// Opt-in executable only; never registered with CTest.
#include "atomiverse/core/frame_schema.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/planewave/pw_solver.h"
#include "atomiverse/cpu/xc/xc_functional.h"
#include "atomiverse/cuda/planewave/pw_solver.cuh"
#include <memory>

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

/// @brief Cheap H GTH plane-wave settings shared by the CPU and CUDA twins.
/// @param xc Exchange–correlation functional.
/// @return Method with ecut 5 Ry, Γ-only mesh, loose SCF tolerance.
template <typename Method>
Method h_gth_method(cpu::xc::XCType xc = cpu::xc::XCType::LSDA) {
    Method m{std::make_shared<cpu::pw::GthPseudopotential>(molrs::Element::H,
                                                           "GTH-PBE-q1"),
             /*ecut_ry=*/5.0,
             /*k_mesh=*/1, xc};
    m.tol = 1e-4;
    m.max_iter = 12;
    m.mix_beta = 0.5;
    return m;
}

} // namespace

TEST(CudaPWCellStress, H2ClosedShellCpuTwin) {
    auto frame = make_h2_cubic(6.0);

    auto cpu_m = h_gth_method<cpu::pw::PW>();
    cpu_m.build(frame);
    cpu::Frame frame_copy = frame; // stress() writes kStress/kEnergy
    const auto s_cpu = cpu_m.stress(frame_copy); // σ [eV/Å³]

    auto cuda_m = h_gth_method<cuda::pw::PW>();
    cuda_m.build(frame);
    double e_ry = 0.0;
    const auto s_cuda = cuda_m.stress(frame);
    e_ry = cuda_m.last_result().energy;

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
        max_abs =
            std::max(max_abs, std::abs(s_cuda[static_cast<std::size_t>(k)]
                                       - s_cpu[static_cast<std::size_t>(k)]));
    }
    EXPECT_LT(max_abs, kTol)
        << "max|σ_cuda-σ_cpu|=" << max_abs << " cpu=(" << s_cpu[0] << ","
        << s_cpu[1] << "," << s_cpu[2] << ") cuda=(" << s_cuda[0] << ","
        << s_cuda[1] << "," << s_cuda[2] << ")";
}

TEST(CudaPWCellStress, HSmearRestrictedFinite) {
    auto frame = make_h_cubic(6.0);
    auto cuda_m = h_gth_method<cuda::pw::PW>();
    cuda_m.smearing = ::ATV::electronic::Smearing::fermi_dirac;
    cuda_m.degauss = 0.02;
    cuda_m.extra_bands = 2;
    cuda_m.max_iter = 12;
    cuda_m.build(frame);
    double e_ry = 0.0;
    const auto s_cuda = cuda_m.stress(frame);
    e_ry = cuda_m.last_result().energy;
    for (int k = 0; k < 6; ++k)
        EXPECT_TRUE(std::isfinite(s_cuda[static_cast<std::size_t>(k)])) << k;
    EXPECT_TRUE(std::isfinite(e_ry));
}

TEST(CudaPWCellStress, H2ClosedShellPbeCpuTwin) {
    auto frame = make_h2_cubic(6.0);

    auto cpu_m = h_gth_method<cpu::pw::PW>(cpu::xc::XCType::PBE);
    cpu_m.build(frame);
    cpu::Frame frame_copy = frame; // stress() writes kStress/kEnergy
    const auto s_cpu = cpu_m.stress(frame_copy); // σ [eV/Å³]

    auto cuda_m = h_gth_method<cuda::pw::PW>(cpu::xc::XCType::PBE);
    cuda_m.build(frame);
    double e_ry = 0.0;
    const auto s_cuda = cuda_m.stress(frame);
    e_ry = cuda_m.last_result().energy;

    for (int k = 0; k < 6; ++k) {
        EXPECT_TRUE(std::isfinite(s_cuda[static_cast<std::size_t>(k)])) << k;
        EXPECT_TRUE(std::isfinite(s_cpu[static_cast<std::size_t>(k)])) << k;
    }
    EXPECT_TRUE(std::isfinite(e_ry));

    constexpr double kTol = 5.0e-2;
    double max_abs = 0.0;
    for (int k = 0; k < 6; ++k) {
        max_abs =
            std::max(max_abs, std::abs(s_cuda[static_cast<std::size_t>(k)]
                                       - s_cpu[static_cast<std::size_t>(k)]));
    }
    EXPECT_LT(max_abs, kTol)
        << "PBE max|σ_cuda-σ_cpu|=" << max_abs << " cpu=(" << s_cpu[0] << ","
        << s_cpu[1] << "," << s_cpu[2] << ") cuda=(" << s_cuda[0] << ","
        << s_cuda[1] << "," << s_cuda[2] << ")";
}
