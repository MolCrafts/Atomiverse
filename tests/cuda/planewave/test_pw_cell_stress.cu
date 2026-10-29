/// @file tests/cuda/planewave/test_pw_cell_stress.cu
/// @brief Admission gates for CUDA plane-wave strain-FD cell stress.
///
/// Empty-frame, occupation and thermal-width rejection only; no SCF.
/// Complete stress baselines are opt-in under tools/diagnostics/pbe-family.

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

TEST(CudaPWCellStress, EmptyFrameThrows) {
    cpu::Frame f(0);
    f.box = cpu::Box::cubic(6.0);
    auto method = h_gth_method<cuda::pw::PW>();
    EXPECT_THROW(method.stress(f), std::invalid_argument);
}

TEST(CudaPWCellStress, OpenShellThrows) {
    // Single H needs spin or smearing; this fixture requests fixed RKS.
    auto f = make_h_cubic(6.0);
    auto method = h_gth_method<cuda::pw::PW>();
    method.build(f);
    EXPECT_THROW(method.stress(f), std::invalid_argument);
}

TEST(CudaPWCellStress, SmearZeroDegaussThrows) {
    auto frame = make_h2_cubic(6.0);
    auto method = h_gth_method<cuda::pw::PW>();
    method.smearing = ::ATV::electronic::Smearing::fermi_dirac;
    method.degauss = 0.0;
    method.build(frame);
    EXPECT_THROW(method.stress(frame), std::invalid_argument);
}
