/// @file tests/cpu/scf/test_cuda_lgt1_pack.cpp
/// @brief Unit: Cartesian shell pack + CUDA RHF route for l > 1.
///
/// No SCF and no integrals. The device kernels (`launch_eri_lgt1`) consume
/// this pack; this test pins the host side of that contract against
/// `ShellList` and the basis tables.

#include <gtest/gtest.h>

#include <stdexcept>
#include <vector>

#include "atomiverse/core/frame_schema.h"
#include "atomiverse/core/gto.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/scf/cart.h"
#include "atomiverse/cpu/scf/internal/basis_workspace.h"
#include "atomiverse/cpu/scf/internal/cartesian_shell_pack.h"

using ATV::cpu::Frame;
using ATV::cpu::scf::AtomList;
using ATV::cpu::scf::BasisSet;
using ATV::cpu::scf::n_cart_of;
using ATV::cpu::scf::ShellList;
using ATV::cpu::scf::internal::CartesianShellPack;
using ATV::cpu::scf::internal::classify_cuda_rhf;
using ATV::cpu::scf::internal::contraction_norm;
using ATV::cpu::scf::internal::CudaRhfRouteKind;
using ATV::cpu::scf::internal::pack_cartesian_shells;

namespace {

Frame water() {
    Frame frame(3);
    frame.span(ATV::kAtom_atomic_number)[0] = 8;
    frame.span(ATV::kAtom_x)[0] = 0.0;
    frame.span(ATV::kAtom_y)[0] = 0.0;
    frame.span(ATV::kAtom_z)[0] = 0.1173;
    frame.span(ATV::kAtom_atomic_number)[1] = 1;
    frame.span(ATV::kAtom_x)[1] = 0.0;
    frame.span(ATV::kAtom_y)[1] = 0.7572;
    frame.span(ATV::kAtom_z)[1] = -0.4692;
    frame.span(ATV::kAtom_atomic_number)[2] = 1;
    frame.span(ATV::kAtom_x)[2] = 0.0;
    frame.span(ATV::kAtom_y)[2] = -0.7572;
    frame.span(ATV::kAtom_z)[2] = -0.4692;
    frame.box = ATV::cpu::Box::cubic(20.0);
    return frame;
}

} // namespace

TEST(CudaLgt1Pack, Sto3gHydrogenIsSpRoute) {
    const int Z[] = {1, 1};
    const auto route = classify_cuda_rhf(BasisSet::STO3G, Z, 2);
    EXPECT_EQ(route.kind, CudaRhfRouteKind::Sto3gSp);
    EXPECT_EQ(route.missing_kernel, nullptr);
}

TEST(CudaLgt1Pack, Pople631GsWaterIsCartesianD) {
    const int Z[] = {8, 1, 1};
    const auto route = classify_cuda_rhf(BasisSet::Pople631Gs, Z, 3);
    ASSERT_EQ(route.kind, CudaRhfRouteKind::CartesianLgt1);
    const CartesianShellPack pack =
        pack_cartesian_shells(BasisSet::Pople631Gs, Z, 3);
    EXPECT_EQ(pack.max_l, 2);
    EXPECT_EQ(pack.n_ao, 19);
    EXPECT_EQ(static_cast<int>(pack.ao_shell.size()), pack.n_ao);

    int d_shell = -1;
    for (int s = 0; s < pack.n_shells; ++s) {
        if (pack.L[static_cast<std::size_t>(s)] == 2) {
            d_shell = s;
            break;
        }
    }
    ASSERT_GE(d_shell, 0);
    EXPECT_EQ(n_cart_of(2), 6);
    const int ao0 = pack.ao_start[static_cast<std::size_t>(d_shell)];
    for (int c = 0; c < 6; ++c) {
        EXPECT_EQ(pack.ao_shell[static_cast<std::size_t>(ao0 + c)], d_shell);
        EXPECT_EQ(pack.ao_cart[static_cast<std::size_t>(ao0 + c)], c);
    }
    // Device cart_ang_device(L=2) must follow fill_cart_comps.
    int ang[6][3];
    ATV::cpu::scf::fill_cart_comps(2, ang);
    EXPECT_EQ(ang[0][0], 2);
    EXPECT_EQ(ang[0][1], 0);
    EXPECT_EQ(ang[0][2], 0);
    EXPECT_EQ(ang[5][0], 0);
    EXPECT_EQ(ang[5][1], 0);
    EXPECT_EQ(ang[5][2], 2);

    const int off = pack.prim_off[static_cast<std::size_t>(d_shell)];
    const double alpha = pack.alpha[static_cast<std::size_t>(off)];
    EXPECT_DOUBLE_EQ(contraction_norm(2, alpha), ATV::phys_norm_d(alpha));

    const Frame frame = water();
    const ShellList shells(AtomList(frame, 0), BasisSet::Pople631Gs);
    EXPECT_EQ(pack.n_ao, shells.n_basis);
}

TEST(CudaLgt1Pack, Pople631GWaterIsRaggedSp) {
    const int Z[] = {8, 1, 1};
    const auto route = classify_cuda_rhf(BasisSet::Pople631G, Z, 3);
    EXPECT_EQ(route.kind, CudaRhfRouteKind::CartesianLgt1);
    const CartesianShellPack pack =
        pack_cartesian_shells(BasisSet::Pople631G, Z, 3);
    EXPECT_EQ(pack.max_l, 1);
    EXPECT_GT(pack.n_shells, 1);
    // 6-31G contractions are not a uniform primitive count.
    EXPECT_NE(pack.nprim.front(), pack.nprim.back());
}

TEST(CudaLgt1Pack, SphericalDIsCartesianRoute) {
    const int Z[] = {8, 1, 1};
    const auto route = classify_cuda_rhf(BasisSet::Def2SVP, Z, 3);
    EXPECT_EQ(route.kind, CudaRhfRouteKind::CartesianLgt1);
    EXPECT_EQ(route.missing_kernel, nullptr);
    const CartesianShellPack pack =
        pack_cartesian_shells(BasisSet::Def2SVP, Z, 3);
    EXPECT_GE(pack.max_l, 2);
    EXPECT_LE(pack.max_l, 3);
}

TEST(CudaLgt1Pack, Def2TzvpWaterHasCartesianF) {
    const int Z[] = {8, 1, 1};
    const auto route = classify_cuda_rhf(BasisSet::Def2TZVP, Z, 3);
    ASSERT_EQ(route.kind, CudaRhfRouteKind::CartesianLgt1);
    const CartesianShellPack pack =
        pack_cartesian_shells(BasisSet::Def2TZVP, Z, 3);
    EXPECT_EQ(pack.max_l, 3);
    int f_shell = -1;
    for (int s = 0; s < pack.n_shells; ++s) {
        if (pack.L[static_cast<std::size_t>(s)] == 3) {
            f_shell = s;
            break;
        }
    }
    ASSERT_GE(f_shell, 0);
    EXPECT_EQ(n_cart_of(3), 10);
    const int ao0 = pack.ao_start[static_cast<std::size_t>(f_shell)];
    for (int c = 0; c < 10; ++c) {
        EXPECT_EQ(pack.ao_shell[static_cast<std::size_t>(ao0 + c)], f_shell);
        EXPECT_EQ(pack.ao_cart[static_cast<std::size_t>(ao0 + c)], c);
    }
    int ang[10][3];
    ATV::cpu::scf::fill_cart_comps(3, ang);
    EXPECT_EQ(ang[0][0], 3);
    EXPECT_EQ(ang[4][0], 1);
    EXPECT_EQ(ang[4][1], 1);
    EXPECT_EQ(ang[4][2], 1);
    EXPECT_EQ(ang[9][2], 3);
    const int off = pack.prim_off[static_cast<std::size_t>(f_shell)];
    const double alpha = pack.alpha[static_cast<std::size_t>(off)];
    EXPECT_DOUBLE_EQ(contraction_norm(3, alpha), ATV::phys_norm_f(alpha));
}

TEST(CudaLgt1Pack, MissingElementThrows) {
    const int Z[] = {2};
    EXPECT_THROW(pack_cartesian_shells(BasisSet::Pople631Gs, Z, 1),
                 std::runtime_error);
}

TEST(CudaLgt1Pack, ContractionNormRejectsG) {
    EXPECT_THROW(contraction_norm(4, 1.0), std::invalid_argument);
}
