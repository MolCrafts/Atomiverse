/// @file tests/cuda/gpw/test_kpoint_stvnl.cu
/// @brief Unit: device multi-k phased ST vs host `build_overlap_kinetic_k`.
///
/// Single-function path only — not a full multi-k SCF. Exercises
/// `KpointStvnlDevice::compute` / download against the host twin on H₂-class
/// DZV-GTH (pure-s; Vnl channels empty).
#include "atomiverse/cpu/gpw/gaussian_basis.h"
#include "atomiverse/cpu/gpw/gpw_solver.h"
#include "atomiverse/cpu/gpw/gth_potential.h"
#include "atomiverse/cpu/gpw/internal/kpoints.h"
#include "atomiverse/cpu/gpw/internal/one_electron.h"
#include "atomiverse/cpu/gpw/internal/periodic.h"
#include "atomiverse/cuda/gpw/internal/kpoint_stvnl.cuh"

#include <cmath>
#include <complex>
#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <vector>
#include <xtensor/containers/xtensor.hpp>

using ATV::cpu::gpw::basis_function_radius;
using ATV::cpu::gpw::GaussianBasis;
using ATV::cpu::gpw::GridBasisFunction;
using ATV::cpu::gpw::GthPotential;
using ATV::cpu::gpw::pair_image_counts;
using ATV::cpu::gpw::internal::build_overlap_kinetic_k;
using ATV::cpu::gpw::internal::CplxMatrix;
using ATV::cpu::gpw::internal::KPoint;
using ATV::cuda::gpw::internal::KpointStvnlDevice;

namespace {

using Cplx = std::complex<double>;

void check_cuda(cudaError_t e, const char* what) {
    ASSERT_EQ(e, cudaSuccess) << what << ": " << cudaGetErrorString(e);
}

std::vector<ATV::cpu::gpw::internal::GpwAtom> h2_atoms() {
    const auto basis = GaussianBasis(molrs::Element::H, "DZV-GTH-PADE");
    const auto pot = GthPotential(molrs::Element::H, "GTH-PADE-q1");
    const double rA[3] = {5.5, 6.0, 6.0};
    const double rB[3] = {6.5, 6.0, 6.0};
    return {ATV::cpu::gpw::internal::GpwAtom(rA, basis, pot, 1),
            ATV::cpu::gpw::internal::GpwAtom(rB, basis, pot, 1)};
}

std::vector<GridBasisFunction> flatten_basis_local(
    const std::vector<ATV::cpu::gpw::internal::GpwAtom>& atoms) {
    std::vector<GridBasisFunction> basis;
    for (const auto& a : atoms)
        for (const auto& bf : a.basis)
            basis.push_back(bf);
    return basis;
}

} // namespace

TEST(CudaGpwKpointStvnl, MatchesHostST) {
    auto atoms = h2_atoms();
    const auto basis = flatten_basis_local(atoms);
    const int nb = static_cast<int>(basis.size());
    ASSERT_GE(nb, 2);

    constexpr double kEps = 1e-8;
    constexpr double kScreen = 1e-14;
    const double cell[3] = {12.0, 12.0, 12.0};
    std::vector<double> radii(static_cast<std::size_t>(nb));
    double rmax = 0.0;
    for (int i = 0; i < nb; ++i) {
        radii[static_cast<std::size_t>(i)] =
            basis_function_radius(basis[static_cast<std::size_t>(i)], kEps);
        rmax = std::max(rmax, radii[static_cast<std::size_t>(i)]);
    }
    int mimg[3] = {0, 0, 0};
    pair_image_counts(cell, /*periodic=*/true, 2.0 * rmax, mimg);

    std::vector<KPoint> kpts(2);
    kpts[0].frac[0] = 0.25;
    kpts[0].frac[1] = 0.0;
    kpts[0].frac[2] = 0.0;
    kpts[0].weight = 0.5;
    kpts[1].frac[0] = -0.25;
    kpts[1].frac[1] = 0.0;
    kpts[1].frac[2] = 0.0;
    kpts[1].weight = 0.5;
    const int nk = 2;

    std::vector<CplxMatrix> S_host, T_host;
    build_overlap_kinetic_k(basis, radii, cell, mimg, kScreen, kpts, S_host,
                            T_host);

    std::vector<double> k_frac(static_cast<std::size_t>(nk) * 3);
    for (int ik = 0; ik < nk; ++ik) {
        k_frac[static_cast<std::size_t>(ik) * 3 + 0] =
            kpts[static_cast<std::size_t>(ik)].frac[0];
        k_frac[static_cast<std::size_t>(ik) * 3 + 1] =
            kpts[static_cast<std::size_t>(ik)].frac[1];
        k_frac[static_cast<std::size_t>(ik) * 3 + 2] =
            kpts[static_cast<std::size_t>(ik)].frac[2];
    }

    KpointStvnlDevice stvnl;
    stvnl.compute(/*stream=*/nullptr, basis, radii, atoms, cell, mimg, kScreen,
                  k_frac.data(), nk);
    std::vector<Cplx> S_dev(static_cast<std::size_t>(nk) * nb * nb);
    std::vector<Cplx> T_dev(S_dev.size());
    std::vector<Cplx> Vnl_dev(S_dev.size());
    stvnl.download_S(S_dev.data());
    stvnl.download_T(T_dev.data());
    stvnl.download_Vnl(Vnl_dev.data());

    // H GTH-PADE-q1 has no nonlocal channels → Vnl must be identically zero.
    for (const auto& v : Vnl_dev) {
        EXPECT_NEAR(v.real(), 0.0, 1e-15);
        EXPECT_NEAR(v.imag(), 0.0, 1e-15);
    }

    constexpr double kAbsTol = 1.0e-10;
    constexpr double kRelTol = 1.0e-9;
    double max_abs_S = 0.0, max_rel_S = 0.0;
    double max_abs_T = 0.0, max_rel_T = 0.0;
    for (int ik = 0; ik < nk; ++ik) {
        for (int u = 0; u < nb; ++u) {
            for (int t = 0; t < nb; ++t) {
                const Cplx hs = S_host[static_cast<std::size_t>(ik)](u, t);
                const Cplx ht = T_host[static_cast<std::size_t>(ik)](u, t);
                const std::size_t idx = static_cast<std::size_t>(ik) * nb * nb
                                        + static_cast<std::size_t>(u) * nb
                                        + static_cast<std::size_t>(t);
                const Cplx ds = S_dev[idx];
                const Cplx dt = T_dev[idx];
                const double as = std::abs(ds - hs);
                const double at = std::abs(dt - ht);
                max_abs_S = std::max(max_abs_S, as);
                max_abs_T = std::max(max_abs_T, at);
                max_rel_S =
                    std::max(max_rel_S, as / std::max(std::abs(hs), 1e-14));
                max_rel_T =
                    std::max(max_rel_T, at / std::max(std::abs(ht), 1e-14));
            }
        }
        // Hermiticity of device S
        for (int u = 0; u < nb; ++u) {
            for (int t = 0; t <= u; ++t) {
                const std::size_t iut = static_cast<std::size_t>(ik) * nb * nb
                                        + static_cast<std::size_t>(u) * nb
                                        + static_cast<std::size_t>(t);
                const std::size_t itu = static_cast<std::size_t>(ik) * nb * nb
                                        + static_cast<std::size_t>(t) * nb
                                        + static_cast<std::size_t>(u);
                const Cplx a = S_dev[iut];
                const Cplx b = std::conj(S_dev[itu]);
                EXPECT_NEAR(a.real(), b.real(), 1e-12);
                EXPECT_NEAR(a.imag(), b.imag(), 1e-12);
            }
        }
    }
    EXPECT_LE(max_abs_S, kAbsTol) << "max|ΔS|=" << max_abs_S;
    EXPECT_LE(max_abs_T, kAbsTol) << "max|ΔT|=" << max_abs_T;
    EXPECT_LE(max_rel_S, kRelTol) << "max rel S=" << max_rel_S;
    EXPECT_LE(max_rel_T, kRelTol) << "max rel T=" << max_rel_T;
    std::printf("ST twin: max|ΔS|=%.3e max|ΔT|=%.3e (abs)\n", max_abs_S,
                max_abs_T);
}

TEST(CudaGpwKpointStvnl, RejectEmpty) {
    KpointStvnlDevice stvnl;
    std::vector<GridBasisFunction> empty;
    std::vector<double> radii;
    std::vector<ATV::cpu::gpw::internal::GpwAtom> atoms;
    double cell[3] = {1, 1, 1};
    int mimg[3] = {0, 0, 0};
    double kfrac[3] = {0, 0, 0};
    EXPECT_THROW(stvnl.compute(nullptr, empty, radii, atoms, cell, mimg, 1e-14,
                               kfrac, 1),
                 std::invalid_argument);
    EXPECT_THROW(stvnl.download_S(nullptr), std::runtime_error);
}
