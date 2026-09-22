/// @file tests/cpu/gpw/test_cpu_gpw_gamma_k_identity.cpp
/// @brief Unit: the Γ-real and k={Γ} paths build IDENTICAL operators.
///
/// At k = Γ the Bloch problem reduces exactly to the real Γ problem, so at
/// the SAME input density both paths must produce the same S, the same
/// H_KS = ∂E/∂P, and the same energy terms — to numerical precision, not to
/// a physics tolerance. A fixed-density evaluation (max_iter == 0) probes
/// the operators in one KS build with no SCF anywhere.
///
/// Regression guard for the pruned-CSR image-coalescing defect: duplicate
/// per-image (point, function) entries broke the KS-integrate triangle's
/// diagonal (cross-image χ_u^{T1}·χ_u^{T2} counted once, not twice), so the
/// Γ H_KS drifted ~5e-7 off the k={Γ} one and the two SCF fixed points —
/// and their analytic forces — diverged (~1e-5 Ha/Bohr on this system).

#include <cmath>
#include <complex>
#include <vector>

#include <gtest/gtest.h>

#include "atomiverse/cpu/gpw/gaussian_basis.h"
#include "atomiverse/cpu/gpw/gpw_solver.h"
#include "atomiverse/cpu/gpw/gth_potential.h"

using namespace ATV::cpu::gpw;
using ATV::cpu::linalg::CplxMatrix;
using ATV::cpu::linalg::RealMatrix;

namespace {

std::vector<GpwAtom> h2_atoms() {
    const auto basis = GaussianBasis(molrs::Element::H, "DZV-GTH-PADE");
    const auto pot = GthPotential(molrs::Element::H, "GTH-PADE-q1");
    const double rA[3] = {5.5, 6.0, 6.0};
    const double rB[3] = {6.5, 6.0, 6.0};
    return {GpwAtom(rA, basis, pot, 1), GpwAtom(rB, basis, pot, 1)};
}

// The 12-Bohr cell with the DZV basis is the sharp configuration: the
// Gaussian tail radius at eps 1e-8 exceeds L/2, so basis functions reach
// the grid through more than one lattice image.
GpwDft fixed_density_cfg() {
    GpwDft m;
    m.cell[0] = m.cell[1] = m.cell[2] = 12.0;
    m.cutoff_ry = 100.0;
    m.n_alpha = 2;
    m.n_beta = 0;
    m.max_iter = 0; // fixed-density: one collocate→Poisson→XC→integrate
    m.nonperiodic = false;
    m.local_pp_on_grid = true;
    m.xc = XcKind::PADE;
    m.eps_default = 1e-8;
    return m;
}

// Synthetic symmetric trial density — any symmetric P probes the operator
// identity; no SCF is needed (or wanted) here.
RealMatrix trial_P(std::size_t nb, double scale) {
    RealMatrix P = xt::zeros<double>({nb, nb});
    for (std::size_t i = 0; i < nb; ++i) {
        P(i, i) = scale;
        for (std::size_t j = i + 1; j < nb; ++j)
            P(i, j) = P(j, i) = scale * 0.1 / static_cast<double>(1 + i + j);
    }
    return P;
}

CplxMatrix to_cplx(const RealMatrix& A) {
    CplxMatrix M = xt::zeros<std::complex<double>>({A.shape(0), A.shape(1)});
    for (std::size_t i = 0; i < A.shape(0); ++i)
        for (std::size_t j = 0; j < A.shape(1); ++j)
            M(i, j) = A(i, j);
    return M;
}

double max_abs_diff_rc(const RealMatrix& A, const CplxMatrix& B) {
    double m = 0.0;
    for (std::size_t i = 0; i < A.shape(0); ++i)
        for (std::size_t j = 0; j < A.shape(1); ++j)
            m = std::max(m, std::abs(A(i, j) - B(i, j).real()));
    return m;
}

double max_im(const CplxMatrix& B) {
    double m = 0.0;
    for (std::size_t i = 0; i < B.shape(0); ++i)
        for (std::size_t j = 0; j < B.shape(1); ++j)
            m = std::max(m, std::abs(B(i, j).imag()));
    return m;
}

} // namespace

TEST(GpwGammaKIdentity, FixedDensityOperatorsMatch) {
    auto atoms = h2_atoms();
    std::size_t nb = 0;
    for (const auto& at : atoms)
        nb += at.basis.size();
    const RealMatrix Pa = trial_P(nb, 0.5);
    const RealMatrix Pb = trial_P(nb, 0.2);

    GpwDft fg = fixed_density_cfg();
    fg.build(atoms);
    const GpwResult& eg = fg.run(atoms, nullptr, &Pa, &Pb);

    GpwDft fk = fixed_density_cfg();
    fk.k_points = {{0.0, 0.0, 0.0}};
    fk.k_point_weights = {1.0};
    fk.build(atoms);
    std::vector<CplxMatrix> Pk_a = {to_cplx(Pa)};
    std::vector<CplxMatrix> Pk_b = {to_cplx(Pb)};
    const GpwResult& ek =
        fk.run(atoms, nullptr, nullptr, nullptr, &Pk_a, &Pk_b);

    // Same functional: every energy term identical.
    EXPECT_NEAR(eg.e_total, ek.e_total, 1e-12);
    EXPECT_NEAR(eg.e_kinetic, ek.e_kinetic, 1e-12);
    EXPECT_NEAR(eg.e_loc_sr, ek.e_loc_sr, 1e-12);
    EXPECT_NEAR(eg.e_es, ek.e_es, 1e-12);
    EXPECT_NEAR(eg.e_xc, ek.e_xc, 1e-12);

    // Same operators: S and both spin KS matrices, element-wise.
    ASSERT_GT(eg.S.shape(0), 0u);
    ASSERT_FALSE(ek.S_k.empty());
    EXPECT_LT(max_abs_diff_rc(eg.S, ek.S_k[0]), 1e-13);
    ASSERT_GT(eg.H_ks.shape(0), 0u);
    ASSERT_FALSE(ek.H_ks_k.empty());
    EXPECT_LT(max_abs_diff_rc(eg.H_ks, ek.H_ks_k[0]), 1e-12);
    EXPECT_LT(max_im(ek.H_ks_k[0]), 1e-14);
    ASSERT_GT(eg.H_ks_beta.shape(0), 0u);
    ASSERT_FALSE(ek.H_ks_k_beta.empty());
    EXPECT_LT(max_abs_diff_rc(eg.H_ks_beta, ek.H_ks_k_beta[0]), 1e-12);
    EXPECT_LT(max_im(ek.H_ks_k_beta[0]), 1e-14);
}
