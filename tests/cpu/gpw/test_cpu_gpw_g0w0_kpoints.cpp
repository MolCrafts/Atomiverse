/// @file tests/cpu/gpw/test_cpu_gpw_g0w0_kpoints.cpp
/// @brief Multi-k spectral G0W0 kernels are invariant under the arbitrary
///        phase of each Bloch MO.
///
/// A diagonalizer returns every complex eigenvector up to a phase e^{iθ}, and
/// that phase differs between LAPACK builds and between runs whose input
/// differs in the last bit. Every quasiparticle observable — Σ_x, Σ_c, the QP
/// energy — must therefore be unchanged when single orbitals are multiplied by
/// arbitrary phases. Σ_x = −Σ_i (pi|ip) = −Σ_i ∫ ρ* v[ρ] with ρ = ψ_p^* ψ_i is
/// real and gauge invariant; the bilinear ∫ ρ v[ρ] is neither.

#include <array>
#include <cmath>
#include <complex>
#include <vector>

#include <gtest/gtest.h>

#include "atomiverse/cpu/gpw/internal/collocate.h"
#include "atomiverse/cpu/gpw/internal/mp2_kernels.h"

namespace {

using ATV::cpu::gpw::make_uniform_grid;
using ATV::cpu::gpw::UniformGrid;
using ATV::cpu::gpw::internal::Cplx;

constexpr int kNk = 2;
constexpr int kNmo = 3;
constexpr int kNocc = 1;
constexpr int kNvirt = 1;

// Smooth, complex, non-separable orbital: Gaussian envelope × plane-wave
// modulation, distinct centre and wavevector per (k, p).
std::vector<Cplx> make_orbitals(const UniformGrid& g, int ik) {
    const long npts = g.npts();
    std::vector<Cplx> psi(static_cast<std::size_t>(kNmo * npts));
    for (int p = 0; p < kNmo; ++p) {
        const double c[3] = {g.L[0] * (0.35 + 0.1 * p + 0.05 * ik),
                             g.L[1] * (0.45 + 0.05 * p),
                             g.L[2] * (0.55 - 0.08 * p)};
        const double s2 = 1.2 + 0.3 * p;
        const double kv[3] = {0.4 + 0.2 * p - 0.3 * ik, -0.25 * p, 0.15 + ik};
        for (long i = 0; i < npts; ++i) {
            double r[3];
            g.point(i, r);
            double d2 = 0.0, phase = 0.0;
            for (int a = 0; a < 3; ++a) {
                d2 += (r[a] - c[a]) * (r[a] - c[a]);
                phase += kv[a] * r[a];
            }
            psi[static_cast<std::size_t>(p * npts + i)] =
                std::exp(-d2 / (2.0 * s2)) * std::polar(1.0, phase);
        }
    }
    return psi;
}

void rephase(std::vector<Cplx>& psi, long npts, int p, double theta) {
    const Cplx z = std::polar(1.0, theta);
    for (long i = 0; i < npts; ++i)
        psi[static_cast<std::size_t>(p * npts + i)] *= z;
}

struct Fixture {
    UniformGrid grid;
    std::vector<std::vector<Cplx>> psi;
    std::vector<std::vector<double>> eps;
    std::vector<std::array<double, 3>> k_frac;
    std::vector<double> k_w;
    std::vector<double> vxc;

    Fixture() {
        const double cell[3] = {7.0, 7.0, 7.0};
        grid = make_uniform_grid(cell, 12.0);
        for (int ik = 0; ik < kNk; ++ik) {
            psi.push_back(make_orbitals(grid, ik));
            eps.push_back({-0.45 - 0.02 * ik, 0.15 + 0.01 * ik, 0.42});
        }
        k_frac = {{{-0.25, 0.0, 0.0}}, {{0.25, 0.0, 0.0}}};
        k_w = {0.5, 0.5};
        vxc.assign(static_cast<std::size_t>(grid.npts()), -0.3);
    }

    // Phases chosen to touch an occupied and a virtual orbital on both k.
    void rephase_all() {
        const long npts = grid.npts();
        rephase(psi[0], npts, 0, 0.7);
        rephase(psi[0], npts, 2, 2.1);
        rephase(psi[1], npts, 0, -1.3);
        rephase(psi[1], npts, 1, 2.9);
    }
};

void expect_same(const std::vector<std::vector<double>>& a,
                 const std::vector<std::vector<double>>& b, const char* what) {
    ASSERT_EQ(a.size(), b.size());
    for (std::size_t k = 0; k < a.size(); ++k) {
        ASSERT_EQ(a[k].size(), b[k].size());
        for (std::size_t p = 0; p < a[k].size(); ++p)
            EXPECT_NEAR(a[k][p], b[k][p], 1e-10)
                << what << " k=" << k << " p=" << p;
    }
}

} // namespace

TEST(GpwG0W0Kpoints, RestrictedIsInvariantUnderOrbitalPhases) {
    Fixture f;
    std::vector<std::vector<double>> qp0, sx0, vx0, qp1, sx1, vx1;
    bool ok0 = false, ok1 = false;
    ATV::cpu::gpw::internal::spectral_g0w0_rhf_kpoints(
        f.grid, f.psi, f.eps, f.k_frac, f.k_w, kNocc, kNvirt, f.vxc, qp0, sx0,
        vx0, ok0);
    f.rephase_all();
    ATV::cpu::gpw::internal::spectral_g0w0_rhf_kpoints(
        f.grid, f.psi, f.eps, f.k_frac, f.k_w, kNocc, kNvirt, f.vxc, qp1, sx1,
        vx1, ok1);
    ASSERT_TRUE(ok0);
    ASSERT_TRUE(ok1);
    expect_same(sx0, sx1, "sigma_x");
    expect_same(qp0, qp1, "qp");
    // Exchange with a positive Coulomb form never raises an orbital.
    for (const auto& sk : sx0)
        for (const double s : sk)
            EXPECT_LE(s, 0.0);
}

TEST(GpwG0W0Kpoints, UnrestrictedIsInvariantUnderOrbitalPhases) {
    Fixture f;
    std::vector<std::vector<Cplx>> psi_b = f.psi;
    std::vector<std::vector<double>> qp0, sx0, vx0, qp1, sx1, vx1;
    bool ok0 = false, ok1 = false;
    ATV::cpu::gpw::internal::spectral_g0w0_uhf_kpoints(
        f.grid, f.psi, psi_b, f.eps, f.eps, f.k_frac, f.k_w, kNocc, kNvirt,
        kNocc, kNvirt, f.vxc, qp0, sx0, vx0, ok0);
    f.rephase_all();
    const long npts = f.grid.npts();
    rephase(psi_b[0], npts, 1, -0.4);
    rephase(psi_b[1], npts, 0, 1.9);
    ATV::cpu::gpw::internal::spectral_g0w0_uhf_kpoints(
        f.grid, f.psi, psi_b, f.eps, f.eps, f.k_frac, f.k_w, kNocc, kNvirt,
        kNocc, kNvirt, f.vxc, qp1, sx1, vx1, ok1);
    ASSERT_TRUE(ok0);
    ASSERT_TRUE(ok1);
    expect_same(sx0, sx1, "sigma_x");
    expect_same(qp0, qp1, "qp");
}

TEST(GpwG0W0Kpoints, UnrestrictedEqualsRestrictedForEqualSpins) {
    Fixture f;
    std::vector<std::vector<double>> qpr, sxr, vxr, qpu, sxu, vxu;
    bool okr = false, oku = false;
    ATV::cpu::gpw::internal::spectral_g0w0_rhf_kpoints(
        f.grid, f.psi, f.eps, f.k_frac, f.k_w, kNocc, kNvirt, f.vxc, qpr, sxr,
        vxr, okr);
    std::vector<std::vector<Cplx>> psi_b = f.psi;
    rephase(psi_b[1], f.grid.npts(), 2, 1.1); // β gauge differs from α
    ATV::cpu::gpw::internal::spectral_g0w0_uhf_kpoints(
        f.grid, f.psi, psi_b, f.eps, f.eps, f.k_frac, f.k_w, kNocc, kNvirt,
        kNocc, kNvirt, f.vxc, qpu, sxu, vxu, oku);
    ASSERT_TRUE(okr);
    ASSERT_TRUE(oku);
    expect_same(sxr, sxu, "sigma_x");
    expect_same(qpr, qpu, "qp");
}
