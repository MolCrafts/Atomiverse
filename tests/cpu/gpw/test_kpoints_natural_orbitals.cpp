/// @file tests/cpu/gpw/test_kpoints_natural_orbitals.cpp
/// @brief Unit test for `gpw::internal::natural_orbitals_k` — the Bloch
///        natural orbitals that seed the k-point OT warm start.

#include <gtest/gtest.h>

#include <complex>

#include "atomiverse/cpu/gpw/internal/kpoints.h"
#include "atomiverse/cpu/linalg/orbital_tangent.h"

using ATV::cpu::gpw::internal::KOrth;
using ATV::cpu::gpw::internal::natural_orbitals_k;
using ATV::cpu::linalg::CplxMatrix;
using Cplx = std::complex<double>;

// [Contract] P = C C† of two S-orthonormal Bloch orbitals in a non-orthogonal
// Hermitian basis has occupations {0, 1, 1}; the two most occupied natural
// orbitals rebuild P.
TEST(GpwKpointNaturalOrbitals, RecoverSpinDensityInNonOrthogonalBasis) {
    const CplxMatrix S{{Cplx(1.0, 0.0), Cplx(0.3, 0.2), Cplx(0.1, 0.0)},
                       {Cplx(0.3, -0.2), Cplx(1.0, 0.0), Cplx(0.0, 0.25)},
                       {Cplx(0.1, 0.0), Cplx(0.0, -0.25), Cplx(1.0, 0.0)}};
    const CplxMatrix V{{Cplx(1.0, 0.0), Cplx(0.0, 0.0)},
                       {Cplx(0.5, 0.1), Cplx(1.0, 0.0)},
                       {Cplx(0.0, 0.0), Cplx(0.3, -0.2)}};
    const CplxMatrix C = ATV::cpu::linalg::lowdin_orthonormalize_k(V, S);
    CplxMatrix P = xt::zeros<Cplx>({std::size_t(3), std::size_t(3)});
    for (std::size_t i = 0; i < 3; ++i)
        for (std::size_t j = 0; j < 3; ++j)
            for (std::size_t o = 0; o < 2; ++o)
                P(i, j) += C(i, o) * std::conj(C(j, o));

    const KOrth orth(S, 1e-10);
    auto [occ, nat] = natural_orbitals_k(P, S, orth);
    ASSERT_EQ(occ.size(), std::size_t(3));
    EXPECT_NEAR(occ[0], 0.0, 1e-12);
    EXPECT_NEAR(occ[1], 1.0, 1e-12);
    EXPECT_NEAR(occ[2], 1.0, 1e-12);

    for (std::size_t i = 0; i < 3; ++i)
        for (std::size_t j = 0; j < 3; ++j) {
            Cplx pij(0.0, 0.0);
            for (std::size_t o = 1; o < 3; ++o)
                pij += nat(i, o) * std::conj(nat(j, o));
            EXPECT_NEAR(std::abs(pij - P(i, j)), 0.0, 1e-12)
                << "P mismatch at (" << i << "," << j << ")";
        }
}
