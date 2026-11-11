#include "atomiverse/cpu/gpw/internal/kpoints.h"
#include <complex>
#include <gtest/gtest.h>
#include <stdexcept>
#include <vector>

namespace {
using ATV::cpu::gpw::internal::accumulate_bloch_density_stack;
using ATV::cpu::gpw::internal::CplxMatrix;
using Cplx = std::complex<double>;

TEST(GpwKpointsDensityStack, ComplexPhaseAndTimeReversalWeights) {
    CplxMatrix P = {{Cplx(2), Cplx(0, 1)}, {Cplx(0, -1), Cplx(4)}};
    std::vector<std::vector<double>> D(2, std::vector<double>(4, 0));
    const std::vector<Cplx> phases{{1, 0}, {0, -1}};
    accumulate_bloch_density_stack(P, 0.25, phases, D);
    EXPECT_DOUBLE_EQ(D[0][0], 0.5);
    EXPECT_DOUBLE_EQ(D[0][3], 1.0);
    EXPECT_DOUBLE_EQ(D[1][1], 0.25); // Re(i * -i) / 4
    EXPECT_DOUBLE_EQ(D[1][2], -0.25);

    // P(-k)=conj(P(k)), phase(-k)=conj(phase(k)); unequal weights still
    // preserve the same physical real density, including off-diagonal signs.
    P(0, 1) = Cplx(0, -1);
    P(1, 0) = Cplx(0, 1);
    const std::vector<Cplx> conjugate{{1, 0}, {0, 1}};
    accumulate_bloch_density_stack(P, 0.75, conjugate, D);
    const std::vector<std::vector<double>> expected{{2, 0, 0, 4},
                                                    {0, 1, -1, 0}};
    EXPECT_EQ(D, expected);
}

TEST(GpwKpointsDensityStack, BadShapesRejectBeforeOutputMutation) {
    const CplxMatrix P = {{Cplx(2), Cplx(0)}, {Cplx(0), Cplx(4)}};
    const std::vector<Cplx> phases{{1, 0}, {0, -1}};
    std::vector<std::vector<double>> D{{7, 8, 9, 10}, {1, 2, 3}};
    const auto before = D;
    EXPECT_THROW(accumulate_bloch_density_stack(P, 1, phases, D),
                 std::invalid_argument);
    EXPECT_EQ(D, before);
}
} // namespace
