/// @file tests/cuda/dynamics/test_hilbert.cu
/// @brief Direct spatial-key bijection and Hilbert face-adjacency invariants.
#include "../../../src/cuda/neighbor/hilbert.cuh"
#include <array>
#include <cmath>
#include <gtest/gtest.h>
#include <vector>

namespace {
template <int Bits>
void check_curve() {
    constexpr int side = 1 << Bits, count = side * side * side;
    std::vector<std::array<int, 3>> point(count);
    std::vector<bool> seen(count);
    for (int x = 0; x < side; ++x)
        for (int y = 0; y < side; ++y)
            for (int z = 0; z < side; ++z) {
                const auto key =
                    ATV::cuda::md::nlist::hilbert_3d<Bits>(x, y, z);
                ASSERT_LT(key, unsigned(count));
                ASSERT_FALSE(seen[key]);
                seen[key] = true;
                point[key] = {x, y, z};
            }
    for (int i = 1; i < count; ++i) {
        int distance = 0;
        for (int d = 0; d < 3; ++d)
            distance += std::abs(point[i][d] - point[i - 1][d]);
        ASSERT_EQ(distance, 1) << "bits=" << Bits << " key=" << i;
    }
}
} // namespace

TEST(CudaHilbert, ExhaustiveSmallCurvesAreBijectiveAndFaceAdjacent) {
    check_curve<1>();
    check_curve<2>();
    check_curve<3>();
    check_curve<4>();
}

TEST(CudaHilbert, FractionalCoordinatesQuantizeAndClampToTenBits) {
    using ATV::cuda::md::nlist::hilbert_3d;
    using ATV::cuda::md::nlist::hilbert_from_frac;
    EXPECT_EQ(hilbert_from_frac(-1., 0., 1.), hilbert_3d<10>(0, 0, 1023));
    EXPECT_EQ(hilbert_from_frac(0.5, 0.25, 0.125),
              hilbert_3d<10>(512, 256, 128));
}
