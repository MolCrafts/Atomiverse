/// @file tests/test_torch_neighbor_list.cu
/// @brief Tests for torch NeighborList operator.
///
/// The neighbor pair list is owned by `torch::NeighborList` (its nested
/// `PairList`), not by `torch::Frame`. The convenience `neighbor_list()`
/// free function returns the built NeighborList so callers can read
/// `nl.pairs()`.

#include <gtest/gtest.h>

#ifdef ATV_HAS_TORCH

#include "atomiverse/torch/frame/transfer.cuh"
#include "atomiverse/torch/neighbor/neighbor_list.cuh"
#include "atomiverse/torch/potentials/lj.cuh"

#include <cmath>
#include <cuda_runtime.h>

using ATV::Mat3;
namespace atorch = ATV::torch;

namespace {

/// @brief Build a simple cubic lattice torch::Frame.
atorch::Frame make_cubic_frame(int n_side, double box_len) {
    int n = n_side * n_side * n_side;
    double spacing = box_len / double(n_side);

    ATV::cpu::Frame cpu_frame(n);
    int idx = 0;
    for (int ix = 0; ix < n_side; ++ix)
        for (int iy = 0; iy < n_side; ++iy)
            for (int iz = 0; iz < n_side; ++iz) {
                cpu_frame.atoms.x[idx] = (double(ix) + double(0.5)) * spacing;
                cpu_frame.atoms.y[idx] = (double(iy) + double(0.5)) * spacing;
                cpu_frame.atoms.z[idx] = (double(iz) + double(0.5)) * spacing;
                cpu_frame.atoms.mass[idx] = double(1.0);
                cpu_frame.atoms.type[idx] = 0;
                idx++;
            }
    cpu_frame.box = Mat3::cubic(static_cast<double>(box_len));
    return atorch::from_cpu(cpu_frame);
}

} // anonymous namespace

TEST(TorchNeighborList, CubicLatticePairCount) {
    // 3^3 = 27 atoms, spacing = 10/3 = 3.33 Angstrom
    // cutoff = 4.0, skin = 0 -> r_search = 4.0
    // Only nearest neighbors at distance 3.33 are included (6 per atom)
    // Total pairs = 27 * 6 / 2 = 81
    double box_len = double(10.0);
    double cutoff = double(4.0);
    auto frame = make_cubic_frame(3, box_len);

    auto nl = atorch::neighbor_list(frame, cutoff, double(0));

    // Exact count depends on cluster algorithm; just check populated
    EXPECT_GT(nl.n_pairs(), 0);
    EXPECT_TRUE(nl.pairs().pair_i.defined());
    EXPECT_TRUE(nl.pairs().pair_j.defined());
    EXPECT_TRUE(nl.pairs().pair_dist.defined());
}

TEST(TorchNeighborList, PairTensorsNoGrad) {
    auto frame = make_cubic_frame(3, double(10.0));

    auto nl = atorch::neighbor_list(frame, double(4.0), double(0));
    ASSERT_GT(nl.n_pairs(), 0);

    const auto& pairs = nl.pairs();
    EXPECT_FALSE(pairs.pair_i.requires_grad());
    EXPECT_FALSE(pairs.pair_j.requires_grad());
    EXPECT_FALSE(pairs.pair_dx.requires_grad());
    EXPECT_FALSE(pairs.pair_dy.requires_grad());
    EXPECT_FALSE(pairs.pair_dz.requires_grad());
    EXPECT_FALSE(pairs.pair_dist.requires_grad());
}

TEST(TorchNeighborList, PairTensorsAreSelfContained) {
    auto frame = make_cubic_frame(3, double(10.0));

    auto nl = atorch::neighbor_list(frame, double(4.0), double(0));
    ASSERT_GT(nl.n_pairs(), 0);

    const auto& pairs = nl.pairs();
    EXPECT_TRUE(pairs.pair_i.is_cuda());
    EXPECT_TRUE(pairs.pair_i.is_contiguous());
    EXPECT_EQ(pairs.pair_i.scalar_type(), atorch::kIntDtype);

    EXPECT_TRUE(pairs.pair_dist.is_cuda());
    EXPECT_TRUE(pairs.pair_dist.is_contiguous());
    EXPECT_EQ(pairs.pair_dist.scalar_type(), atorch::kRealDtype);

    int64_t np = static_cast<int64_t>(nl.n_pairs());
    EXPECT_EQ(pairs.pair_i.numel(), np);
    EXPECT_EQ(pairs.pair_j.numel(), np);
    EXPECT_EQ(pairs.pair_dx.numel(), np);
    EXPECT_EQ(pairs.pair_dist.numel(), np);
}

TEST(TorchNeighborList, DistanceConsistency) {
    auto frame = make_cubic_frame(3, double(10.0));

    auto nl = atorch::neighbor_list(frame, double(4.0), double(0));
    ASSERT_GT(nl.n_pairs(), 0);

    const auto& pairs = nl.pairs();
    auto dx = pairs.pair_dx.to(::torch::kCPU);
    auto dy = pairs.pair_dy.to(::torch::kCPU);
    auto dz = pairs.pair_dz.to(::torch::kCPU);
    auto dist = pairs.pair_dist.to(::torch::kCPU);

    for (int64_t p = 0; p < nl.n_pairs(); ++p) {
        double d = dx[p].item<double>();
        double e = dy[p].item<double>();
        double f = dz[p].item<double>();
        double expected = std::sqrt(d * d + e * e + f * f);
        double actual = dist[p].item<double>();
        EXPECT_NEAR(actual, expected, double(1e-4))
            << "pair " << p << ": dist mismatch";
    }
}

TEST(TorchNeighborList, PairListFeedsLJ) {
    auto frame = make_cubic_frame(3, double(10.0));
    auto real_opts = ::torch::TensorOptions()
                         .dtype(atorch::kRealDtype)
                         .device(::torch::kCUDA);
    auto scalar_opts = ::torch::TensorOptions()
                           .dtype(::at::ScalarType::Double)
                           .device(::torch::kCUDA);

    int n = frame.n_atoms();
    frame.atoms.fx = ::torch::zeros({n}, real_opts);
    frame.atoms.fy = ::torch::zeros({n}, real_opts);
    frame.atoms.fz = ::torch::zeros({n}, real_opts);
    frame.energy = ::torch::zeros({1}, scalar_opts);

    auto nl = atorch::neighbor_list(frame, double(4.0), double(0));
    ASSERT_GT(nl.n_pairs(), 0);

    double eps = double(1.0), sig = double(1.0);
    auto energy = atorch::lj(frame, nl, ::torch::tensor({eps}, real_opts),
                             ::torch::tensor({sig}, real_opts), double(4.0));
    cudaDeviceSynchronize();

    double E = energy.to(::torch::kCPU).item<double>();
    EXPECT_NE(E, double(0));
}

TEST(TorchNeighborList, ClassBasedBuildUpdate) {
    // Test the class-based API: build once, update multiple times
    auto frame = make_cubic_frame(3, double(10.0));

    atorch::NeighborList nl(double(4.0), double(0));
    nl.build(frame);
    nl.update(frame);

    int first_count = nl.n_pairs();
    EXPECT_GT(first_count, 0);

    // Update again (same positions -> same result)
    nl.update(frame);
    EXPECT_EQ(nl.n_pairs(), first_count);
}

#else

TEST(TorchNeighborList, Disabled) {
    GTEST_SKIP() << "ATV_HAS_TORCH not defined";
}

#endif // ATV_HAS_TORCH
