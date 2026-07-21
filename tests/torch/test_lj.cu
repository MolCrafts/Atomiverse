/// @file tests/test_torch_lj.cu
/// @brief Tests for differentiable LJ operator in the torch backend.

#include <gtest/gtest.h>

#ifdef ATV_HAS_TORCH

#include "atomiverse/torch/frame/transfer.cuh"
#include "atomiverse/torch/potentials/lj.cuh"

#include <cmath>
#include <cuda_runtime.h>

namespace atorch = ATV::torch;

namespace {

/// @brief Analytical LJ energy for one pair at distance r.
double analytical_lj_energy(double r, double eps, double sig, double cutoff) {
    double sr_cut = sig / cutoff;
    double sr_cut6 = sr_cut * sr_cut * sr_cut * sr_cut * sr_cut * sr_cut;
    double sr_cut12 = sr_cut6 * sr_cut6;
    double v_shift = double(4) * eps * (sr_cut12 - sr_cut6);

    double sr = sig / r;
    double sr6 = sr * sr * sr * sr * sr * sr;
    double sr12 = sr6 * sr6;
    return double(4) * eps * (sr12 - sr6) - v_shift;
}

/// @brief Build a 2-atom torch::Frame with one pair at distance r along x.
atorch::Frame make_lj_frame(double r) {
    auto real_opts = ::torch::TensorOptions()
                         .dtype(atorch::kRealDtype)
                         .device(::torch::kCUDA);
    auto int_opts = ::torch::TensorOptions()
                        .dtype(atorch::kIntDtype)
                        .device(::torch::kCUDA);

    atorch::Frame frame;
    frame.n_atoms_ = 2;
    frame.x = ::torch::tensor({double(0), r}, real_opts);
    frame.y = ::torch::zeros({2}, real_opts);
    frame.z = ::torch::zeros({2}, real_opts);
    frame.fx = ::torch::zeros({2}, real_opts);
    frame.fy = ::torch::zeros({2}, real_opts);
    frame.fz = ::torch::zeros({2}, real_opts);
    frame.energy = ::torch::zeros({1}, real_opts);
    frame.n_pairs_ = 1;
    frame.pair_i = ::torch::tensor({int(0)}, int_opts);
    frame.pair_j = ::torch::tensor({int(1)}, int_opts);
    frame.pair_dx = ::torch::tensor({r}, real_opts);
    frame.pair_dy = ::torch::zeros({1}, real_opts);
    frame.pair_dz = ::torch::zeros({1}, real_opts);
    frame.pair_dist = ::torch::tensor({r}, real_opts);
    return frame;
}

} // anonymous namespace

TEST(TorchLJ, TwoAtomAnalytical) {
    double eps = double(1.0), sig = double(1.0), cutoff = double(3.0),
           r = double(1.5);
    auto frame = make_lj_frame(r);
    auto real_opts = ::torch::TensorOptions()
                         .dtype(atorch::kRealDtype)
                         .device(::torch::kCUDA);

    auto energy = atorch::lj(frame, ::torch::tensor({eps}, real_opts),
                             ::torch::tensor({sig}, real_opts), cutoff);
    cudaDeviceSynchronize();

    double computed = energy.to(::torch::kCPU).item<double>();
    double expected = analytical_lj_energy(r, eps, sig, cutoff);
    EXPECT_NEAR(computed, expected, double(1e-4));
}

TEST(TorchLJ, GradcheckEpsilon) {
    double eps_val = double(0.5), sig_val = double(1.0), cutoff = double(3.0),
           r = double(1.5);
    auto real_opts = ::torch::TensorOptions()
                         .dtype(atorch::kRealDtype)
                         .device(::torch::kCUDA);

    double h = double(1e-4);
    double numerical = (analytical_lj_energy(r, eps_val + h, sig_val, cutoff)
                        - analytical_lj_energy(r, eps_val - h, sig_val, cutoff))
                       / (double(2) * h);

    auto frame = make_lj_frame(r);
    auto t_eps = ::torch::tensor({eps_val}, real_opts).requires_grad_(true);
    auto t_sig = ::torch::tensor({sig_val}, real_opts);

    auto energy = atorch::lj(frame, t_eps, t_sig, cutoff);
    energy.backward();
    cudaDeviceSynchronize();

    double autograd = t_eps.grad().to(::torch::kCPU).item<double>();
    EXPECT_NEAR(autograd, numerical, double(1e-2));
}

TEST(TorchLJ, GradcheckSigma) {
    double eps_val = double(0.5), sig_val = double(1.0), cutoff = double(3.0),
           r = double(1.5);
    auto real_opts = ::torch::TensorOptions()
                         .dtype(atorch::kRealDtype)
                         .device(::torch::kCUDA);

    double h = double(1e-4);
    double numerical = (analytical_lj_energy(r, eps_val, sig_val + h, cutoff)
                        - analytical_lj_energy(r, eps_val, sig_val - h, cutoff))
                       / (double(2) * h);

    auto frame = make_lj_frame(r);
    auto t_eps = ::torch::tensor({eps_val}, real_opts);
    auto t_sig = ::torch::tensor({sig_val}, real_opts).requires_grad_(true);

    auto energy = atorch::lj(frame, t_eps, t_sig, cutoff);
    energy.backward();
    cudaDeviceSynchronize();

    double autograd = t_sig.grad().to(::torch::kCPU).item<double>();
    EXPECT_NEAR(autograd, numerical, std::abs(numerical) * double(0.05));
}

TEST(TorchLJ, ZeroPairs) {
    auto real_opts = ::torch::TensorOptions()
                         .dtype(atorch::kRealDtype)
                         .device(::torch::kCUDA);
    auto int_opts = ::torch::TensorOptions()
                        .dtype(atorch::kIntDtype)
                        .device(::torch::kCUDA);

    atorch::Frame frame;
    frame.n_atoms_ = 2;
    frame.x = ::torch::zeros({2}, real_opts);
    frame.y = ::torch::zeros({2}, real_opts);
    frame.z = ::torch::zeros({2}, real_opts);
    frame.fx = ::torch::zeros({2}, real_opts);
    frame.fy = ::torch::zeros({2}, real_opts);
    frame.fz = ::torch::zeros({2}, real_opts);
    frame.energy = ::torch::zeros({1}, real_opts);
    frame.n_pairs_ = 0;
    frame.pair_i = ::torch::zeros({0}, int_opts);
    frame.pair_j = ::torch::zeros({0}, int_opts);
    frame.pair_dx = ::torch::zeros({0}, real_opts);
    frame.pair_dy = ::torch::zeros({0}, real_opts);
    frame.pair_dz = ::torch::zeros({0}, real_opts);
    frame.pair_dist = ::torch::zeros({0}, real_opts);

    auto energy =
        atorch::lj(frame, ::torch::tensor({double(1.0)}, real_opts),
                   ::torch::tensor({double(1.0)}, real_opts), double(3.0));
    cudaDeviceSynchronize();

    EXPECT_NEAR(energy.to(::torch::kCPU).item<double>(), double(0),
                double(1e-6));
}

TEST(TorchLJ, ForceIsNegativeGradient) {
    double eps = double(1.0), sig = double(1.0), cutoff = double(3.0),
           r = double(1.5);
    auto real_opts = ::torch::TensorOptions()
                         .dtype(atorch::kRealDtype)
                         .device(::torch::kCUDA);
    auto int_opts = ::torch::TensorOptions()
                        .dtype(atorch::kIntDtype)
                        .device(::torch::kCUDA);

    // Build frame with positions that require grad
    atorch::Frame frame;
    frame.n_atoms_ = 2;
    frame.x = ::torch::tensor({double(0), r}, real_opts).requires_grad_(true);
    frame.y = ::torch::zeros({2}, real_opts).requires_grad_(true);
    frame.z = ::torch::zeros({2}, real_opts).requires_grad_(true);
    frame.fx = ::torch::zeros({2}, real_opts);
    frame.fy = ::torch::zeros({2}, real_opts);
    frame.fz = ::torch::zeros({2}, real_opts);
    frame.energy = ::torch::zeros({1}, real_opts);
    frame.n_pairs_ = 1;
    frame.pair_i = ::torch::tensor({int(0)}, int_opts);
    frame.pair_j = ::torch::tensor({int(1)}, int_opts);
    frame.pair_dx = ::torch::tensor({r}, real_opts);
    frame.pair_dy = ::torch::zeros({1}, real_opts);
    frame.pair_dz = ::torch::zeros({1}, real_opts);
    frame.pair_dist = ::torch::tensor({r}, real_opts);

    auto energy = atorch::lj(frame, ::torch::tensor({eps}, real_opts),
                             ::torch::tensor({sig}, real_opts), cutoff);
    energy.backward();
    cudaDeviceSynchronize();

    // Forces written by the kernel should equal -dE/dx (the negative gradient)
    auto fx_cpu = frame.fx.to(::torch::kCPU);
    auto neg_grad_x = (-frame.x.grad()).to(::torch::kCPU);

    for (int i = 0; i < 2; ++i) {
        EXPECT_NEAR(fx_cpu[i].item<double>(), neg_grad_x[i].item<double>(),
                    double(1e-4))
            << "fx[" << i << "] does not match -dE/dx[" << i << "]";
    }
}

TEST(TorchLJ, MultiAtomConsistency) {
    // 4 atoms along x-axis: 0.0, 1.5, 3.0, 4.5
    // cutoff = 3.0, sigma = 1.0, epsilon = 1.0
    // Pairs within cutoff: (0,1) r=1.5, (1,2) r=1.5, (2,3) r=1.5
    // Pair (0,2) r=3.0 is AT cutoff => excluded (< cutoff check)
    // Pair (1,3) r=3.0 same => excluded
    // Pair (0,3) r=4.5 => beyond cutoff
    double eps = double(1.0), sig = double(1.0), cutoff = double(3.0);
    double r12 = double(1.5), r23 = double(1.5), r34 = double(1.5);

    auto real_opts = ::torch::TensorOptions()
                         .dtype(atorch::kRealDtype)
                         .device(::torch::kCUDA);
    auto int_opts = ::torch::TensorOptions()
                        .dtype(atorch::kIntDtype)
                        .device(::torch::kCUDA);

    // Build a 4-atom frame with 3 pairs
    atorch::Frame frame;
    frame.n_atoms_ = 4;
    frame.x = ::torch::tensor(
        {double(0), double(1.5), double(3.0), double(4.5)}, real_opts);
    frame.y = ::torch::zeros({4}, real_opts);
    frame.z = ::torch::zeros({4}, real_opts);
    frame.fx = ::torch::zeros({4}, real_opts);
    frame.fy = ::torch::zeros({4}, real_opts);
    frame.fz = ::torch::zeros({4}, real_opts);
    frame.energy = ::torch::zeros({1}, real_opts);
    frame.n_pairs_ = 3;
    frame.pair_i = ::torch::tensor({int(0), int(1), int(2)}, int_opts);
    frame.pair_j = ::torch::tensor({int(1), int(2), int(3)}, int_opts);
    frame.pair_dx = ::torch::tensor({r12, r23, r34}, real_opts);
    frame.pair_dy = ::torch::zeros({3}, real_opts);
    frame.pair_dz = ::torch::zeros({3}, real_opts);
    frame.pair_dist = ::torch::tensor({r12, r23, r34}, real_opts);

    auto energy = atorch::lj(frame, ::torch::tensor({eps}, real_opts),
                             ::torch::tensor({sig}, real_opts), cutoff);
    cudaDeviceSynchronize();

    // Total energy should be the sum of 3 identical pair energies
    double single_pair_energy =
        analytical_lj_energy(double(1.5), eps, sig, cutoff);
    double expected_total = double(3) * single_pair_energy;
    double computed = energy.to(::torch::kCPU).item<double>();

    EXPECT_NEAR(computed, expected_total, double(1e-4));
}

#else

TEST(TorchLJ, Disabled) { GTEST_SKIP() << "ATV_HAS_TORCH not defined"; }

#endif // ATV_HAS_TORCH
