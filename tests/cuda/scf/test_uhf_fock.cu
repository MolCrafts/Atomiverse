/// @file test_uhf_fock.cu
/// @brief Device UHF Fock allocation-boundary validation.
#include <array>
#include <gtest/gtest.h>
#include <stdexcept>

#include "atomiverse/cuda/scf/internal/uhf_fock.cuh"

TEST(CudaUhfFock, RejectsDenseEriIndexOverflowBeforeAllocation) {
    const double alpha = 1.0;
    const double coeff = 1.0;
    const int angular_momentum = 0;
    const int atom = 0;
    ATV::cuda::Frame frame;
    for (int n : {0, 216, 46341}) {
        ATV::cuda::scf::internal::UHFFockDevice fock{
            1, n, 1, &alpha, &coeff, &angular_momentum, &atom, 0, 0};
        ATV::cuda::scf::SCFWorkspace workspace;
        ATV::cuda::scf::internal::UhfBeta beta;
        EXPECT_THROW(fock.build({nullptr, frame, 0}, workspace, beta),
                     std::invalid_argument)
            << "AO count=" << n;
        EXPECT_EQ(workspace.ERI, nullptr);
        EXPECT_EQ(workspace.H, nullptr);
    }
}

TEST(CudaUhfFock, RejectsInvalidSpinOccupationBeforeAllocation) {
    const double alpha = 1.0;
    const double coeff = 1.0;
    const int angular_momentum = 0;
    const int atom = 0;
    ATV::cuda::Frame frame;
    for (const auto& occupation :
         {std::array<int, 2>{-1, 0}, {0, -1}, {2, 0}, {0, 2}}) {
        ATV::cuda::scf::internal::UHFFockDevice fock{1,
                                                     1,
                                                     1,
                                                     &alpha,
                                                     &coeff,
                                                     &angular_momentum,
                                                     &atom,
                                                     occupation[0],
                                                     occupation[1]};
        ATV::cuda::scf::SCFWorkspace workspace;
        ATV::cuda::scf::internal::UhfBeta beta;
        EXPECT_THROW(fock.build({nullptr, frame, 0}, workspace, beta),
                     std::invalid_argument)
            << "alpha=" << occupation[0] << " beta=" << occupation[1];
        EXPECT_EQ(workspace.ERI, nullptr);
        EXPECT_EQ(workspace.H, nullptr);
        EXPECT_EQ(beta.H, nullptr);
    }
}
