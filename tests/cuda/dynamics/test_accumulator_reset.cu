/// @brief ZeroEnergy::launch resets every force contribution, preserving motion outputs.
#include "atomiverse/cpu/frame/transfer.h"
#include "atomiverse/cuda/device/error.cuh"
#include "atomiverse/cuda/dynamics/init.cuh"
#include "atomiverse/cuda/frame/transfer.cuh"
#include <gtest/gtest.h>

TEST(ZeroEnergy, ClearsImproperAndPreservesKineticState) {
    ATV::cpu::Frame host(0);
    auto device = ATV::cpu::to_cuda(host);
    ATV::SystemScalars values;
    values.energy = 1;
    values.energy_pair = 2;
    values.energy_bond = 3;
    values.energy_angle = 4;
    values.energy_dihedral = 5;
    values.energy_improper = 6;
    values.kinetic = 7;
    values.conserved = 8;
    values.stress[0] = 9;
    ATV_CUDA_CHECK(cudaMemcpy(device.scalars, &values, sizeof(values),
                              cudaMemcpyHostToDevice));
    ATV::cuda::Context runtime;
    ATV::cuda::md::ZeroEnergy zero;
    zero.launch({runtime, device, 0, 0});
    ATV_CUDA_CHECK(cudaMemcpy(&values, device.scalars, sizeof(values),
                              cudaMemcpyDeviceToHost));
    EXPECT_EQ(values.energy, 0);
    EXPECT_EQ(values.energy_pair, 0);
    EXPECT_EQ(values.energy_bond, 0);
    EXPECT_EQ(values.energy_angle, 0);
    EXPECT_EQ(values.energy_dihedral, 0);
    EXPECT_EQ(values.energy_improper, 0);
    EXPECT_EQ(values.kinetic, 7);
    EXPECT_EQ(values.conserved, 8);
    EXPECT_EQ(values.stress[0], 9);
    ATV::cuda::destroy(device);
}

TEST(ZeroKinetic, ClearsAllKineticStressWithoutChangingForceResults) {
    ATV::cpu::Frame host(0);
    auto device = ATV::cpu::to_cuda(host);
    ATV::SystemScalars values;
    values.energy = 11;
    values.kinetic = 12;
    values.conserved = 13;
    for (int c = 0; c < 6; ++c) {
        values.stress[c] = 20 + c;
        values.stress_kinetic[c] = 30 + c;
    }
    ATV_CUDA_CHECK(cudaMemcpy(device.scalars, &values, sizeof(values),
                              cudaMemcpyHostToDevice));
    ATV::cuda::Context runtime;
    ATV::cuda::md::ZeroKinetic zero;
    zero.launch({runtime, device, 0, 0});
    ATV_CUDA_CHECK(cudaMemcpy(&values, device.scalars, sizeof(values),
                              cudaMemcpyDeviceToHost));
    EXPECT_EQ(values.energy, 11);
    EXPECT_EQ(values.kinetic, 0);
    EXPECT_EQ(values.conserved, 13);
    for (int c = 0; c < 6; ++c) {
        EXPECT_EQ(values.stress[c], 20 + c);
        EXPECT_EQ(values.stress_kinetic[c], 0);
    }
    ATV::cuda::destroy(device);
}
