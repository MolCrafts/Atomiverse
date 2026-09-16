/// @file tests/cuda/driver/test_long_range_contract.cu
/// @brief cuda::Driver long-range pairing at build() — no device work.

#include <gtest/gtest.h>

#include <string>

#include "atomiverse/core/stage.h"
#include "atomiverse/cuda/evaluate/evaluate.cuh"
#include "atomiverse/cuda/longrange/ewald.cuh"
#include "atomiverse/cuda/neighbor/neighbor_list.cuh"
#include "atomiverse/cuda/potentials/coul_long.cuh"

namespace {

void bind_nl(ATV::cuda::Evaluate& eval, ATV::cuda::NeighborList& nl) {
    eval.on(ATV::evaluate::Phase::evaluate).add_ref(nl, -10);
}

} // namespace

TEST(TestCudaLongRange, CoulLongAloneThrows) {
    ATV::cuda::Evaluate eval;
    ATV::cuda::NeighborList nl{9.0};
    ATV::cuda::CoulLong real(/*alpha=*/0.3, /*cutoff=*/9.0);
    bind_nl(eval, nl);
    eval.on(ATV::evaluate::Phase::evaluate).add_ref(real);
    try {
        eval.build();
        FAIL() << "expected missing reciprocal partner";
    } catch (const std::runtime_error& e) {
        EXPECT_NE(std::string(e.what()).find("reciprocal-space partner"),
                  std::string::npos)
            << e.what();
    }
}

TEST(TestCudaLongRange, EwaldAloneThrows) {
    ATV::cuda::Evaluate eval;
    ATV::cuda::KSpaceEwald rec(/*alpha=*/0.3);
    eval.on(ATV::evaluate::Phase::evaluate).add_ref(rec);
    try {
        eval.build();
        FAIL() << "expected missing real-space partner";
    } catch (const std::runtime_error& e) {
        EXPECT_NE(std::string(e.what()).find("real-space partner"),
                  std::string::npos)
            << e.what();
    }
}

TEST(TestCudaLongRange, MatchedPairBuilds) {
    ATV::cuda::Evaluate eval;
    ATV::cuda::NeighborList nl{9.0};
    ATV::cuda::CoulLong real(/*alpha=*/0.3, /*cutoff=*/9.0);
    ATV::cuda::KSpaceEwald rec(/*alpha=*/0.3);
    bind_nl(eval, nl);
    eval.on(ATV::evaluate::Phase::evaluate).add_ref(real);
    eval.on(ATV::evaluate::Phase::evaluate).add_ref(rec);
    EXPECT_NO_THROW(eval.build());
}

TEST(TestCudaLongRange, AlphaMismatchThrows) {
    ATV::cuda::Evaluate eval;
    ATV::cuda::NeighborList nl{9.0};
    ATV::cuda::CoulLong real(/*alpha=*/0.3, /*cutoff=*/9.0);
    ATV::cuda::KSpaceEwald rec(/*alpha=*/0.4);
    bind_nl(eval, nl);
    eval.on(ATV::evaluate::Phase::evaluate).add_ref(real);
    eval.on(ATV::evaluate::Phase::evaluate).add_ref(rec);
    try {
        eval.build();
        FAIL() << "expected alpha mismatch";
    } catch (const std::runtime_error& e) {
        EXPECT_NE(std::string(e.what()).find("alpha mismatch"),
                  std::string::npos)
            << e.what();
    }
}
