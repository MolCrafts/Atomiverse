/// @file tests/cuda/test_driver_binding_kind.cu
/// @brief PhaseKind gate on CUDA categorical setters (mirrors the CPU
///        `test_driver_binding_kind` `set_observer` / `set_evaluator` cases).
///
/// `on(phase).add` is explicit and is not gated. Cross-workflow default
/// bindings are rejected when they arrive through `set_observer` /
/// `bind_defaults`.

#include <gtest/gtest.h>

#include <stdexcept>
#include <string>
#include <vector>

#include "atomiverse/core/binding.h"
#include "atomiverse/core/stage.h"
#include "atomiverse/cuda/component/component.cuh"
#include "atomiverse/cuda/dynamics/md.cuh"
#include "atomiverse/cuda/dynamics/velocity_verlet.cuh"
#include "atomiverse/cuda/evaluate/evaluate.cuh"
#include "atomiverse/cuda/minimize/minimize.cuh"
#include "atomiverse/cuda/minimize/sd.cuh"
#include "atomiverse/cuda/scf/scf_evaluator.cuh"

namespace {

bool message_mentions(const std::runtime_error& e, const std::string& needle) {
    return std::string(e.what()).find(needle) != std::string::npos;
}

class EvaluateKindProbe : public ATV::cuda::Component {
  public:
    std::string name() const override { return "evaluate_kind_probe"; }
    std::vector<ATV::DefaultBinding> default_bindings() const override {
        return {ATV::make_binding(ATV::evaluate::Phase::evaluate, 0)};
    }
    void launch(const ATV::cuda::LaunchContext&) override {}
};

} // namespace

TEST(CudaDriverBindingKind, MdSetEvaluatorScfNoThrow) {
    ATV::cuda::md::MD md;
    EXPECT_NO_THROW(md.set_evaluator(ATV::cuda::scf::SCF{2, 0.1, 1e-6, 100}));
}

TEST(CudaDriverBindingKind,
     EvaluateSetObserverVelocityVerletThrowsWithBothKinds) {
    ATV::cuda::Evaluate ev;
    try {
        ev.set_observer(ATV::cuda::md::VelocityVerlet{0.01});
        FAIL() << "expected throw — MD-kind default bindings registered "
                  "into a cuda::Evaluate driver";
    } catch (const std::runtime_error& e) {
        EXPECT_TRUE(message_mentions(e, "MD")) << e.what();
        EXPECT_TRUE(message_mentions(e, "Evaluate")) << e.what();
    }
}

TEST(CudaDriverBindingKind,
     MinimizeSetObserverVelocityVerletThrowsWithBothKinds) {
    ATV::cuda::minimizer::Minimize opt;
    try {
        opt.set_observer(ATV::cuda::md::VelocityVerlet{0.01});
        FAIL() << "expected throw — MD-kind default bindings registered "
                  "into a cuda::Minimize driver";
    } catch (const std::runtime_error& e) {
        EXPECT_TRUE(message_mentions(e, "MD")) << e.what();
        EXPECT_TRUE(message_mentions(e, "Minimizer")) << e.what();
    }
}

TEST(CudaDriverBindingKind, MdSetObserverEvaluateKindProbeThrows) {
    ATV::cuda::md::MD md;
    try {
        md.set_observer(EvaluateKindProbe{});
        FAIL() << "expected throw — Evaluate-kind default bindings "
                  "registered into an MD driver";
    } catch (const std::runtime_error& e) {
        EXPECT_TRUE(message_mentions(e, "Evaluate")) << e.what();
        EXPECT_TRUE(message_mentions(e, "MD")) << e.what();
    }
}

TEST(CudaDriverBindingKind, MdSetIntegratorSameKindNoThrow) {
    ATV::cuda::md::MD md;
    EXPECT_NO_THROW(md.set_integrator(ATV::cuda::md::VelocityVerlet{0.01}));
}

TEST(CudaDriverBindingKind, MinimizeSetOptimizerSameKindNoThrow) {
    ATV::cuda::minimizer::Minimize opt;
    EXPECT_NO_THROW(
        opt.set_optimizer(ATV::cuda::minimizer::SteepestDescent{0.01}));
}
