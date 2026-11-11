/// @file tests/cpu/path/test_path_validation.cpp
/// @brief Unit: `cpu::PathDriver::validate_components` recognises a nested
///        Gaussian method-driver as the band's force source.
///
/// Mirrors `include/atomiverse/cpu/path/path_driver.h` (`cpu::PathDriver`).
/// Spec: `.claude/specs/esmm-contract-02-validation-walks` — ac-002 / ac-004.
///
/// `HF` is a pure polymorphic tag (`cpu/component/hf.h`), **not** a Component.
/// `on(phase).add` copies `provides_force` onto the slot, so a band whose
/// force source is a nested method is visible to `for_each_pending`.
///
/// Scope is **recognition**, which is what `validate_components` does and where
/// `build()` calls it. Whether the recognised node then actually *runs* per
/// replica is `esmm-contract-03` (`Runtime::run_slot_` skips nested drivers, so
/// the stub below is never launched); asserting band forces here would assert
/// that out-of-scope behaviour, so these cases stop at `build()`.

#include <gtest/gtest.h>

#include <stdexcept>
#include <string>

#include "atomiverse/core/stage.h"
#include "atomiverse/cpu/component/hf.h"
#include "atomiverse/cpu/component/optimizer.h"
#include "atomiverse/cpu/driver/driver.h"
#include "atomiverse/cpu/driver/driver_node.h"
#include "atomiverse/cpu/path/path_driver.h"

namespace {

/// @brief Band optimizer that moves nothing — present only to satisfy the
///        `Optimizer` half of the requirement.
class FrozenOptimizer : public ATV::cpu::Optimizer {
  public:
    std::string name() const override { return "frozen_optimizer"; }
    void launch(const ATV::cpu::LaunchContext&) override {}
};

/// @brief The exact shape a Gaussian method presents to an outer driver:
///        `Driver<scf::Phase>` (hence a `cpu::DriverNode`) + the pure `HF` kind
///        tag, and **not** a `Component`.
///
/// `cpu::scf::RHF` is this plus physics (`MeanFieldMethod` derives the same two
/// bases). The three `DriverNode` lifecycle ops — `build_on`,
/// `run_nested_erased`, `destroy_on` — are already implemented by
/// `cpu::Driver<PhaseEnum>`, so the only pure virtual left to satisfy is
/// `validate_components()`: a bare loop requires nothing. A stub keeps the case
/// off `ATV_BUILD_HF` and in the milliseconds.
struct StubHfNode : ATV::cpu::Driver<ATV::scf::Phase>, ATV::cpu::HF {
    void validate_components() const override {}
};

/// @brief A nested loop node that is NOT a force source: same
///        `Driver<scf::Phase>` node shape, **no** `HF` tag.
struct StubBareNode : ATV::cpu::Driver<ATV::scf::Phase> {
    void validate_components() const override {}
};

/// @brief Substring probe on a thrown validation message.
bool mentions(const std::runtime_error& e, const std::string& needle) {
    return std::string(e.what()).find(needle) != std::string::npos;
}

} // namespace

/// ac-002 — a nested HF node IS the band's force source. RED today: `build()`
/// throws "missing required Potential / Bias / HF / Semiempirical / PlaneWave /
/// GPW" because the component-only walk never looks at a driver node.
TEST(PathValidation, NestedHfNodeSatisfiesForceValidation) {
    FrozenOptimizer opt;
    StubHfNode method;

    ATV::cpu::PathDriver band;
    band.on(ATV::path::Phase::search_direction).add_ref(opt, /*order=*/0);
    // NOTE: `add`/`add_ref`'s second positional argument is `order`, the third
    // is the nested budget — not `add(node, budget)`.
    band.on(ATV::path::Phase::force)
        .add_ref(method, /*order=*/0, /*budget=*/0); // the ONLY force source

    EXPECT_NO_THROW(band.build());
}

/// ac-002 (second witness) — with the HF node counted, the *only* thing left
/// missing from an optimizer-less band is the `Optimizer`. Which requirement
/// the driver reports is therefore a direct read-out of whether it was seen:
/// today the message names the force kinds, after the walk fix it names
/// `Optimizer`.
TEST(PathValidation, NestedHfNodeIsCreditedAsForceNotOptimizer) {
    StubHfNode method;

    ATV::cpu::PathDriver band;
    band.on(ATV::path::Phase::force).add_ref(method, /*order=*/0, /*budget=*/0);

    try {
        band.build();
        FAIL() << "expected throw — the band has no Optimizer";
    } catch (const std::runtime_error& e) {
        EXPECT_TRUE(mentions(e, "Optimizer"))
            << "the nested HF node was not credited as the force source: "
            << e.what();
    }
}

/// ac-004 — no source of any kind (neither component kind nor HF node) still
/// throws. The guard gains visibility, never permissiveness.
TEST(PathValidation, StillThrowsWithNoForceSource) {
    FrozenOptimizer opt;

    ATV::cpu::PathDriver band;
    band.on(ATV::path::Phase::search_direction).add_ref(opt, /*order=*/0);

    try {
        band.build();
        FAIL() << "expected throw — the band has no force source";
    } catch (const std::runtime_error& e) {
        EXPECT_TRUE(mentions(e, "force source")) << e.what();
    }
}

/// ac-004 — a nested node WITHOUT the `HF` tag is not a force source either.
/// Pins the GREEN against a `dynamic_cast<const DriverNode*>`-shaped shortcut
/// that would count any nested loop.
TEST(PathValidation, StillThrowsWithNonHfNestedNode) {
    FrozenOptimizer opt;
    StubBareNode not_a_method;

    ATV::cpu::PathDriver band;
    band.on(ATV::path::Phase::search_direction).add_ref(opt, /*order=*/0);
    band.on(ATV::path::Phase::force)
        .add_ref(not_a_method, /*order=*/0, /*budget=*/0);

    EXPECT_THROW(band.build(), std::runtime_error);
}
