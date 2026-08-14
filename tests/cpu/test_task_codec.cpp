/// @file tests/cpu/test_task_codec.cpp
/// @brief TaskSpec v4 typed-payload and exact-wire contract tests.

#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "atomiverse/cpu/task/task_codec.h"
#include "atomiverse/cpu/task/task_spec.h"

using namespace ATV;
using namespace ATV::cpu::task;

namespace {

constexpr std::size_t kVersionOffset = 4;
constexpr std::size_t kTaskKindOffset = 8;
constexpr std::size_t kFirstComponentIdOffset = 13;
constexpr std::size_t kFirstPhaseOffset = 16;
// SinglePoint carries no entry-count word: its single component ID follows
// the task-kind byte directly (magic 4 + version 4 + kind 1).
constexpr std::size_t kSinglePointComponentIdOffset = 9;

template <typename T>
void write_scalar(std::vector<std::byte>& bytes, std::size_t offset, T value) {
    std::memcpy(bytes.data() + offset, &value, sizeof(T));
}

TaskSpec make_md_spec() {
    std::vector<MDComponentEntry> entries;
    entries.push_back(MDComponentEntry{NeighborListParams{6.0, 0.3},
                                       MDBinding{ATV::md::Phase::force, -10}});
    entries.push_back(MDComponentEntry{PairLJCutParams{0.0104, 3.4, 6.0},
                                       MDBinding{ATV::md::Phase::force, 0}});
    entries.push_back(
        MDComponentEntry{BondHarmonicParams{{15.2, 2.5}, {1.53, 1.09}},
                         MDBinding{ATV::md::Phase::force, 1}});
    entries.push_back(MDComponentEntry{AngleHarmonicParams{{2.7}, {1.911}},
                                       MDBinding{ATV::md::Phase::force, 2}});
    entries.push_back(MDComponentEntry{
        DihedralMultiHarmonicParams{{{1.73, -4.49, 0.776, 6.99, 0.0}}},
        MDBinding{ATV::md::Phase::pre_force, 3}});
    entries.push_back(
        MDComponentEntry{VelocityVerletParams{0.5}, std::nullopt});
    entries.push_back(
        MDComponentEntry{InitVelocityParams{300.0, 4242u}, std::nullopt});
    entries.push_back(
        MDComponentEntry{NoseHooverParams{300.0, 100.0, 0.5}, std::nullopt});
    entries.push_back(MDComponentEntry{
        FixLangevinParams{300.0, 250.0, 100.0, 0.5, 97u}, std::nullopt});
    entries.push_back(
        MDComponentEntry{FixCsvrParams{300.0, 50.0, 0.5, 13u}, std::nullopt});
    // Every NPT knob is spelled with a NON-default value so the roundtrip
    // cannot pass by accident on a codec that drops the tail fields.
    entries.push_back(MDComponentEntry{NoseHooverNptParams{300.0,
                                                           100.0,
                                                           6.2e-7,
                                                           1000.0,
                                                           0.5,
                                                           0.2,
                                                           4,
                                                           5,
                                                           {true, false, true},
                                                           350.0},
                                       std::nullopt});
    entries.push_back(MDComponentEntry{
        ParrinelloRahmanNptParams{300.0, 1e-4, 100.0, 1000.0, 0.5},
        std::nullopt});
    entries.push_back(MDComponentEntry{
        BerendsenBarostatParams{1e-4, 1000.0, 0.5, 2.0}, std::nullopt});
    // Electrostatics. All three ride the categorical (`std::nullopt`)
    // evaluator slot — it is plural, so cut-off Coulomb and the matched
    // Ewald real/reciprocal pair join the pair styles rather than replace
    // them. α is spelled identically on `CoulLongParams` and
    // `KSpacePmeParams`: an unmatched α is a driver-level build error, so
    // the wire mirror has to carry the number twice, exactly.
    entries.push_back(MDComponentEntry{PairCoulCutParams{6.0}, std::nullopt});
    entries.push_back(MDComponentEntry{CoulLongParams{0.5833333333333334, 6.0},
                                       std::nullopt});
    entries.push_back(MDComponentEntry{
        KSpacePmeParams{0.5833333333333334, 0.5, 6}, std::nullopt});
    entries.push_back(MDComponentEntry{
        ThermoParams{100}, MDBinding{ATV::md::Phase::end_of_step, 50}});
    // The first payload carrying a VARIABLE-LENGTH field (the filename): a
    // codec that writes a fixed-size struct blob, or that forgets the length
    // prefix, cannot survive the field-exact roundtrip below. Rides the
    // categorical (`std::nullopt`) recorder slot, as `set_record` does.
    entries.push_back(
        MDComponentEntry{XyzRecorderParams{"traj.xyz", 100}, std::nullopt});
    // AM1 is MD-legal as well as SinglePoint-legal (py-am1-forces): the
    // evaluator slot is plural, so a semiempirical force source rides the
    // categorical (`std::nullopt`) evaluator slot exactly like the pair
    // styles above and gives BOMD from a launcher payload. The byte layout
    // is the one `SinglePointTask` already writes (i32 charge, i32
    // multiplicity, i32 max_iter, f64 density_tol, f64 energy_tol) — the
    // widening is append-only, so `kWireVersion` stays 4.
    entries.push_back(MDComponentEntry{
        AM1EvaluatorParams{-1, 2, 50, 1e-6, 1e-8}, std::nullopt});
    return TaskSpec(MDTask{std::move(entries), 7});
}

TaskSpec make_minimize_spec() {
    std::vector<MinimizeComponentEntry> entries;
    entries.push_back(MinimizeComponentEntry{
        NeighborListParams{6.0, 0.3},
        MinimizeBinding{ATV::minimizer::Phase::force, -10}});
    entries.push_back(MinimizeComponentEntry{
        PairLJCutParams{0.0104, 3.4, 6.0},
        MinimizeBinding{ATV::minimizer::Phase::force, 0}});
    entries.push_back(
        MinimizeComponentEntry{LBFGSParams{7, 0.02, 12}, std::nullopt});
    return TaskSpec(
        MinimizeTask{std::move(entries), 55,
                     ForceConvergedParams{0.02, 0.01, 0.003, 0.002, 2e-6}});
}

void expect_runtime_error(const std::vector<std::byte>& bytes,
                          const std::string& needle) {
    try {
        (void)unpack(bytes.data(), bytes.size());
        FAIL() << "tampered payload was accepted";
    } catch (const std::runtime_error& err) {
        EXPECT_NE(std::string(err.what()).find(needle), std::string::npos)
            << err.what();
    }
}

TEST(TaskCodec, ComponentIdsAreExplicitAndStable) {
    EXPECT_EQ(static_cast<std::uint16_t>(ComponentId::PairLJCut), 0u);
    EXPECT_EQ(static_cast<std::uint16_t>(ComponentId::AM1Evaluator), 9u);
    EXPECT_EQ(static_cast<std::uint16_t>(ComponentId::LBFGS), 10u);
    EXPECT_EQ(static_cast<std::uint16_t>(ComponentId::FixLangevin), 11u);
    EXPECT_EQ(static_cast<std::uint16_t>(ComponentId::FixCSVR), 12u);
    EXPECT_EQ(static_cast<std::uint16_t>(ComponentId::NoseHooverNPT), 13u);
    EXPECT_EQ(static_cast<std::uint16_t>(ComponentId::ParrinelloRahmanNPT),
              14u);
    EXPECT_EQ(static_cast<std::uint16_t>(ComponentId::BerendsenBarostat), 15u);
    EXPECT_EQ(static_cast<std::uint16_t>(ComponentId::PairCoulCut), 16u);
    EXPECT_EQ(static_cast<std::uint16_t>(ComponentId::CoulLong), 17u);
    EXPECT_EQ(static_cast<std::uint16_t>(ComponentId::KSpacePME), 18u);
    EXPECT_EQ(static_cast<std::uint16_t>(ComponentId::XyzRecorder), 19u);
    EXPECT_EQ(component_id(MDComponentParams{VelocityVerletParams{}}),
              ComponentId::VelocityVerlet);
    EXPECT_EQ(component_id(MinimizeComponentParams{LBFGSParams{}}),
              ComponentId::LBFGS);
    // The three electrostatic params are structurally similar (doubles, and
    // two of them lead with α), so pin the variant→ID map as well as the
    // numbers: a swapped arm is otherwise invisible on the wire.
    EXPECT_EQ(component_id(MDComponentParams{PairCoulCutParams{}}),
              ComponentId::PairCoulCut);
    EXPECT_EQ(component_id(MDComponentParams{CoulLongParams{}}),
              ComponentId::CoulLong);
    EXPECT_EQ(component_id(MDComponentParams{KSpacePmeParams{}}),
              ComponentId::KSpacePME);
    // XyzRecorder is the only MD recorder besides Thermo and both are
    // `{… , int freq}`-shaped, so pin the variant→ID map here too: a swapped
    // arm would otherwise decode a filename out of a Thermo frequency.
    EXPECT_EQ(component_id(MDComponentParams{XyzRecorderParams{}}),
              ComponentId::XyzRecorder);
    // One params struct, one ID, two legal homes: `AM1EvaluatorParams` now
    // sits in BOTH the MD and the SinglePoint payload, and both must map to
    // the same wire ID 9 — a second ID for the MD arm would make the same
    // component decode differently depending on the task that carries it.
    EXPECT_EQ(component_id(MDComponentParams{AM1EvaluatorParams{}}),
              ComponentId::AM1Evaluator);
}

// v4: the NoseHooverNptParams payload grew (drag / mtchain / mpchain /
// couple_axis / T_stop), so a v3 payload is no longer readable — the
// exact-match version gate in `unpack` is what makes that break loud.
TEST(TaskCodec, WireVersionIsFour) { EXPECT_EQ(TaskSpec::kWireVersion, 4u); }

TEST(TaskCodec, MdRoundtripIsFieldExact) {
    const TaskSpec spec = make_md_spec();
    const auto bytes = pack(spec);
    const TaskSpec back = unpack(bytes.data(), bytes.size());
    ASSERT_EQ(back.kind(), TaskKind::MD);
    EXPECT_TRUE(back.md() == spec.md());
}

TEST(TaskCodec, SinglePointRoundtripIsFieldExact) {
    AM1EvaluatorParams p{-1, 2, 50, 1e-6, 1e-8};
    const TaskSpec spec(SinglePointTask{p});
    const auto bytes = pack(spec);
    const TaskSpec back = unpack(bytes.data(), bytes.size());
    ASSERT_EQ(back.kind(), TaskKind::SinglePoint);
    EXPECT_EQ(back.single_point().evaluator, p);
}

TEST(TaskCodec, MinimizeRoundtripKeepsTypedPhaseBudgetAndAllTolerances) {
    const TaskSpec spec = make_minimize_spec();
    const auto bytes = pack(spec);
    const TaskSpec back = unpack(bytes.data(), bytes.size());
    ASSERT_EQ(back.kind(), TaskKind::Minimize);
    EXPECT_TRUE(back.minimize() == spec.minimize());
    ASSERT_TRUE(back.minimize().entries[0].binding);
    EXPECT_EQ(back.minimize().entries[0].binding->phase,
              ATV::minimizer::Phase::force);
}

TEST(TaskCodec, VersionTwoAndUnknownVersionsAreRejected) {
    auto bytes = pack(make_md_spec());
    write_scalar<std::uint32_t>(bytes, kVersionOffset, 2);
    expect_runtime_error(bytes, "version 2");
    bytes = pack(make_md_spec());
    write_scalar<std::uint32_t>(bytes, kVersionOffset, 99);
    expect_runtime_error(bytes, "version 99");
}

TEST(TaskCodec, UnknownTaskAndComponentIdsAreRejected) {
    auto bytes = pack(make_md_spec());
    bytes[kTaskKindOffset] = std::byte{9};
    expect_runtime_error(bytes, "task_kind 9");
    bytes = pack(make_md_spec());
    write_scalar<std::uint16_t>(bytes, kFirstComponentIdOffset, 999);
    expect_runtime_error(bytes, "component_id 999");
}

TEST(TaskCodec, CrossKindComponentsNameSourceAndTarget) {
    auto md_bytes = pack(make_md_spec());
    write_scalar<std::uint16_t>(md_bytes, kFirstComponentIdOffset,
                                static_cast<std::uint16_t>(ComponentId::LBFGS));
    expect_runtime_error(md_bytes, "belongs to Minimize, not MD");

    auto min_bytes = pack(make_minimize_spec());
    write_scalar<std::uint16_t>(
        min_bytes, kFirstComponentIdOffset,
        static_cast<std::uint16_t>(ComponentId::VelocityVerlet));
    expect_runtime_error(min_bytes, "belongs to MD, not Minimize");

    // AM1 has TWO legal homes (MD and SinglePoint) and one illegal one, so
    // its diagnostic names both — the dual "MD/Minimize" shape PairLJCut and
    // NeighborList already use. A single-home string here would tell a user
    // rebuilding a Minimize payload that AM1 is SinglePoint-only, which is
    // no longer true.
    auto min_am1_bytes = pack(make_minimize_spec());
    write_scalar<std::uint16_t>(
        min_am1_bytes, kFirstComponentIdOffset,
        static_cast<std::uint16_t>(ComponentId::AM1Evaluator));
    expect_runtime_error(min_am1_bytes,
                         "belongs to MD/SinglePoint, not Minimize");

    // The reverse direction of the same rule: SinglePoint carries exactly one
    // component and it is AM1. Widening AM1 into MD must not widen the
    // SinglePoint payload into an anything-goes slot — any other ID there is
    // still refused, naming where that component does belong.
    auto sp_bytes = pack(TaskSpec(SinglePointTask{AM1EvaluatorParams{}}));
    write_scalar<std::uint16_t>(
        sp_bytes, kSinglePointComponentIdOffset,
        static_cast<std::uint16_t>(ComponentId::VelocityVerlet));
    expect_runtime_error(sp_bytes, "belongs to MD, not SinglePoint");
}

TEST(TaskCodec, EachPhaseSlotIsCheckedAgainstItsOwnDomain) {
    auto bytes = pack(make_minimize_spec());
    write_scalar<std::uint32_t>(
        bytes, kFirstPhaseOffset,
        static_cast<std::uint32_t>(ATV::slot_count<ATV::minimizer::Phase>()));
    expect_runtime_error(bytes, "minimizer::Phase");
}

TEST(TaskCodec, InvalidBudgetsLbfgsAndTolerancesFailBeforePacking) {
    EXPECT_THROW(TaskSpec(MDTask{{}, 0}), std::invalid_argument);
    EXPECT_THROW(TaskSpec(MinimizeTask{{}, -1, {}}), std::invalid_argument);

    MinimizeTask bad_lbfgs{
        {MinimizeComponentEntry{LBFGSParams{0, 0.01, 10}, std::nullopt}},
        10,
        {}};
    EXPECT_THROW(TaskSpec(std::move(bad_lbfgs)), std::invalid_argument);

    MinimizeTask bad_tol{{}, 10, {}};
    bad_tol.convergence.ftol = std::numeric_limits<double>::quiet_NaN();
    EXPECT_THROW(TaskSpec(std::move(bad_tol)), std::invalid_argument);
}

TEST(TaskCodec, TypedCountSentinelsAreRejectedInMemory) {
    MDTask md{{MDComponentEntry{PairLJCutParams{1, 1, 2},
                                MDBinding{ATV::md::Phase::COUNT, 0}}},
              1};
    EXPECT_THROW(TaskSpec(std::move(md)), std::invalid_argument);

    MinimizeTask min{
        {MinimizeComponentEntry{
            LBFGSParams{}, MinimizeBinding{ATV::minimizer::Phase::COUNT, 0}}},
        1,
        {}};
    EXPECT_THROW(TaskSpec(std::move(min)), std::invalid_argument);
}

TEST(TaskCodec, TruncationAndTrailingBytesAreRejected) {
    const auto full = pack(make_minimize_spec());
    EXPECT_THROW((void)unpack(full.data(), full.size() / 2),
                 std::runtime_error);
    auto trailing = full;
    trailing.push_back(std::byte{0});
    EXPECT_THROW((void)unpack(trailing.data(), trailing.size()),
                 std::runtime_error);
}

TEST(TaskCodec, TaskResultRoundtripAndTruncationAreExact) {
    const TaskResult result{17, true, "force_converged"};
    const auto bytes = pack_result(result);
    EXPECT_EQ(unpack_result(bytes.data(), bytes.size()), result);
    EXPECT_THROW((void)unpack_result(bytes.data(), bytes.size() - 1),
                 std::runtime_error);
}

TEST(TaskCodec, EmptyMdAndMinimizePayloadsRemainCodecLevelCitizens) {
    const TaskSpec md(MDTask{{}, 3});
    const auto md_bytes = pack(md);
    EXPECT_TRUE(unpack(md_bytes.data(), md_bytes.size()).md().entries.empty());

    const TaskSpec min(MinimizeTask{{}, 3, {}});
    const auto min_bytes = pack(min);
    EXPECT_TRUE(
        unpack(min_bytes.data(), min_bytes.size()).minimize().entries.empty());
}

} // namespace
