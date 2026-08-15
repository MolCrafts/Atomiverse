/// @file tests/cpu/test_task_codec.cpp
/// @brief TaskSpec v5 typed-payload and exact-wire contract tests.

#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <variant>
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
    // Plane-wave DFT as an MD force source (py-pw-forces): like AM1 above it
    // rides the categorical (`std::nullopt`) evaluator slot, giving
    // plane-wave BOMD from a launcher payload. Wire ID 20 with a fresh
    // payload, so `kWireVersion` stays 4 (append-only).
    //
    // Fifteen fields — the widest mirror on the wire — so every one is
    // spelled with a NON-DEFAULT value (the NPT house rule): a codec that
    // drops a tail field would otherwise decode back into the `cpu::pw::PW`
    // default and the roundtrip would pass on a hole. Two traps that make
    // the rule concrete here:
    //   * `xc = 0` is LSDA. `XCType::PBE == 1` is the default, so a `1`
    //     here would be exactly the silent-drop hole this entry closes.
    //   * `smearing = 2` is Methfessel-Paxton (`Smearing::none == 0` is the
    //     default), and `mp_order = 2` is the order that only that arm
    //     reads — both enums cross the wire as i32.
    // `upf_path` stays empty on purpose: it is mutually exclusive with
    // `gth_name`, so "both non-default" is not a legal payload. Emptiness is
    // a value the string codec has to survive too (zero-length prefix).
    entries.push_back(
        MDComponentEntry{PwEvaluatorParams{/*upf_path=*/"",
                                           /*gth_name=*/"GTH-PBE-q4",
                                           /*ecut_ry=*/4.0,
                                           /*k_mesh=*/2,
                                           /*tol=*/1e-4,
                                           /*max_iter=*/3,
                                           /*mix_beta=*/0.5,
                                           /*pulay_depth=*/3,
                                           /*xc=*/0,
                                           /*spin_polarized=*/true,
                                           /*multiplicity=*/2,
                                           /*smearing=*/2,
                                           /*degauss=*/0.05,
                                           /*mp_order=*/2,
                                           /*extra_bands=*/6},
                         std::nullopt});
    // ── The widened real-space catalogue (py-pair-styles) ────────────────
    //
    // Nine appended IDs (27-35) with fresh payloads, so `kWireVersion` stays
    // 5: no existing byte layout moves. All nine ride the categorical
    // (`std::nullopt`) evaluator slot — the slot is plural, so a pair style
    // JOINS `PairLJCut` above rather than replacing it, and the three bonded
    // styles join it the way `BondHarmonic` already does.
    //
    // The house rule applies throughout: EVERY field of every mirror is
    // spelled with a non-default, mutually distinct value. Six of the nine
    // are "some doubles then a cutoff" and would decode into each other's
    // shapes unnoticed otherwise.

    // `PairMorseParams` — D [eV], α [Å⁻¹], r₀ [Å], cutoff [Å]. Four doubles,
    // the same arity as `PairBuckParams` below, so the two are told apart
    // only by their IDs; the values differ by orders of magnitude here so a
    // swapped arm is a visible diff and not a plausible one.
    entries.push_back(MDComponentEntry{
        PairMorseParams{/*D=*/0.35, /*alpha=*/1.4, /*r0=*/2.9, /*cutoff=*/7.0},
        std::nullopt});
    // `PairBuckParams` — A [eV], ρ [Å], C [eV·Å⁶], cutoff [Å].
    entries.push_back(MDComponentEntry{PairBuckParams{/*A=*/1500.0,
                                                      /*rho=*/0.29,
                                                      /*C=*/25.0,
                                                      /*cutoff=*/8.0},
                                       std::nullopt});
    // `PairBornParams` — the MULTI-TYPE Fumi-Tosi table, mirrored as PARALLEL
    // VECTORS (the `BondHarmonicParams` shape) rather than as a vector of
    // `BornTypeCoeff`: the codec's variable-length primitive is a
    // length-prefixed run of one scalar type, so a vector-of-struct has no
    // layout. The single-type C++ constructor stays expressible as one-row
    // vectors — that is what the materializer parity test uses.
    //
    // TWO rows, keyed by the atomic numbers a real frame carries in
    // `kAtom_type` (Na = 11, Cl = 17 — the Z-keying reality, not 0/1): a
    // Na-Cl row and a Cl-Cl row. `type_i` and `type_j` therefore differ from
    // each other AND row to row, so a codec that swapped the two integer
    // vectors, or wrote one row twice, cannot survive the roundtrip. Five
    // parallel double vectors on top of that is the widest parallel-vector
    // payload on the wire.
    entries.push_back(MDComponentEntry{PairBornParams{/*type_i=*/{11, 17},
                                                      /*type_j=*/{17, 17},
                                                      /*A=*/{0.2637, 1.6},
                                                      /*rho=*/{0.317, 0.338},
                                                      /*sigma=*/{2.755, 3.170},
                                                      /*C=*/{1.05, 72.4},
                                                      /*D=*/{0.499, 145.4},
                                                      /*cutoff=*/9.0},
                                       std::nullopt});
    // `PairYukawaParams` — A [eV·Å], κ [Å⁻¹], cutoff [Å].
    entries.push_back(MDComponentEntry{
        PairYukawaParams{/*A=*/2.5, /*kappa=*/0.8, /*cutoff=*/7.5},
        std::nullopt});
    // `PairWcaParams` — ε [eV], σ [Å] and NOTHING ELSE: WCA's cutoff is
    // derived (2^(1/6)·σ), not chosen, so the mirror is the only two-double
    // pair payload on the wire. A codec that appended a cutoff "for
    // symmetry" would be caught by `require_end()`, not by these values.
    entries.push_back(MDComponentEntry{
        PairWcaParams{/*epsilon=*/0.0125, /*sigma=*/3.15}, std::nullopt});
    // `PairDpdParams` — a [eV/Å], γ, T [K], cutoff [Å], dt [fs], seed.
    // The seed is a **u64** (`cpu::PairDpd`'s own type), where `FixLangevin`
    // and `FixCSVR` carry u32 ones: 99 is deliberately small so a codec that
    // wrote it through the existing `u32` helper would still roundtrip the
    // VALUE — the width is pinned by the layout, and the field-exact
    // roundtrip plus `require_end()` is what catches four missing bytes.
    entries.push_back(MDComponentEntry{PairDpdParams{/*a=*/25.0,
                                                     /*gamma=*/4.5,
                                                     /*temperature=*/300.0,
                                                     /*cutoff=*/2.5,
                                                     /*dt=*/0.5,
                                                     /*seed=*/99u},
                                       std::nullopt});
    // `BondFeneParams` — per-bond-type K [eV/Å²] and R₀ [Å]. Same two-vector
    // shape as `BondHarmonicParams`, hence its own ID and two rows here: a
    // swapped arm would decode a maximum extension as an equilibrium length.
    entries.push_back(MDComponentEntry{
        BondFeneParams{/*K=*/{30.0, 21.5}, /*R0=*/{1.5, 2.0}}, std::nullopt});
    // `DihedralCharmmParams` — K [eV], multiplicity n (≥ 1), δ [rad]. The
    // FIRST payload on the wire carrying a variable-length INTEGER vector;
    // the existing `vec` helper is f64-only, so `n` needs its own
    // length-prefixed run. Two types with DIFFERENT multiplicities, because
    // an implementation that pushed `n` through the double vector would
    // roundtrip {1, 3} unharmed as doubles and only differ in bytes.
    entries.push_back(MDComponentEntry{
        DihedralCharmmParams{/*K=*/{0.25, 0.6},
                             /*n=*/{1, 3},
                             /*delta=*/{3.141592653589793, 0.7853981633974483}},
        std::nullopt});
    // `ImproperHarmonicParams` — K [eV/rad²], χ₀ [rad]. Two-vector shape
    // again, so distinct values in both rows and both vectors.
    entries.push_back(MDComponentEntry{
        ImproperHarmonicParams{/*K=*/{0.4, 0.9},
                               /*chi0=*/{1.5707963267948966, 0.2}},
        std::nullopt});
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
    // ALL SEVEN optimizer mirrors ride one spec. A Minimize driver accepts
    // exactly one Optimizer — but the codec layer is validation-free: an
    // entry list is a list, and singularity is the driver's rule, enforced
    // at `Minimize::build()` (`validate_components`), not on the wire. So
    // one spec is the cheapest way to make the roundtrip cover every new
    // payload AND their relative order in the byte stream.
    //
    // Every field of every mirror is spelled with a NON-DEFAULT value (the
    // `NoseHooverNptParams` house rule): a codec that drops a tail field
    // would otherwise decode back into the C++ default and the field-exact
    // roundtrip would pass on a hole.
    entries.push_back(
        MinimizeComponentEntry{CGParams{/*alpha0=*/0.03}, std::nullopt});
    entries.push_back(MinimizeComponentEntry{FIREParams{/*dt0=*/0.2,
                                                        /*dt_max=*/2.5,
                                                        /*n_min=*/3,
                                                        /*f_inc=*/1.2,
                                                        /*f_dec=*/0.4,
                                                        /*alpha0=*/0.15,
                                                        /*f_alpha=*/0.98,
                                                        /*max_step=*/0.3},
                                             std::nullopt});
    entries.push_back(
        MinimizeComponentEntry{IsotropicCellRelaxParams{/*P_target=*/1e-4,
                                                        /*alpha0=*/0.08,
                                                        /*max_scale=*/0.07,
                                                        /*bulk_mod=*/0.5},
                               std::nullopt});
    // The per-axis mask is the only bool triple on the Minimize wire, so it
    // gets an ASYMMETRIC pattern: {false, false, true} survives neither a
    // codec that writes one bool for all three nor one that swaps two of
    // them. Not all-false on purpose — that would be a cell relax that
    // relaxes nothing, which is not a payload anyone should be able to
    // point at as precedent.
    entries.push_back(
        MinimizeComponentEntry{AnisotropicCellRelaxParams{/*P_target=*/2e-4,
                                                          /*alpha0=*/0.09,
                                                          /*max_scale=*/0.06,
                                                          /*bulk_mod=*/0.75,
                                                          /*relax_x=*/false,
                                                          /*relax_y=*/false,
                                                          /*relax_z=*/true},
                               std::nullopt});
    entries.push_back(
        MinimizeComponentEntry{TriclinicCellRelaxParams{/*alpha0=*/0.03,
                                                        /*max_strain=*/0.04,
                                                        /*bulk_mod=*/0.6},
                               std::nullopt});
    entries.push_back(
        MinimizeComponentEntry{JointIsotropicRelaxParams{/*P_target=*/3e-4,
                                                         /*atom_step=*/0.02,
                                                         /*alpha0=*/0.07,
                                                         /*max_scale=*/0.04,
                                                         /*bulk_mod=*/0.8},
                               std::nullopt});
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
    EXPECT_EQ(static_cast<std::uint16_t>(ComponentId::PWEvaluator), 20u);
    // The full CPU minimize domain (py-minimize-full): six optimizer mirrors
    // appended after `PWEvaluator`, in the spec's declared order. The VALUE
    // is what crosses the wire, so these six numbers are frozen the moment
    // they ship — a renumbering is a decoder change, not a refactor.
    EXPECT_EQ(static_cast<std::uint16_t>(ComponentId::CG), 21u);
    EXPECT_EQ(static_cast<std::uint16_t>(ComponentId::FIRE), 22u);
    EXPECT_EQ(static_cast<std::uint16_t>(ComponentId::IsotropicCellRelax), 23u);
    EXPECT_EQ(static_cast<std::uint16_t>(ComponentId::AnisotropicCellRelax),
              24u);
    EXPECT_EQ(static_cast<std::uint16_t>(ComponentId::TriclinicCellRelax), 25u);
    EXPECT_EQ(static_cast<std::uint16_t>(ComponentId::JointIsotropicRelax),
              26u);
    EXPECT_EQ(component_id(MDComponentParams{VelocityVerletParams{}}),
              ComponentId::VelocityVerlet);
    EXPECT_EQ(component_id(MinimizeComponentParams{LBFGSParams{}}),
              ComponentId::LBFGS);
    // The three cell relaxers are structurally near-identical — the first
    // four doubles of `IsotropicCellRelaxParams` and
    // `AnisotropicCellRelaxParams` are the same four quantities, and
    // `JointIsotropicRelaxParams` differs only by an inserted `atom_step` —
    // so pin the variant→ID map as well as the numbers. A swapped arm is
    // invisible on the wire otherwise: the bytes still decode, into the
    // wrong optimizer.
    EXPECT_EQ(component_id(MinimizeComponentParams{CGParams{}}),
              ComponentId::CG);
    EXPECT_EQ(component_id(MinimizeComponentParams{FIREParams{}}),
              ComponentId::FIRE);
    EXPECT_EQ(component_id(MinimizeComponentParams{IsotropicCellRelaxParams{}}),
              ComponentId::IsotropicCellRelax);
    EXPECT_EQ(
        component_id(MinimizeComponentParams{AnisotropicCellRelaxParams{}}),
        ComponentId::AnisotropicCellRelax);
    EXPECT_EQ(component_id(MinimizeComponentParams{TriclinicCellRelaxParams{}}),
              ComponentId::TriclinicCellRelax);
    EXPECT_EQ(
        component_id(MinimizeComponentParams{JointIsotropicRelaxParams{}}),
        ComponentId::JointIsotropicRelax);
    // The widened real-space catalogue (py-pair-styles): nine mirrors
    // appended after `JointIsotropicRelax`, in the spec's declared order.
    // The VALUE is what crosses the wire, so these nine numbers are frozen
    // the moment they ship — a renumbering is a decoder change, not a
    // refactor.
    EXPECT_EQ(static_cast<std::uint16_t>(ComponentId::PairMorse), 27u);
    EXPECT_EQ(static_cast<std::uint16_t>(ComponentId::PairBuck), 28u);
    EXPECT_EQ(static_cast<std::uint16_t>(ComponentId::PairBorn), 29u);
    EXPECT_EQ(static_cast<std::uint16_t>(ComponentId::PairYukawa), 30u);
    EXPECT_EQ(static_cast<std::uint16_t>(ComponentId::PairWCA), 31u);
    EXPECT_EQ(static_cast<std::uint16_t>(ComponentId::PairDpd), 32u);
    EXPECT_EQ(static_cast<std::uint16_t>(ComponentId::BondFene), 33u);
    EXPECT_EQ(static_cast<std::uint16_t>(ComponentId::DihedralCharmm), 34u);
    EXPECT_EQ(static_cast<std::uint16_t>(ComponentId::ImproperHarmonic), 35u);
    // …and the variant→ID map for each, for the reason the cell relaxers
    // above already demonstrate: `PairMorseParams` and `PairBuckParams` are
    // both four doubles, `PairYukawaParams` is three, and the three bonded
    // mirrors are all "two vectors" (`BondFeneParams` /
    // `ImproperHarmonicParams` identically so). A swapped arm still decodes
    // — into the wrong potential, with entirely plausible numbers.
    EXPECT_EQ(component_id(MDComponentParams{PairMorseParams{}}),
              ComponentId::PairMorse);
    EXPECT_EQ(component_id(MDComponentParams{PairBuckParams{}}),
              ComponentId::PairBuck);
    EXPECT_EQ(component_id(MDComponentParams{PairBornParams{}}),
              ComponentId::PairBorn);
    EXPECT_EQ(component_id(MDComponentParams{PairYukawaParams{}}),
              ComponentId::PairYukawa);
    EXPECT_EQ(component_id(MDComponentParams{PairWcaParams{}}),
              ComponentId::PairWCA);
    EXPECT_EQ(component_id(MDComponentParams{PairDpdParams{}}),
              ComponentId::PairDpd);
    EXPECT_EQ(component_id(MDComponentParams{BondFeneParams{}}),
              ComponentId::BondFene);
    EXPECT_EQ(component_id(MDComponentParams{DihedralCharmmParams{}}),
              ComponentId::DihedralCharmm);
    EXPECT_EQ(component_id(MDComponentParams{ImproperHarmonicParams{}}),
              ComponentId::ImproperHarmonic);
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
    // `PwEvaluatorParams` is the second payload that LEADS with a string
    // (`XyzRecorderParams` was the first), so pin its variant→ID map too: a
    // swapped arm would decode a trajectory filename as a pseudopotential
    // name and vice versa, which the length-prefixed string codec cannot
    // notice on its own.
    EXPECT_EQ(component_id(MDComponentParams{PwEvaluatorParams{}}),
              ComponentId::PWEvaluator);
}

// v5: `MinimizeTask::convergence` stopped being a bare `ForceConvergedParams`
// blob and became a criterion VARIANT (Force / Pressure / Stress), so the
// Minimize payload now ends in a `u16` criterion id plus that criterion's own
// fields. A v4 payload's five bare doubles would decode as an id followed by
// garbage — the NPT precedent: a LAYOUT change is a bump, an appended
// component ID with a fresh payload is not. The exact-match version gate in
// `unpack` is what makes the break loud instead of silent.
TEST(TaskCodec, WireVersionIsFive) { EXPECT_EQ(TaskSpec::kWireVersion, 5u); }

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

// The criterion is the LAST thing the Minimize packer writes (entries, then
// `max_iter`, then the convergence criterion), so every alternative has to
// survive as the byte-stream tail: a decoder that reads a fixed five doubles
// there, or that forgets to dispatch on the criterion id, ends up either
// short or with trailing bytes — and `require_end()` is the only thing
// standing between that and a silently wrong tolerance.
TEST(TaskCodec, MinimizeCriterionRoundtripCoversEveryAlternative) {
    // Pressure/Stress spelled with NON-DEFAULT values throughout. The target
    // is deliberately positive on both: only the TOLERANCE is required to be
    // positive (`PressureConverged` / `StressConverged` throw on ptol/stol
    // <= 0 and on nothing else), and the mask triple is asymmetric so a
    // codec writing one bool for three cannot pass.
    const std::vector<MinimizeCriterionParams> criteria{
        ForceConvergedParams{0.02, 0.01, 0.003, 0.002, 2e-6},
        PressureConvergedParams{/*P_target=*/1e-4, /*ptol=*/5e-4},
        StressConvergedParams{/*sigma_target=*/1.5e-3, /*stol=*/2e-3,
                              /*check_x=*/true, /*check_y=*/false,
                              /*check_z=*/true}};

    for (std::size_t i = 0; i < criteria.size(); ++i) {
        SCOPED_TRACE("criterion alternative " + std::to_string(i));
        std::vector<MinimizeComponentEntry> entries;
        entries.push_back(MinimizeComponentEntry{
            NeighborListParams{6.0, 0.3},
            MinimizeBinding{ATV::minimizer::Phase::force, -10}});
        entries.push_back(MinimizeComponentEntry{
            PairLJCutParams{0.0104, 3.4, 6.0},
            MinimizeBinding{ATV::minimizer::Phase::force, 0}});
        // A cell relaxer, because the pressure/stress criteria are what a
        // variable-cell run terminates on — this is the pairing the wire
        // has to carry, not an arbitrary optimizer.
        entries.push_back(MinimizeComponentEntry{
            IsotropicCellRelaxParams{0.0, 0.08, 0.08, 0.5}, std::nullopt});
        const TaskSpec spec(MinimizeTask{std::move(entries), 33, criteria[i]});

        const auto bytes = pack(spec);
        const TaskSpec back = unpack(bytes.data(), bytes.size());
        ASSERT_EQ(back.kind(), TaskKind::Minimize);
        // The alternative itself, not just the numbers: `PressureConverged`
        // and the first two `ForceConverged` tolerances are both "two
        // doubles", so an id written but ignored would roundtrip a
        // pressure test back as a force test with plausible values.
        EXPECT_EQ(back.minimize().convergence.index(), criteria[i].index());
        EXPECT_TRUE(back.minimize() == spec.minimize());
    }
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

    // Same rule for the six optimizers added with the full minimize domain:
    // CG is Minimize-only, so an MD payload naming it is refused by NAME,
    // not by a bare "unknown component". One representative is enough — the
    // six share the `read_md_params` default arm — and CG is the one whose
    // three-double neighbour on the MD side (`PairCoulCut` is one double,
    // `CoulLong` two) would otherwise decode as plausible numbers.
    md_bytes = pack(make_md_spec());
    write_scalar<std::uint16_t>(md_bytes, kFirstComponentIdOffset,
                                static_cast<std::uint16_t>(ComponentId::CG));
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

    // `convergence` is a criterion variant now; a default-constructed one is
    // still the `ForceConverged` arm, so this stays the same claim — the
    // tolerance check has to reach INTO the alternative rather than trusting
    // the struct it used to be.
    MinimizeTask bad_tol{{}, 10, {}};
    std::get<ForceConvergedParams>(bad_tol.convergence).ftol =
        std::numeric_limits<double>::quiet_NaN();
    EXPECT_THROW(TaskSpec(std::move(bad_tol)), std::invalid_argument);

    // The new criteria carry their own invariant: the TOLERANCE must be
    // positive (`PressureConverged` / `StressConverged` throw on it), and a
    // spec must not be able to smuggle past that only to blow up at
    // materialization.
    MinimizeTask bad_ptol{{}, 10, PressureConvergedParams{0.0, 0.0}};
    EXPECT_THROW(TaskSpec(std::move(bad_ptol)), std::invalid_argument);

    MinimizeTask bad_stol{
        {}, 10, StressConvergedParams{0.0, -1e-4, true, true, true}};
    EXPECT_THROW(TaskSpec(std::move(bad_stol)), std::invalid_argument);

    // `StressConverged` refuses an all-masked axis set (it would judge
    // nothing and declare victory); the wire mirror must refuse it too.
    MinimizeTask all_masked{
        {}, 10, StressConvergedParams{0.0, 1e-4, false, false, false}};
    EXPECT_THROW(TaskSpec(std::move(all_masked)), std::invalid_argument);
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
