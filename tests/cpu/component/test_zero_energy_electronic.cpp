/// @file tests/cpu/component/test_zero_energy_electronic.cpp
/// @brief Unit: the `kEnergyElectronic` per-term channel — `ZeroEnergy` clears
///        it (U8), and a co-bound `Evaluate` pass leaves the electronic share
///        alone in it (U7b).
///
/// Mirrors `src/cpu/component/zero.cpp` (`ATV::cpu::ZeroEnergy`) and, for the
/// second case, the per-term write the five ES evaluators gain.
/// Spec: `.claude/specs/esmm-contract-05-accumulate-switch` (U7b, U8).
///
/// ## Why the key exists
///
/// `ZeroEnergy` clears `kEnergy` plus every per-term key in lockstep so the
/// per-step invariant `kEnergy == Σ per-term (+ unclassified)` holds
/// (`zero.cpp:44-53`). The decomposition today covers only the classical terms —
/// pair / bond / angle / dihedral / improper / charging-work. Once an
/// electronic-structure source is allowed to *accumulate* into `kEnergy` beside
/// a classical one (U7), the total is no longer attributable: nothing records
/// how much of it was the QM part. `kEnergyElectronic` is that term, deliberately
/// **one** key shared by all ES sources — co-bound ES contributions legitimately
/// merge into a single "electronic share of kEnergy" (Δ-learning, QM/MM's QM
/// region); per-kind keys would be key explosion with no consumer.
///
/// ## Expected RED state (compile, not assert)
///
/// This translation unit does **not compile** until
/// `inline constexpr MetaKey<double> kEnergyElectronic{"energy_electronic",
/// true};` lands in `include/atomiverse/core/frame_schema.h` (after
/// `kEnergyChargingWork`) and `cpu::Frame(int)` pre-inserts it. That is the
/// intended RED for a new API surface: the signature is pinned by a caller
/// before it exists. It is kept OUT of
/// `tests/cpu/evaluate/test_source_cobinding.cpp` precisely so that file's
/// assert-level RED still builds and runs today.
///
/// ## Determinism
///
/// `set_threads(1)`; the SCF value is a hard-coded literature pin (see the
/// provenance on `kRhfH2Energy`), never a live oracle. No RNG, no wall clock,
/// no filesystem, no network.

#include <string>

#include <gtest/gtest.h>

#include "atomiverse/core/frame_schema.h"
#include "atomiverse/cpu/component/component.h"
#include "atomiverse/cpu/component/context.h"
#include "atomiverse/cpu/component/zero.h"
#include "atomiverse/cpu/frame/box.h"
#include "atomiverse/cpu/frame/frame.h"

#ifdef ATV_HAS_HF
#include "atomiverse/core/stage.h"
#include "atomiverse/cpu/component/potential.h"
#include "atomiverse/cpu/evaluate/evaluate.h"
#include "atomiverse/cpu/scf/basis_set.h"
#include "atomiverse/cpu/scf/methods.h"
#endif

using namespace ATV;

namespace {

/// @brief Two-atom Frame in a 20 Å cubic box (also the U7b H₂ geometry).
/// @param bond H–H separation [Å].
cpu::Frame make_h2(double bond) {
    cpu::Frame f(2);
    f.span(kAtom_x)[0] = -0.5 * bond;
    f.span(kAtom_x)[1] = 0.5 * bond;
    f.span(kAtom_y)[0] = 0.0;
    f.span(kAtom_y)[1] = 0.0;
    f.span(kAtom_z)[0] = 0.0;
    f.span(kAtom_z)[1] = 0.0;
    f.span(kAtom_type)[0] = 1; // Z = 1
    f.span(kAtom_type)[1] = 1;
    f.span(kAtom_mass)[0] = 1.008;
    f.span(kAtom_mass)[1] = 1.008;
    f.span(kAtom_tag)[0] = 1;
    f.span(kAtom_tag)[1] = 2;
    f.box = cpu::Box::cubic(20.0);
    return f;
}

} // namespace

// ── U8: ZeroEnergy clears the new per-term key ──────────────────────────────

/// The whole point of a per-term key is the lockstep invariant: a term that
/// `ZeroEnergy` forgets accumulates across steps while `kEnergy` restarts, and
/// the decomposition drifts apart silently within one MD run. Mirrors
/// `ChargingWork.ZeroEnergyClearsChargingWorkMeta` (the last key added).
///
/// Exact equality, not a tolerance: zeroing writes the literal 0.
TEST(ZeroEnergyElectronic, ZeroEnergyClearsElectronicTerm) {
    cpu::Frame f = make_h2(0.7122);
    cpu::Context ctx;

    f.meta.get(kEnergy) = 7.0;
    f.meta.get(kEnergyElectronic) = 3.0;

    cpu::ZeroEnergy zero;
    zero.launch({ctx, f, /*stage=*/0, /*step=*/0});

    EXPECT_EQ(static_cast<double>(f.meta.get(kEnergy)), 0.0);
    EXPECT_EQ(static_cast<double>(f.meta.get(kEnergyElectronic)), 0.0)
        << "ZeroEnergy left kEnergyElectronic at "
        << static_cast<double>(f.meta.get(kEnergyElectronic))
        << " — a per-term key that is not cleared in lockstep with kEnergy "
           "accumulates across steps and breaks `kEnergy == sum of terms`";
}

/// Edge: zeroing an already-zero frame is a no-op, and zeroing twice does not
/// resurrect a value. Cheap, but it pins that the clear is unconditional rather
/// than "clear if present/non-zero".
TEST(ZeroEnergyElectronic, ZeroEnergyIsIdempotentOnTheElectronicTerm) {
    cpu::Frame f = make_h2(0.7122);
    cpu::Context ctx;

    cpu::ZeroEnergy zero;
    zero.launch({ctx, f, /*stage=*/0, /*step=*/0});
    EXPECT_EQ(static_cast<double>(f.meta.get(kEnergyElectronic)), 0.0);

    f.meta.get(kEnergyElectronic) = -12.5; // sign matters: ES terms are negative
    zero.launch({ctx, f, /*stage=*/0, /*step=*/0});
    EXPECT_EQ(static_cast<double>(f.meta.get(kEnergyElectronic)), 0.0);
}

// ── U7b: the electronic share is attributable after co-binding ──────────────

#ifdef ATV_HAS_HF

namespace {

/// @brief Energy the classical stub contributes, per launch [eV].
///        Same constant as `tests/cpu/evaluate/test_source_cobinding.cpp`.
constexpr double kStubEnergy = 1.5;

/// @brief H–H separation [Å] — CCCBDB HF/STO-3G r_e at table precision.
constexpr double kH2Bond = 0.7122;

/// @brief RHF/STO-3G total energy of H₂ at `kH2Bond` [eV].
///
/// NIST CCCBDB (NIST Standard Reference Database 101,
/// <https://cccbdb.nist.gov>), *calculated* HF/STO-3G data for H₂ (¹Σg⁺):
/// E_total = −1.11751 Hartree; geometry originally W. J. Hehre, R. F. Stewart,
/// J. A. Pople, *J. Chem. Phys.* **51**, 2657 (1969), DOI 10.1063/1.1672392.
/// In-tree regressions pin −1.117506 Ha = −30.4088842472 eV.
/// **Hard-coded golden** — nothing here reads PySCF or any other package.
constexpr double kRhfH2Energy = -30.408884;

/// @brief Assertion tolerance on energies [eV] — the CCCBDB quote resolution
///        (5 decimals in Hartree ≈ 1.4 × 10⁻⁴ eV). See the sibling U7 file.
constexpr double kEnergyTol = 1e-4;

/// @brief Classical `+=` energy source, the co-author of the QM term.
class AccumulatingStub final : public cpu::Potential {
  public:
    /// @param e Energy added to `kEnergy` on every launch [eV].
    explicit AccumulatingStub(double e) : e_(e) {}
    std::string name() const override { return "accumulating_stub"; }
    void launch(const cpu::LaunchContext& ctx) override {
        ++launches;
        ctx.frame.meta.get(kEnergy) += e_;
    }
    int launches = 0; ///< Launch counter — guards against a dead binding.
  private:
    double e_; ///< Contribution per launch [eV].
};

} // namespace

/// U7b — after a co-bound pass, `kEnergy` carries the sum but
/// `kEnergyElectronic` carries the **QM share alone**. Without this the flip to
/// `+=` buys composability at the cost of attribution: a QM/MM user could no
/// longer say how much of the total was the QM region.
///
/// The classical stub deliberately does NOT touch `kEnergyElectronic` — that is
/// what makes the electronic term a measurement and not a copy of `kEnergy`.
TEST(ZeroEnergyElectronic, CoBoundPassLeavesOnlyTheQmShareInElectronicTerm) {
    cpu::Frame frame = make_h2(kH2Bond);

    AccumulatingStub stub{kStubEnergy};
    cpu::scf::RHF rhf;
    rhf.basis = cpu::scf::BasisSet::STO3G;
    rhf.tol = 1e-10;
    rhf.with_gradient(false);

    cpu::Evaluate eval;
    eval.on(evaluate::Phase::evaluate).add_ref(stub, /*order=*/0);
    eval.on(evaluate::Phase::evaluate).add_ref(rhf, /*order=*/10);
    eval.set_threads(1);
    eval.build();
    eval.run(frame);

    ASSERT_EQ(stub.launches, 1);
    ASSERT_TRUE(rhf.converged());

    // The total (the U7 claim, restated here so a partial fix cannot pass this
    // file while failing the contract).
    EXPECT_NEAR(static_cast<double>(frame.meta.get(kEnergy)),
                kStubEnergy + kRhfH2Energy, kEnergyTol)
        << "kEnergy did not carry both co-bound sources";

    // The attribution (the U7b claim).
    EXPECT_NEAR(static_cast<double>(frame.meta.get(kEnergyElectronic)),
                kRhfH2Energy, kEnergyTol)
        << "kEnergyElectronic = "
        << static_cast<double>(frame.meta.get(kEnergyElectronic))
        << " eV, expected the QM share alone (" << kRhfH2Energy
        << " eV). A value near " << (kStubEnergy + kRhfH2Energy)
        << " means the ES source copied the running total instead of writing "
           "its own term; 0 means it never wrote the term at all";
}

/// Sole-source control: with RHF the only author, `kEnergyElectronic` equals
/// `kEnergy`. Pins that the term is written unconditionally by the ES source
/// (not only when a co-author is present) — the case a single-point QM user hits.
TEST(ZeroEnergyElectronic, SoleSourceRhfWritesTheWholeEnergyAsElectronic) {
    cpu::Frame frame = make_h2(kH2Bond);

    cpu::scf::RHF rhf;
    rhf.basis = cpu::scf::BasisSet::STO3G;
    rhf.tol = 1e-10;
    rhf.with_gradient(false);

    cpu::Evaluate eval;
    eval.on(evaluate::Phase::evaluate).add_ref(rhf, /*order=*/0);
    eval.set_threads(1);
    eval.build();
    eval.run(frame);

    ASSERT_TRUE(rhf.converged());
    EXPECT_NEAR(static_cast<double>(frame.meta.get(kEnergy)), kRhfH2Energy,
                kEnergyTol);
    EXPECT_NEAR(static_cast<double>(frame.meta.get(kEnergyElectronic)),
                static_cast<double>(frame.meta.get(kEnergy)), kEnergyTol)
        << "sole-source RHF: the electronic term must account for the whole "
           "energy";
}

#endif // ATV_HAS_HF
