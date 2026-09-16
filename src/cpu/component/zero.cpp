/// @file cpu/component/zero.cpp
/// @brief Phase-agnostic accumulator-zeroing component implementations.

#include "atomiverse/cpu/component/zero.h"

#include "atomiverse/core/frame_schema.h"
#include "atomiverse/cpu/component/owned_range.h"

namespace ATV {
namespace cpu {

// ── ZeroForces ───────────────────────────────────────────────────

void ZeroForces::launch(const cpu::LaunchContext& ctx) {
    auto& f = ctx.frame;
    // Zero only the OWNED rows [0, n_local). In multi-rank the backing store
    // is sized to the loaded (global) atom count, so zeroing [0, atoms.size())
    // would touch ~global-N slots on every rank every step — measured as the
    // single largest scaling cost at high rank counts (it dwarfed the LJ
    // kernel). Ghost rows [n_local, n_local + n_ghost) are zeroed by
    // `CommBrick` right after it (re)populates them this step,
    // giving `CommBrick` its known-zero ghost baseline.
    // Single-rank: `owned_count` == atoms.size() and n_ghost == 0, so this is
    // the full (and only) array. See owned_range.h for the row-range contract.
    const int n = owned_count(ctx);
    // Hoisted once before the OMP region -- never span() inside a parallel
    // region (data race + the runtime resolution-budget assertion).
    AccReal* __restrict sp_fx = f.span(kAtom_fx);
    AccReal* __restrict sp_fy = f.span(kAtom_fy);
    AccReal* __restrict sp_fz = f.span(kAtom_fz);
#pragma omp parallel for schedule(static)
    for (int i = 0; i < n; ++i) {
        sp_fx[i] = 0;
        sp_fy[i] = 0;
        sp_fz[i] = 0;
    }
}

// ── ZeroEnergy ───────────────────────────────────────────────────

void ZeroEnergy::launch(const cpu::LaunchContext& ctx) {
    auto& meta = ctx.frame.meta;
    meta.get(kEnergy) = 0;
    // Per-term decomposition (msu-pe-cuda-components-01 / Phase G.3;
    // `energy_electronic` added by esmm-contract-05). Each evaluator
    // accumulates into both `energy` and its term-specific field, so
    // ZeroEnergy must clear all of them in lockstep.
    //
    // The honest invariant is
    //     energy == Σ per-term + unclassified,
    // NOT `energy == Σ per-term`: a `Bias` component accumulates into
    // `energy` and carries no per-term key of its own, so its contribution
    // lands only in the total. That gap is pre-existing, deliberately NAMED
    // debt (esmm-contract-05, "Out of scope"); a bias term key is a separate
    // change. Every source that DOES own a term key must keep both writes in
    // step, or the decomposition drifts from the total within one run.
    meta.get(kEnergyPair) = 0;
    meta.get(kEnergyBond) = 0;
    meta.get(kEnergyAngle) = 0;
    meta.get(kEnergyDihedral) = 0;
    meta.get(kEnergyImproper) = 0;
    meta.get(kEnergyChargingWork) = 0;
    meta.get(kEnergyElectronic) = 0;
}

// ── ZeroStress ───────────────────────────────────────────────────

void ZeroStress::launch(const cpu::LaunchContext& ctx) {
    auto& f = ctx.frame;
    for (int k = 0; k < 6; ++k)
        f.meta.get(kStress)[k] = 0;
    // `stress_kinetic` is overwritten (not accumulated) by Virial
    // each step, so it does not strictly need zeroing -- but zero it
    // here for symmetry and so observers that read it before Virial
    // ever ran see a deterministic zero rather than uninitialised data.
    for (int k = 0; k < 6; ++k)
        f.meta.get(kStressKinetic)[k] = 0;
}

// ── ZeroKinetic ──────────────────────────────────────────────────

void ZeroKinetic::launch(const cpu::LaunchContext& ctx) {
    ctx.frame.meta.get(kKinetic) = 0;
}

} // namespace cpu
} // namespace ATV
