/// @file cpu/comm/reduce.h
/// @brief End-of-step MPI reductions for MD: energy, kinetic, stress.
///
/// `Reduce` does one batched `MPI_Allreduce(SUM)` of the 18 per-rank
/// scalar partials `{energy, energy_pair, energy_bond, energy_angle,
/// energy_dihedral, energy_improper, kinetic, stress[6], stress_kinetic[6]}`
/// into a single
/// collective, so per-rank bookkeeping is globally consistent before
/// observers or the snapshot publisher read the frame's scalar state.
/// `n_local` is NOT reduced — it is authoritative per rank from
/// `dd->n_local`. Single-rank: no-op early-return.
///
/// Order is tight: must run after `KineticEnergy @ -100` so the reduced KE
/// is the locally-summed one, and before `Publish @ +10` so async
/// observers see globally-reduced scalars.
///
/// @see .claude/specs/parallel-md-engine.md  § A.4
#pragma once

#include <string>
#include <vector>

#include "atomiverse/core/binding.h"
#include "atomiverse/cpu/component/component.h"
#include "atomiverse/facade/stage.h"

namespace ATV {
namespace cpu {
namespace md {
namespace mpi {

/// @brief Batched 18-double `MPI_Allreduce(SUM)` of the per-rank energy,
/// kinetic, and stress partials. Single-rank: no-op (state already global).
///
/// `every` sets the reduction cadence: with `every == 1` (default) the
/// collective fires every step — the only correct setting whenever a per-step
/// consumer reads the global scalars (a NoseHoover thermostat integrates the
/// global temperature, a barostat the global pressure, every step). With
/// `every > 1` the reduce fires only on step 0 and every `every`-th step,
/// gated on the all-ranks-identical `ctx.step` so the collective stays in
/// lockstep (no rank-divergent skip → no deadlock). That is safe ONLY for runs
/// whose global scalars are pure diagnostics sampled at a known cadence — plain
/// NVE / per-atom Langevin (the throughput stage) — where it elides the
/// per-step latency-bound collective LAMMPS likewise avoids until thermo output.
class Reduce : public cpu::Component {
  public:
    Reduce() = default;
    explicit Reduce(int every) : every_(every < 1 ? 1 : every) {}
    std::string name() const override { return "reduce"; }
    std::vector<DefaultBinding> default_bindings() const override {
        return {ATV::make_binding(ATV::md::Phase::end_of_step, -50)};
    }
    void launch(const cpu::LaunchContext& ctx) override;

  private:
    int every_ = 1;
};

} // namespace mpi
} // namespace md
} // namespace cpu
} // namespace ATV
