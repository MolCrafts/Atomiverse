/// @file src/cpu/semiempirical/am1_solver.cpp
#include "atomiverse/cpu/semiempirical/am1_solver.h"

#include <stdexcept>

#include "atomiverse/core/frame_schema.h"
#include "atomiverse/core/units.h"

namespace ATV {
namespace cpu {
namespace semiempirical {

AM1::AM1() = default;

AM1::AM1(int charge_, int multiplicity_, int max_iter_, double density_tol_,
         double energy_tol_)
    : charge(charge_), multiplicity(multiplicity_), max_iter(max_iter_),
      density_tol(density_tol_), energy_tol(energy_tol_) {}

AM1::~AM1() = default;
AM1::AM1(AM1&&) noexcept = default;
AM1& AM1::operator=(AM1&&) noexcept = default;

AM1::AM1(const AM1& other)
    : charge(other.charge), multiplicity(other.multiplicity),
      max_iter(other.max_iter), density_tol(other.density_tol),
      energy_tol(other.energy_tol) {}

AM1& AM1::operator=(const AM1& other) {
    if (this != &other) {
        charge = other.charge;
        multiplicity = other.multiplicity;
        max_iter = other.max_iter;
        density_tol = other.density_tol;
        energy_tol = other.energy_tol;
        engine_.reset();
    }
    return *this;
}

void AM1::sync_engine_() {
    NDDOConfig c;
    c.method = SemiEmpiricalMethod::AM1;
    c.charge = charge;
    c.multiplicity = multiplicity;
    c.max_iter = max_iter;
    c.density_tol = density_tol;
    c.energy_tol = energy_tol;
    engine_ = std::make_unique<NDDOSolver>(c);
}

void AM1::build(const cpu::Frame& reference) {
    sync_engine_();
    engine_->build(reference);
}

AM1Result& AM1::run(const cpu::Frame& frame) {
    if (!engine_)
        build(frame);
    return engine_->run(frame);
}

AM1Result& AM1::compute_forces(cpu::Frame& frame, double step_angstrom) {
    if (!engine_)
        throw std::runtime_error("AM1::compute_forces before build/run");
    return engine_->compute_forces(frame, step_angstrom);
}

const AM1Result& AM1::last_result() const {
    if (!engine_) {
        static AM1Result empty;
        return empty;
    }
    return engine_->last_result();
}

bool AM1::built() const noexcept { return engine_ && engine_->built(); }

bool AM1::has_density() const noexcept {
    return engine_ && engine_->has_density();
}

void AM1::prepare(const cpu::BuildContext& qm_ctx) { build(qm_ctx.frame); }

void AM1::publish(cpu::Frame& frame, bool forces_requested, bool accumulate) {
    auto& result = run(frame);
    if (accumulate) {
        frame.meta.get(kEnergy) += result.total_energy_ev;
        frame.meta.get(kEnergyElectronic) += result.total_energy_ev;
    } else {
        frame.meta.get(kEnergy) = result.total_energy_ev;
        frame.meta.get(kEnergyElectronic) = result.total_energy_ev;
    }
    const int n = static_cast<int>(frame.nlocal());
    // Charges are last-writer-wins (state channel), never accumulated.
    StorageReal* __restrict sp_charge = frame.span(kAtom_charge);
    for (int i = 0; i < n; ++i) {
        sp_charge[i] = static_cast<StorageReal>(
            result.mulliken_charges[static_cast<std::size_t>(i)]);
    }
    if (forces_requested) {
        auto& fr = compute_forces(frame);
        AccReal* __restrict fx = frame.span(kAtom_fx);
        AccReal* __restrict fy = frame.span(kAtom_fy);
        AccReal* __restrict fz = frame.span(kAtom_fz);
        for (int i = 0; i < n; ++i) {
            const std::size_t base = static_cast<std::size_t>(3 * i);
            const AccReal fxi = static_cast<AccReal>(fr.forces_ev_a[base + 0]);
            const AccReal fyi = static_cast<AccReal>(fr.forces_ev_a[base + 1]);
            const AccReal fzi = static_cast<AccReal>(fr.forces_ev_a[base + 2]);
            if (accumulate) {
                fx[i] += fxi;
                fy[i] += fyi;
                fz[i] += fzi;
            } else {
                fx[i] = fxi;
                fy[i] = fyi;
                fz[i] = fzi;
            }
        }
    }
}

void AM1::evaluate(const cpu::LaunchContext& qm_ctx) {
    // Private QM frame: overwrite energy/forces.
    publish(qm_ctx.frame, qm_ctx.forces_requested, /*accumulate=*/false);
}

void AM1::reset_history() {
    // Drop persisted density by rebuilding the engine on the next run.
    engine_.reset();
}

const NDDOSystem& AM1::system() const {
    if (!engine_)
        throw std::runtime_error("AM1::system before build");
    return engine_->system();
}

const NDDOScfState& AM1::state() const {
    if (!engine_)
        throw std::runtime_error("AM1::state before build");
    return engine_->state();
}

} // namespace semiempirical
} // namespace cpu
} // namespace ATV
