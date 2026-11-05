/// @file src/cpu/semiempirical/nddo_method.cpp
/// @brief NDDO Driver loop: diagonalize → density → fock → check.

#include "atomiverse/cpu/semiempirical/nddo_method.h"

#include <algorithm>
#include <array>
#include <cstdio>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "atomiverse/core/frame_schema.h"
#include "atomiverse/cpu/semiempirical/internal/nddo_solver.h"
#include "atomiverse/cpu/semiempirical/stages/nddo_stages.h"

namespace ATV {
namespace cpu {
namespace semiempirical {

namespace {

/// @brief Wrapped positions saved around a molecular unwrap.
struct WrappedPositions {
    std::vector<StorageReal> x, y, z;
};

/// @brief Anchor-on-atom-0 minimum-image unwrap for an isolated molecule.
///
/// NDDO is molecular (no lattice sums): MD wrapping can place bonded partners
/// on opposite faces of the box, which the SCF would see as dissociated.
/// Coordinates are rewritten in place so other components' spans stay valid.
WrappedPositions unwrap_molecule(cpu::Frame& frame) {
    const int n = static_cast<int>(frame.nlocal());
    StorageReal* __restrict x = frame.span(kAtom_x);
    StorageReal* __restrict y = frame.span(kAtom_y);
    StorageReal* __restrict z = frame.span(kAtom_z);
    WrappedPositions saved{std::vector<StorageReal>(x, x + n),
                           std::vector<StorageReal>(y, y + n),
                           std::vector<StorageReal>(z, z + n)};
    if (n <= 1
        || !(frame.box.length(0) > 0.0 && frame.box.length(1) > 0.0
             && frame.box.length(2) > 0.0))
        return saved;
    const double x0 = x[0], y0 = y[0], z0 = z[0];
    for (int i = 1; i < n; ++i) {
        double dx = x[i] - x0, dy = y[i] - y0, dz = z[i] - z0;
        const double ox = dx, oy = dy, oz = dz;
        frame.box.min_image(dx, dy, dz);
        // Preserve the input exactly when no periodic image changed.
        if (dx != ox)
            x[i] = static_cast<StorageReal>(x0 + dx);
        if (dy != oy)
            y[i] = static_cast<StorageReal>(y0 + dy);
        if (dz != oz)
            z[i] = static_cast<StorageReal>(z0 + dz);
    }
    return saved;
}

/// @brief What a caller needs to judge a failed SCF: method, iterations,
///        final residuals and the thresholds they were judged against.
std::string nonconvergence_message(const NDDOSolver& solver, int iterations,
                                   double density_residual,
                                   double energy_residual) {
    const auto& config = solver.config();
    char buffer[480];
    std::snprintf(buffer, sizeof buffer,
                  "%s SCF did not converge in %d iteration(s): last "
                  "max|dP| = %.3e (density_tol %.3e), last |dE| = %.3e eV "
                  "(energy_tol %.3e eV), max|FP-PF| = %.3e eV. Both "
                  "tolerances must be met. Check the geometry, charge and "
                  "multiplicity, or raise max_iter.",
                  ATV::semiempirical::nddo_method_name(config.method),
                  iterations, density_residual, config.density_tol,
                  energy_residual, config.energy_tol,
                  solver.commutator_residual());
    return buffer;
}

void restore_positions(cpu::Frame& frame, const WrappedPositions& saved) {
    StorageReal* __restrict x = frame.span(kAtom_x);
    StorageReal* __restrict y = frame.span(kAtom_y);
    StorageReal* __restrict z = frame.span(kAtom_z);
    for (std::size_t i = 0; i < saved.x.size(); ++i) {
        x[i] = saved.x[i];
        y[i] = saved.y[i];
        z[i] = saved.z[i];
    }
}

} // namespace

struct NddoMethod::Workspace {
    NDDOSolver* active_solver = nullptr;
    int budget = -1;
    double density_residual = std::numeric_limits<double>::infinity();
    double energy_residual = std::numeric_limits<double>::infinity();
    std::array<NddoStep, 4> stages;
    NddoConverged criterion;

    explicit Workspace(NddoMethod& owner)
        : stages{{{ATV::nddo::Phase::diagonalize, "nddo_diagonalize",
                   [this] { solver().scf_diagonalize(); }},
                  {ATV::nddo::Phase::density, "nddo_density",
                   [this] { solver().scf_density(); }},
                  {ATV::nddo::Phase::fock, "nddo_fock",
                   [this] { solver().scf_fock(); }},
                  {ATV::nddo::Phase::check, "nddo_check",
                   [this] {
                       solver().scf_check();
                       density_residual = solver().density_residual();
                       energy_residual = solver().energy_residual();
                   }}}},
          criterion(density_residual, energy_residual, 1.0, 1.0) {
        for (int i = 0; i < 4; ++i)
            owner.on(static_cast<ATV::nddo::Phase>(i)).add_ref(stages[i]);
        owner.add_criterion_ref(criterion);
    }

    NDDOSolver& solver() const {
        if (!active_solver)
            throw std::logic_error("NddoMethod: stage outside an active solve");
        return *active_solver;
    }
};

NddoMethod::NddoMethod() : workspace_(std::make_unique<Workspace>(*this)) {
    cpu::Driver<ATV::nddo::Phase>::build();
}

NddoMethod::NddoMethod(NddoMethod&& other)
    : cpu::Driver<ATV::nddo::Phase>([&]() -> NddoMethod&& {
          if (other.workspace_ && other.workspace_->active_solver)
              throw std::logic_error("cannot move an active NDDO solve");
          return std::move(other);
      }()),
      iterations_(other.iterations_), workspace_(std::move(other.workspace_)) {}

void NddoMethod::build_on(cpu::Runtime& rt) {
    cpu::Driver<ATV::nddo::Phase>::build_on(rt);
    host_runtime_ = &rt;
}

void NddoMethod::destroy_on(cpu::Runtime& rt) {
    if (attached_runtime_ != &rt)
        throw std::logic_error("NddoMethod: not attached to this Runtime");
    host_runtime_ = nullptr;
    cpu::Driver<ATV::nddo::Phase>::destroy_on(rt);
}

int NddoMethod::run_nested_erased(cpu::Context& ctx, cpu::Frame& frame,
                                  int budget) {
    publish_in_context_({ctx, frame, 0, 0, ctx.exec.forces_requested,
                         ctx.exec.stress_requested},
                        /*accumulate=*/true, budget);
    return iterations_;
}

void NddoMethod::publish_in_context_(const cpu::LaunchContext& ctx,
                                     bool accumulate, int budget) {
    if (!workspace_)
        throw std::logic_error(
            "NddoMethod: cannot solve with a moved-from method");
    cpu::Runtime* const host = ctx.runtime.exec.runtime;
    if (host_runtime_ && host_runtime_ != host)
        throw std::logic_error("NddoMethod: already nested on another Runtime");
    auto& frame = ctx.frame;
    const WrappedPositions saved = unwrap_molecule(frame);
    struct Restore {
        NddoMethod& self;
        cpu::Frame& frame;
        const WrappedPositions& positions;
        cpu::Runtime* host;
        bool nested;
        bool accept_history;
        int budget;
        ~Restore() {
            restore_positions(frame, positions);
            self.host_runtime_ = host;
            self.nested_active_ = nested;
            self.accept_history_ = accept_history;
            self.workspace_->budget = budget;
        }
    } restore{*this,
              frame,
              saved,
              host_runtime_,
              nested_active_,
              accept_history_,
              workspace_->budget};
    workspace_->budget = budget;
    host_runtime_ = host;
    nested_active_ = host != nullptr;
    // Line-search trials stage like committed solves (the step commits the
    // last one staged); MC trials and force preparation accept nothing.
    accept_history_ = !(ctx.runtime.exec.initializing_forces
                        || (ctx.runtime.exec.trial_evaluation
                            && !ctx.runtime.opt.line_search_trial));
    publish(frame, ctx.forces_requested && ctx.runtime.exec.forces_requested,
            accumulate);
}

void NddoMethod::invalidate_inputs(InputChange changes, cpu::Runtime& runtime) {
    reset_history();
    cpu::Driver<ATV::nddo::Phase>::invalidate_inputs(changes, runtime);
}

NddoMethod::~NddoMethod() {
    // The plan contains references into workspace_: release every lifecycle
    // while those objects are still alive, as in the persistent DFTB2 driver.
    if (attached_runtime_)
        destroy_on(*attached_runtime_);
    else
        this->shutdown();
}

ATV::semiempirical::NDDOResult& NddoMethod::drive_(NDDOSolver& solver,
                                                   const cpu::Frame& frame) {
    if (!workspace_)
        throw std::logic_error(
            "NddoMethod: cannot solve with a moved-from method");
    auto& work = *workspace_;
    if (work.active_solver)
        throw std::logic_error("NddoMethod: a solve is already active");
    if (attached_runtime_ && !nested_active_)
        throw std::logic_error(
            "NddoMethod: a mounted method runs through its parent");
    // Preparation and zero-budget failures happen before run_nested resets
    // its loop. No new solve may retain a prior successful Driver verdict.
    this->mutable_outer_status().reset();
    this->metrics_ = ConvergenceMetrics{};
    solver.result().converged = false;
    work.active_solver = &solver;
    struct ReleaseSolver {
        Workspace& work;
        ~ReleaseSolver() { work.active_solver = nullptr; }
    } release_solver{work};
    iterations_ = 0;
    work.density_residual = std::numeric_limits<double>::infinity();
    work.energy_residual = std::numeric_limits<double>::infinity();
    work.criterion.set_tolerances(solver.config().density_tol,
                                  solver.config().energy_tol);
    const int max_iter = work.budget < 0
                             ? solver.config().max_iter
                             : std::min(solver.config().max_iter, work.budget);
    if (max_iter == 0) {
        solver.result().iterations = 0;
        solver.result().converged = false;
        solver.result().forces_ev_a.clear();
        throw std::runtime_error("NDDOSolver: zero SCF iteration budget");
    }
    solver.scf_prepare(frame);

    // A mounted method keeps its plan on the parent's Runtime. QmEngine borrows
    // a Runtime for one evaluation; standalone entry owns only this pass.
    cpu::Runtime* host = nested_active_ ? host_runtime_ : nullptr;
    cpu::Frame loop_frame;
    bool standalone = false, temporary_plan = false;
    auto release = [&] {
        if (temporary_plan) {
            temporary_plan = false;
            host->destroy_plan(this->plan());
        }
        if (standalone) {
            standalone = false;
            this->teardown_pass();
        }
    };
    try {
        if (!host) {
            if (!this->backend_)
                this->set_serial_on_replica();
            setup_pass(loop_frame);
            standalone = true;
            host = this->backend_.get();
        } else if (host != attached_runtime_) {
            host->build_plan(this->plan());
            temporary_plan = true;
        }
        iterations_ = cpu::Driver<ATV::nddo::Phase>::run_nested(
            host->context(), host->frame(), max_iter);
        release();
        solver.result().converged = this->converged();
        if (!solver.result().converged)
            throw std::runtime_error(nonconvergence_message(
                solver, iterations_, work.density_residual,
                work.energy_residual));
        return solver.result();
    } catch (...) {
        this->mutable_outer_status().reset();
        solver.result().converged = false;
        // Preserve the original failure if teardown also reports a failure.
        // Runtime destroys its complete component ledger before throwing.
        try {
            release();
        } catch (...) {
        }
        throw;
    }
}

void NddoMethod::publish_(NDDOSolver& solver, cpu::Frame& frame,
                          bool forces_requested, bool accumulate) {
    auto& result = drive_(solver, frame);
    if (forces_requested)
        solver.compute_forces(frame);
    if (accumulate) {
        frame.meta.get(kEnergy) += result.total_energy_ev;
        frame.meta.get(kEnergyElectronic) += result.total_energy_ev;
    } else {
        frame.meta.get(kEnergy) = result.total_energy_ev;
        frame.meta.get(kEnergyElectronic) = result.total_energy_ev;
    }
    const int n = static_cast<int>(frame.nlocal());
    StorageReal* __restrict sp_charge = frame.span(kAtom_charge);
    for (int i = 0; i < n; ++i) {
        sp_charge[i] = static_cast<StorageReal>(
            result.mulliken_charges[static_cast<std::size_t>(i)]);
    }
    if (forces_requested) {
        const auto& fr = result;
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
    // Accept only after the complete requested evaluation has succeeded.
    // Forces use the current SCF state, independently of the accepted guess.
    if (accept_history_) {
        auto* transaction =
            host_runtime_ ? host_runtime_->context().exec.evaluation_transaction
                          : nullptr;
        if (transaction && transaction->active()) {
            transaction->enlist(*this);
            pending_solver_ = &solver;
            solver.stage_density();
        } else {
            solver.commit_density();
        }
    }
}

void NddoMethod::commit_evaluation() noexcept {
    if (pending_solver_)
        pending_solver_->commit_staged_density();
    pending_solver_ = nullptr;
}

void NddoMethod::discard_evaluation() noexcept {
    if (pending_solver_)
        pending_solver_->discard_staged_density();
    pending_solver_ = nullptr;
}

} // namespace semiempirical
} // namespace cpu
} // namespace ATV
