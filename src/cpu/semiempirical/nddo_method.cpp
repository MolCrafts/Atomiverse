/// @file src/cpu/semiempirical/nddo_method.cpp
/// @brief NDDO Driver loop: diagonalize → density → fock → check.

#include "atomiverse/cpu/semiempirical/nddo_method.h"

#include <stdexcept>
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

NddoMethod::NddoMethod() = default;

void NddoMethod::build_on(cpu::Runtime& rt) {
    if (host_runtime_ && host_runtime_ != &rt)
        throw std::logic_error("NddoMethod: already nested on another Runtime");
    host_runtime_ = &rt;
}

void NddoMethod::destroy_on(cpu::Runtime& rt) {
    if (host_runtime_ == &rt)
        host_runtime_ = nullptr;
}

int NddoMethod::run_nested_erased(cpu::Context& ctx, cpu::Frame& frame,
                                  int /*budget*/) {
    const WrappedPositions saved = unwrap_molecule(frame);
    struct Restore {
        cpu::Frame& frame;
        const WrappedPositions& saved;
        bool& nested;
        ~Restore() {
            restore_positions(frame, saved);
            nested = false;
        }
    } restore{frame, saved, nested_active_};
    nested_active_ = true;
    // The hosting phase owns zeroing, so every write is `+=`.
    publish(frame, ctx.exec.forces_requested, /*accumulate=*/true);
    return iterations_;
}

NddoMethod::~NddoMethod() { reset_built_driver_(); }

void NddoMethod::reset_built_driver_() {
    diagonalize_.reset();
    density_.reset();
    fock_.reset();
    check_.reset();
    converged_.reset();
    this->pending_.clear();
    this->owned_components_.clear();
    this->plan_ = {};
    this->criteria_.clear();
    this->criteria_owned_.clear();
    this->backend_.reset();
}

NDDOResult& NddoMethod::drive_(NDDOSolver& solver, const cpu::Frame& frame) {
    reset_built_driver_();
    solver.scf_prepare(frame);
    loop_converged_ = false;

    diagonalize_ = std::make_unique<NddoStep>(
        ATV::nddo::Phase::diagonalize, "nddo_diagonalize",
        [&solver]() { solver.scf_diagonalize(); });
    density_ =
        std::make_unique<NddoStep>(ATV::nddo::Phase::density, "nddo_density",
                                   [&solver]() { solver.scf_density(); });
    fock_ = std::make_unique<NddoStep>(ATV::nddo::Phase::fock, "nddo_fock",
                                       [&solver]() { solver.scf_fock(); });
    check_ = std::make_unique<NddoStep>(
        ATV::nddo::Phase::check, "nddo_check", [&solver, this]() {
            solver.scf_check();
            loop_converged_ = solver.result().converged;
        });
    converged_ = std::make_unique<NddoConverged>(loop_converged_);

    this->on(ATV::nddo::Phase::diagonalize).add_ref(*diagonalize_);
    this->on(ATV::nddo::Phase::density).add_ref(*density_);
    this->on(ATV::nddo::Phase::fock).add_ref(*fock_);
    this->on(ATV::nddo::Phase::check).add_ref(*check_);
    this->add_criterion_ref(*converged_);

    this->set_serial_on_replica();
    cpu::Driver<ATV::nddo::Phase>::build();

    // The stages close over the solver. Destroy them on every exit, including
    // a throw from setup or from a step, before those unique_ptrs die.
    struct ResetGuard {
        NddoMethod& self;
        ~ResetGuard() { self.reset_built_driver_(); }
    } reset_guard{*this};

    struct PassGuard {
        NddoMethod& self;
        bool armed = false;
        ~PassGuard() {
            if (armed)
                self.teardown_pass();
        }
    } pass_guard{*this};

    // Nested: the stage plan runs on the parent Runtime for this solve.
    // Standalone: this method is the root loop and owns its backend.
    cpu::Runtime* const host = nested_active_ ? host_runtime_ : nullptr;
    cpu::Frame loop_frame;
    cpu::Context* host_ctx = nullptr;
    LoopServices parent_loop{};
    struct HostGuard {
        NddoMethod& self;
        cpu::Runtime* host;
        cpu::Context* ctx = nullptr;
        LoopServices saved{};
        ~HostGuard() {
            if (ctx)
                ctx->loop = saved;
            if (host)
                host->destroy_plan(self.plan());
        }
    } host_guard{*this, nullptr};
    if (host) {
        host->build_plan(this->plan());
        host_guard.host = host;
        host_ctx = &host->context();
        parent_loop = host_ctx->loop;
        host_guard.ctx = host_ctx;
        host_guard.saved = parent_loop;
        host_ctx->loop.status = &this->mutable_outer_status();
        host_ctx->loop.metrics = &this->metrics_;
        this->metrics_ = ConvergenceMetrics{};
    } else {
        setup_pass(loop_frame);
        pass_guard.armed = true;
    }

    const int max_iter = solver.config().max_iter;
    iterations_ = 0;
    for (int iter = 0; iter < max_iter; ++iter) {
        mutable_outer_status().reset();
        ++iterations_;
        if (host) {
            host->step_plan(this->plan(), iter);
            if (this->poll_criteria_(iter, frame, *host_ctx)
                != ATV::Verdict::Continue)
                break;
        } else {
            step_once(iter);
            if (poll_criteria_(iter) != ATV::Verdict::Continue)
                break;
        }
    }
    if (!host) {
        pass_guard.armed = false;
        teardown_pass();
    }

    if (!(this->converged() || loop_converged_))
        throw std::runtime_error("NDDOSolver AM1 SCF did not converge");
    return solver.result();
}

void NddoMethod::publish_(NDDOSolver& solver, double force_step,
                          cpu::Frame& frame, bool forces_requested,
                          bool accumulate) {
    auto& result = drive_(solver, frame);
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
        auto& fr = solver.compute_forces(frame, force_step);
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

} // namespace semiempirical
} // namespace cpu
} // namespace ATV
