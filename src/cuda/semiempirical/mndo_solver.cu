/// @file cuda/semiempirical/mndo_solver.cu
/// @brief MNDO concrete Driver: this is the sole owner of the SCF loop.
#include "atomiverse/core/frame_schema.h"
#include "atomiverse/cuda/device/error.cuh"
#include "atomiverse/cuda/frame/transfer.cuh"
#include "atomiverse/cuda/semiempirical/internal/nddo_workspace.cuh"
#include "atomiverse/cuda/semiempirical/mndo_solver.cuh"
#include <algorithm>
#include <cmath>
#include <exception>
#include <stdexcept>
#include <type_traits>
#include <utility>

namespace ATV::cuda::semiempirical {
MNDO::MNDO(int q, int mult, int cap, double dtol, double etol)
    : charge(q), multiplicity(mult), max_iter(cap), density_tol(dtol),
      energy_tol(etol), workspace_(std::make_unique<internal::NddoWorkspace>(
                            ATV::semiempirical::SemiEmpiricalMethod::MNDO)) {
    if (q != 0 || mult != 1)
        throw std::invalid_argument(
            "CUDA MNDO supports neutral closed-shell molecules only");
    start_();
    for (int i = 0; i < 4; ++i)
        this->on(static_cast<ATV::nddo::Phase>(i))
            .add_ref(workspace_->stages[i]);
    this->add_criterion_ref(workspace_->criterion);
    cuda::Driver<ATV::nddo::Phase>::build();
}
MNDO::MNDO(const MNDO& o)
    : MNDO(o.charge, o.multiplicity, o.max_iter, o.density_tol, o.energy_tol) {}
MNDO::MNDO(MNDO&& o)
    : cuda::Driver<ATV::nddo::Phase>([&]() -> MNDO&& {
          if (o.active_ || o.attached_)
              throw std::logic_error("Cannot move an attached CUDA MNDO");
          // Detach the idle host boundary before core moves its borrowed plan.
          // Scientific workspace and accepted density remain method-owned.
          o.release_backend_();
          return std::move(o);
      }()),
      charge(o.charge), multiplicity(o.multiplicity), max_iter(o.max_iter),
      density_tol(o.density_tol), energy_tol(o.energy_tol),
      workspace_(std::move(o.workspace_)), last_stream_(o.last_stream_),
      standalone_atoms_(o.standalone_atoms_) {
    if (this->backend_)
        this->backend_->load(this->plan());
    o.last_stream_ = nullptr;
    o.standalone_atoms_ = 0;
}
MNDO::~MNDO() {
    try {
        release_backend_();
    } catch (...) {
        // release_backend_ attempts all cleanup before reporting an error.
        // Destruction cannot propagate it; explicit shutdown remains observable.
    }
}
void MNDO::start_() {
    if (!workspace_)
        throw std::logic_error("Cannot run a moved-from CUDA MNDO");
    if (active_)
        throw std::logic_error("CUDA MNDO solve is already active");
    this->status_.reset();
    this->metrics_ = ConvergenceMetrics{};
    workspace_->result.converged = false;
    workspace_->result.iterations = 0;
    workspace_->result.forces_ev_a.clear();
    if (max_iter <= 0)
        throw std::invalid_argument(
            "CUDA MNDO requires a positive SCF iteration budget");
    if (!std::isfinite(density_tol) || density_tol <= 0
        || !std::isfinite(energy_tol) || energy_tol <= 0)
        throw std::invalid_argument(
            "CUDA MNDO requires positive finite SCF controls");
}
void MNDO::release_backend_() {
    std::exception_ptr failure;
    auto cleanup = [&](auto&& operation) {
        try {
            operation();
        } catch (...) {
            if (!failure)
                failure = std::current_exception();
        }
    };
    if (this->backend_) {
        cleanup([&] { this->backend_->sync(); });
        cleanup([&] {
            this->destroy_borrowed_(this->backend_->context(),
                                    this->backend_->frame());
        });
        cleanup([&] { this->backend_->shutdown(); });
        this->backend_.reset();
    }
    standalone_atoms_ = 0;
    last_stream_ = nullptr;
    if (failure)
        std::rethrow_exception(failure);
}
void MNDO::shutdown() {
    this->require_mutable_();
    if (active_ || attached_)
        throw std::logic_error("Cannot shut down an attached CUDA MNDO");
    std::exception_ptr failure;
    try {
        release_backend_();
    } catch (...) {
        failure = std::current_exception();
    }
    if (workspace_)
        workspace_->release();
    this->status_.reset();
    try {
        cuda::Driver<ATV::nddo::Phase>::shutdown();
    } catch (...) {
        if (!failure)
            failure = std::current_exception();
    }
    if (failure)
        std::rethrow_exception(failure);
}
void MNDO::standalone_(const cpu::Frame& frame) {
    if (attached_)
        throw std::logic_error("A mounted CUDA MNDO runs through its parent");
    if (cuda::kPrecisionMode != cuda::PrecisionMode::Double
        || !std::is_same_v<StorageReal, double>)
        throw std::invalid_argument(
            "CUDA MNDO requires FP64 host and device Frame storage");
    if (frame.nlocal() <= 0 || frame.nghost() != 0)
        throw std::invalid_argument(
            "CUDA MNDO requires a nonempty complete molecule");
    const int count = static_cast<int>(frame.nlocal());
    if (!this->backend_ || standalone_atoms_ != count) {
        release_backend_();
        this->backend_ = std::make_unique<cuda::Backend>();
        try {
            this->backend_->init();
            this->backend_->load(this->plan());
            this->backend_->materialize(frame);
            standalone_atoms_ = count;
        } catch (...) {
            try {
                release_backend_();
            } catch (...) {
                // Keep the original initialization/materialization failure.
            }
            throw;
        }
    } else {
        const auto stream =
            static_cast<cudaStream_t>(this->backend_->context().stream);
        auto& device = this->backend_->frame();
        const std::size_t bytes =
            static_cast<std::size_t>(count) * sizeof(double);
        ATV_CUDA_CHECK(cudaMemcpyAsync(device.atoms.x,
                                       frame.span(kAtom_x).data(), bytes,
                                       cudaMemcpyHostToDevice, stream));
        ATV_CUDA_CHECK(cudaMemcpyAsync(device.atoms.y,
                                       frame.span(kAtom_y).data(), bytes,
                                       cudaMemcpyHostToDevice, stream));
        ATV_CUDA_CHECK(cudaMemcpyAsync(device.atoms.z,
                                       frame.span(kAtom_z).data(), bytes,
                                       cudaMemcpyHostToDevice, stream));
        ATV_CUDA_CHECK(cudaMemcpyAsync(
            device.atoms.type, frame.span(kAtom_type).data(),
            count * sizeof(int), cudaMemcpyHostToDevice, stream));
        double cell[9];
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j)
                cell[i * 3 + j] = frame.box.cell()(i, j);
        ATV_CUDA_CHECK(cudaMemcpyAsync(device.box, cell, sizeof(cell),
                                       cudaMemcpyHostToDevice, stream));
        // The local cell buffer must outlive the asynchronous upload.
        ATV_CUDA_CHECK(cudaStreamSynchronize(stream));
    }
    auto& ctx = this->backend_->context();
    ctx.exec.forces_requested = false;
    ctx.exec.stress_requested = false;
    workspace_->ensure(this->backend_->frame(),
                       static_cast<cudaStream_t>(ctx.stream));
    this->build_borrowed_(ctx, this->backend_->frame());
    last_stream_ = ctx.stream;
}
void MNDO::build(const cpu::Frame& frame) {
    start_();
    standalone_(frame);
}

int MNDO::solve_(cuda::Context& ctx, cuda::Frame& frame, int budget,
                 bool accumulate) {
    const int cap = budget < 0 ? max_iter : std::min(max_iter, budget);
    if (ctx.exec.stress_requested)
        throw std::invalid_argument("CUDA MNDO does not provide system stress");
    if (cap == 0)
        throw std::runtime_error("CUDA MNDO zero SCF iteration budget");
    struct Restore {
        MNDO& method;
        cuda::Context& ctx;
        cuda::LoopServices loop;
        const bool* lock;
        ~Restore() {
            ctx.loop = loop;
            method.active_ = false;
            method.launching_ = false;
            method.mutation_lock_ = lock;
        }
    } restore{*this, ctx, ctx.loop, this->mutation_lock_};
    active_ = true;
    this->launching_ = true;
    this->mutation_lock_ = &this->launching_;
    ctx.loop.status = &this->status_;
    ctx.loop.metrics = &this->metrics_;
    const auto stream = static_cast<cudaStream_t>(ctx.stream);
    last_stream_ = ctx.stream;
    workspace_->criterion.density_tol = density_tol;
    workspace_->criterion.energy_tol = energy_tol;
    // Device-side identity changes trigger a rebuild at the solve boundary.
    // Standalone inputs were already checked before the borrowed plan build.
    if (accumulate)
        workspace_->ensure(frame, stream, ctx.exec.trial_evaluation);
    workspace_->prepare(frame, stream);
    for (int i = 0; i < cap; ++i) {
        this->step_borrowed_(ctx, frame, i);
        const auto verdict = this->poll_criteria_(i, frame, ctx);
        if (verdict != Verdict::Continue)
            break;
    }
    if (!this->converged())
        throw std::runtime_error(
            "CUDA MNDO SCF failed or exhausted its iteration budget");
    if (ctx.exec.forces_requested)
        workspace_->compute_forces(frame, stream);
    workspace_->publish(frame, stream, accumulate);
    ATV_CUDA_CHECK(cudaStreamSynchronize(stream));
    if (ctx.exec.forces_requested)
        cuda::throw_if_force_fault(frame);
    workspace_->result.converged = true;
    return workspace_->result.iterations;
}
MNDOResult& MNDO::run_host_(const cpu::Frame& frame, bool force_request) {
    start_();
    standalone_(frame);
    auto& ctx = this->backend_->context();
    ctx.exec.forces_requested = force_request;
    EvaluationTransaction::Scope evaluation(*ctx.exec.evaluation_transaction);
    try {
        if (force_request)
            ATV_CUDA_CHECK(
                cudaMemsetAsync(this->backend_->frame().force_fault, 0,
                                sizeof(cuda::ForceFault),
                                static_cast<cudaStream_t>(ctx.stream)));
        solve_(ctx, this->backend_->frame(), -1, false);
        workspace_->download(static_cast<cudaStream_t>(ctx.stream));
        ctx.exec.evaluation_transaction->enlist(*this);
        workspace_->stage();
        evaluation.accept();
    } catch (...) {
        workspace_->result.converged = false;
        workspace_->result.forces_ev_a.clear();
        workspace_->forces_valid = false;
        this->status_.converged = false;
        // Drain pending work without replacing the original solve failure.
        cudaStreamSynchronize(static_cast<cudaStream_t>(ctx.stream));
        throw;
    }
    return workspace_->result;
}
MNDOResult& MNDO::run(const cpu::Frame& frame) {
    return run_host_(frame, false);
}
MNDOResult& MNDO::run(cpu::Frame& frame) {
    auto& result = run(static_cast<const cpu::Frame&>(frame));
    frame.meta.get(kEnergy) = result.total_energy_ev;
    frame.meta.get(kEnergyElectronic) = result.total_energy_ev;
    auto q = frame.span(kAtom_charge);
    for (int i = 0; i < frame.nlocal(); ++i)
        q[i] = result.mulliken_charges[i];
    return result;
}
MNDOResult& MNDO::forces(cpu::Frame& frame) {
    auto& result = run_host_(frame, true);
    frame.meta.get(kEnergy) = result.total_energy_ev;
    frame.meta.get(kEnergyElectronic) = result.total_energy_ev;
    auto q = frame.span(kAtom_charge);
    auto fx = frame.span(kAtom_fx), fy = frame.span(kAtom_fy);
    auto fz = frame.span(kAtom_fz);
    for (int i = 0; i < frame.nlocal(); ++i) {
        q[i] = result.mulliken_charges[i];
        fx[i] = result.forces_ev_a[3 * i];
        fy[i] = result.forces_ev_a[3 * i + 1];
        fz[i] = result.forces_ev_a[3 * i + 2];
    }
    return result;
}
const MNDOResult& MNDO::last_result() const {
    if (!workspace_)
        throw std::logic_error("Moved-from CUDA MNDO");
    workspace_->download(static_cast<cudaStream_t>(last_stream_));
    return workspace_->result;
}
void MNDO::commit_evaluation() noexcept {
    if (workspace_)
        workspace_->commit();
}
void MNDO::discard_evaluation() noexcept {
    if (workspace_)
        workspace_->discard();
}
bool MNDO::has_density() const noexcept {
    return workspace_ && workspace_->accepted;
}
bool MNDO::built() const noexcept { return workspace_ && workspace_->matrices; }
bool MNDO::supports_input_changes(InputChange changes) const {
    for (auto* component : this->plan_.components())
        if (!component->supports_input_changes(changes))
            return false;
    return true;
}
void MNDO::invalidate_inputs(InputChange changes, cuda::Context& ctx,
                             cuda::Frame& frame) {
    workspace_->accepted = false;
    workspace_->discard();
    for (auto* component : this->plan_.components())
        component->invalidate_inputs(changes, {ctx, frame});
}

void MNDO::build_on(cuda::Context& ctx, cuda::Frame& frame) {
    if (!workspace_)
        throw std::logic_error("Moved-from CUDA MNDO");
    if (active_ || (attached_ && attached_ != &ctx))
        throw std::logic_error("CUDA MNDO is already attached");
    release_backend_();
    workspace_->ensure(frame, static_cast<cudaStream_t>(ctx.stream),
                       ctx.exec.trial_evaluation);
    this->build_borrowed_(ctx, frame);
    attached_ = &ctx;
}
void MNDO::destroy_on(cuda::Context& ctx, cuda::Frame& frame) {
    if (!attached_)
        return;
    if (attached_ != &ctx)
        throw std::logic_error("CUDA MNDO parent mismatch");
    struct Detach {
        MNDO& method;
        ~Detach() {
            method.attached_ = nullptr;
            method.last_stream_ = nullptr;
        }
    } detach{*this};
    std::exception_ptr failure;
    try {
        ATV_CUDA_CHECK(
            cudaStreamSynchronize(static_cast<cudaStream_t>(ctx.stream)));
    } catch (...) {
        failure = std::current_exception();
    }
    try {
        this->destroy_borrowed_(ctx, frame);
    } catch (...) {
        if (!failure)
            failure = std::current_exception();
    }
    if (failure)
        std::rethrow_exception(failure);
}
int MNDO::run_nested_erased(cuda::Context& ctx, cuda::Frame& frame,
                            int budget) {
    start_();
    if (attached_ != &ctx)
        throw std::logic_error("CUDA MNDO was not built on this parent");
    try {
        const int iterations = solve_(ctx, frame, budget, true);
        if (!ctx.exec.initializing_forces && !ctx.exec.trial_evaluation) {
            if (!ctx.exec.evaluation_transaction)
                throw std::logic_error(
                    "CUDA NDDO requires its parent's evaluation transaction");
            ctx.exec.evaluation_transaction->enlist(*this);
            workspace_->stage();
        }
        return iterations;
    } catch (...) {
        workspace_->result.converged = false;
        workspace_->result.forces_ev_a.clear();
        workspace_->forces_valid = false;
        this->status_.converged = false;
        // Drain pending work without replacing the original solve failure.
        cudaStreamSynchronize(static_cast<cudaStream_t>(ctx.stream));
        throw;
    }
}
} // namespace ATV::cuda::semiempirical
