/// @file cuda/scf/uhf.cu
/// @brief cuda::scf::UHF — device unrestricted HF.
///
/// STO-3G l ≤ 1: integrals, both Fock builds, CDIIS, diagonalization,
/// densities, and analytic forces stay on device. Other bases throw and
/// name `launch_eri_sp` / `launch_uhf_forces_sp`.

#ifdef ATV_HAS_HF

#include "atomiverse/cuda/scf/uhf.cuh"

#include <cuda_runtime.h>

#include <array>
#include <stdexcept>
#include <string>
#include <vector>

#include <xtensor/containers/xtensor.hpp>

#include "atomiverse/core/frame_schema.h"
#include "atomiverse/core/units.h"
#include "atomiverse/cpu/frame/transfer.h"
#include "atomiverse/cpu/scf/basis_data/accessors.h"
#include "atomiverse/cuda/device/error.cuh"
#include "atomiverse/cuda/frame/transfer.cuh"
#include "atomiverse/cuda/frame/types.cuh"
#include "atomiverse/cuda/scf/device_integrals.cuh"
#include "atomiverse/cuda/scf/internal/nuclear_identity.cuh"
#include "atomiverse/cuda/scf/internal/uhf_scf_cycle.cuh"

namespace ATV {
namespace cuda {
namespace scf {

namespace {

bool device_sp_sto3g_ok(cpu::scf::BasisSet basis, const int* h_Z, int natm) {
    if (basis != cpu::scf::BasisSet::STO3G || natm <= 0 || h_Z == nullptr)
        return false;
    int n_prim0 = -1;
    for (int i = 0; i < natm; ++i) {
        const int Z = h_Z[i];
        if (!cpu::scf::basis_data::is_sp_sto3g(Z))
            return false;
        const int np = cpu::scf::basis_data::sp_sto3g_n_prim(Z);
        if (n_prim0 < 0)
            n_prim0 = np;
        else if (np != n_prim0)
            return false;
    }
    return true;
}

std::unique_ptr<internal::UhfScfCycle>
make_device_solver(const int* h_Z, int natm, int n_alpha, int n_beta,
                   int charge, double tol, int max_iter, double level_shift) {
    int zsum = 0;
    for (int i = 0; i < natm; ++i)
        zsum += h_Z[i];
    const int n_elec = zsum - charge;
    if (n_alpha < 0 || n_beta < 0 || n_alpha + n_beta != n_elec) {
        throw std::invalid_argument(
            "cuda::scf::UHF: n_alpha + n_beta must equal sum(Z) - charge "
            "(n_alpha="
            + std::to_string(n_alpha) + ", n_beta=" + std::to_string(n_beta)
            + ", charge=" + std::to_string(charge)
            + ", sumZ=" + std::to_string(zsum)
            + ", n_electrons=" + std::to_string(n_elec) + ").");
    }

    int n_shells = 0;
    int n_ao = 0;
    int n_prim = -1;
    for (int a = 0; a < natm; ++a) {
        const auto& shells = cpu::scf::basis_data::sp_sto3g_shells(h_Z[a]);
        if (n_prim < 0)
            n_prim = cpu::scf::basis_data::sp_sto3g_n_prim(h_Z[a]);
        for (const auto& sh : shells) {
            ++n_shells;
            n_ao += cpu::scf::basis_data::n_cart_l(sh.l);
        }
    }
    std::vector<double> alpha_mol(static_cast<std::size_t>(n_shells)
                                  * static_cast<std::size_t>(n_prim));
    std::vector<double> coeff_mol(alpha_mol.size());
    std::vector<int> L_mol(static_cast<std::size_t>(n_shells));
    std::vector<int> atom_of(static_cast<std::size_t>(n_shells));
    int s = 0;
    for (int a = 0; a < natm; ++a) {
        const auto& shells = cpu::scf::basis_data::sp_sto3g_shells(h_Z[a]);
        for (const auto& sh : shells) {
            L_mol[static_cast<std::size_t>(s)] = sh.l;
            atom_of[static_cast<std::size_t>(s)] = a;
            for (int p = 0; p < n_prim; ++p) {
                const std::size_t k = static_cast<std::size_t>(s * n_prim + p);
                alpha_mol[k] = sh.alpha[static_cast<std::size_t>(p)];
                coeff_mol[k] = sh.coeff[static_cast<std::size_t>(p)];
            }
            ++s;
        }
    }
    return std::make_unique<internal::UhfScfCycle>(
        n_shells, n_ao, n_prim, alpha_mol.data(), coeff_mol.data(),
        L_mol.data(), atom_of.data(), n_alpha, n_beta, tol, max_iter,
        level_shift, h_Z, natm);
}

__global__ void uhf_add_mat_kernel(prec_t* __restrict__ out,
                                   const prec_t* __restrict__ a,
                                   const prec_t* __restrict__ b, int nn) {
    const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (i < nn)
        out[i] = static_cast<prec_t>(static_cast<work_t>(a[i])
                                     + static_cast<work_t>(b[i]));
}

} // namespace

struct UhfEngineSlot {
    std::unique_ptr<internal::UhfScfCycle> cycle;
    cuda::Context* ctx = nullptr;
    cuda::Frame* frame = nullptr;
};

class UhfStage : public cuda::Component {
  public:
    using Step = void (internal::UhfScfCycle::*)(void*, cuda::Frame&, int);
    UhfStage(UhfEngineSlot* slot, ATV::scf::Phase phase, const char* stage_name,
             Step step)
        : slot_(slot), phase_(phase), name_(stage_name), step_(step) {}
    std::string name() const override { return name_; }
    void launch(const cuda::LaunchContext& ctx) override {
        if (!ctx.at(phase_) || !slot_ || !slot_->cycle)
            return;
        (slot_->cycle.get()->*step_)(ctx.runtime.stream, ctx.frame, ctx.step);
    }

  private:
    UhfEngineSlot* slot_ = nullptr;
    ATV::scf::Phase phase_ = ATV::scf::Phase::fock;
    const char* name_ = "";
    Step step_ = nullptr;
};

class UhfConverged : public cuda::Criterion {
  public:
    bool contributes_to_convergence() const override { return true; }
    explicit UhfConverged(UhfEngineSlot* slot) : slot_(slot) {}
    std::string name() const override { return "scf_converged"; }
    ATV::Verdict verdict(const cuda::LoopState& state) override {
        if (!slot_ || !slot_->cycle)
            return ATV::Verdict::Continue;
        if (slot_->cycle->poll_converged(state.runtime.stream, state.iteration))
            return ATV::Verdict::Converged;
        return ATV::Verdict::Continue;
    }

  private:
    UhfEngineSlot* slot_ = nullptr;
};

struct UhfLoopState {
    UhfEngineSlot slot;
    internal::NuclearIdentity identity;
    UhfStage fock;
    UhfStage mix;
    UhfStage diag;
    UhfStage density;
    UhfStage check;
    UhfLoopState()
        : fock(&slot, ATV::scf::Phase::fock, "scf_fock_stage",
               &internal::UhfScfCycle::step_fock),
          mix(&slot, ATV::scf::Phase::mix, "scf_mix_stage",
              &internal::UhfScfCycle::step_mix),
          diag(&slot, ATV::scf::Phase::diagonalize, "scf_diagonalize_stage",
               &internal::UhfScfCycle::step_diagonalize),
          density(&slot, ATV::scf::Phase::density, "scf_density_stage",
                  &internal::UhfScfCycle::step_density),
          check(&slot, ATV::scf::Phase::check, "scf_check_stage",
                &internal::UhfScfCycle::step_check) {}
};

UHF::UHF(cpu::scf::BasisSet basis_in, int n_alpha_in, int n_beta_in,
         int charge_in, double tol_in, int max_iter_in, double level_shift_in,
         bool compute_force_in)
    : basis(basis_in), n_alpha(n_alpha_in), n_beta(n_beta_in),
      charge(charge_in), tol(tol_in), max_iter(max_iter_in),
      level_shift(level_shift_in), compute_force(compute_force_in),
      loop_(std::make_unique<UhfLoopState>()) {
    wire_stages_();
}

UHF::~UHF() { release_engine_(); }

UHF::UHF(UHF&& other) noexcept
    : cuda::Driver<ATV::scf::Phase>(std::move(other)), basis(other.basis),
      n_alpha(other.n_alpha), n_beta(other.n_beta), charge(other.charge),
      tol(other.tol), max_iter(other.max_iter), level_shift(other.level_shift),
      compute_force(other.compute_force), loop_(std::move(other.loop_)),
      energy_ha_(other.energy_ha_), energy_available_(other.energy_available_),
      iterations_(other.iterations_),
      last_force_device_(other.last_force_device_),
      last_device_sp_(other.last_device_sp_),
      host_Z_(std::move(other.host_Z_)) {
    other.energy_ha_ = 0.0;
    other.energy_available_ = false;
    other.iterations_ = 0;
    other.compute_force = false;
    other.last_force_device_ = false;
    other.last_device_sp_ = false;
}

void UHF::wire_stages_() {
    this->on(ATV::scf::Phase::fock).add_ref(loop_->fock);
    this->on(ATV::scf::Phase::mix).add_ref(loop_->mix);
    this->on(ATV::scf::Phase::diagonalize).add_ref(loop_->diag);
    this->on(ATV::scf::Phase::density).add_ref(loop_->density);
    this->on(ATV::scf::Phase::check).add_ref(loop_->check);
    this->add_criterion(UhfConverged{&loop_->slot});
    this->build();
}

void UHF::cache_host_types(const cpu::Frame& frame) {
    const int n = static_cast<int>(frame.nlocal());
    host_Z_.resize(static_cast<std::size_t>(n > 0 ? n : 0));
    if (n <= 0)
        return;
    const auto type = frame.span(kAtom_type);
    for (int i = 0; i < n; ++i)
        host_Z_[static_cast<std::size_t>(i)] = type[i];
}

void UHF::release_engine_() {
    if (!loop_ || !loop_->slot.cycle)
        return;
    cuda::Context* ctx = loop_->slot.ctx;
    cuda::Frame* frame = loop_->slot.frame;
    if (ctx && frame)
        loop_->slot.cycle->destroy(ctx->stream, *frame);
    loop_->slot.cycle.reset();
    loop_->slot.ctx = nullptr;
    loop_->slot.frame = nullptr;
}

void UHF::assemble_(cuda::Context& ctx, cuda::Frame& frame, bool host_current) {
    const int natm = frame.n_atoms();
    if (natm <= 0) {
        throw std::invalid_argument(
            "cuda::scf::UHF: device frame has no atoms");
    }
    auto stream = static_cast<cudaStream_t>(ctx.stream);
    // Host entry points already inspected the exact incoming identity. Nested
    // entries compare on device and return only a boolean; coordinates can
    // change without repeatedly transferring the nuclear-charge array.
    const bool changed =
        host_current
        || !loop_->identity.matches(frame.atoms.type, natm, stream);
    if (changed) {
        if (!host_current) {
            host_Z_.resize(static_cast<std::size_t>(natm));
            ATV_CUDA_CHECK(cudaMemcpyAsync(host_Z_.data(), frame.atoms.type,
                                           natm * sizeof(int),
                                           cudaMemcpyDeviceToHost, stream));
            ATV_CUDA_CHECK(cudaStreamSynchronize(stream));
        }
        loop_->identity.record(frame.atoms.type, natm, stream);
    }
    if (!device_sp_sto3g_ok(basis, host_Z_.data(), natm)) {
        throw std::invalid_argument(
            "cuda::scf::UHF: missing kernel launch_eri_sp (and "
            "launch_1e_integrals_sp) for this system. Device UHF covers "
            "STO-3G with l<=1 on every atom. Use cpu::scf::UHF.");
    }
    release_engine_();
    loop_->slot.ctx = &ctx;
    loop_->slot.frame = &frame;
    loop_->slot.cycle =
        make_device_solver(host_Z_.data(), natm, n_alpha, n_beta, charge, tol,
                           max_iter, level_shift);
    loop_->slot.cycle->build(ctx.stream, frame);
}

void UHF::assemble_forces_(cuda::Context& ctx, cuda::Frame& frame) {
    const auto& ws = loop_->slot.cycle->workspace();
    const auto& beta = loop_->slot.cycle->beta();
    auto* stream = static_cast<cudaStream_t>(ctx.stream);
    const bool ready = ws.device_sp_shells && ws.n > 0 && ws.P && ws.C && ws.eig
                       && beta.P && beta.C && beta.eig && ws.d_shell_alpha
                       && ws.d_shell_atom && ws.d_shell_L && ws.d_ao_shell
                       && ws.d_ao_cart && ws.d_atom_pos && ws.d_W && ws.work
                       && beta.work && ws.d_gradient && frame.atoms.fx
                       && frame.atoms.fy && frame.atoms.fz;
    if (!ready) {
        throw std::runtime_error(
            "cuda::scf::UHF: missing kernel launch_uhf_forces_sp — the device "
            "s/p force workspace is incomplete (P/C/eig/shell maps).");
    }
    const int nn = ws.n * ws.n;
    // The check-stage residual scratch is no longer needed after convergence.
    // Reuse it for total density and beta energy-weighted density.
    prec_t* d_Ptot = ws.work;
    prec_t* d_Wb = beta.work;
    const int threads = 256;
    const int blocks = (nn + threads - 1) / threads;
    uhf_add_mat_kernel<<<blocks, threads, 0, stream>>>(d_Ptot, ws.P, beta.P,
                                                       nn);
    launch_build_W_uhf(ws.d_W, ws.C, ws.eig, ws.n, ws.n_occ, stream);
    launch_build_W_uhf(d_Wb, beta.C, beta.eig, ws.n, beta.n_occ, stream);
    uhf_add_mat_kernel<<<blocks, threads, 0, stream>>>(ws.d_W, ws.d_W, d_Wb,
                                                       nn);
    // The analytic gradient is assembled in FP64 and then published to the
    // Frame's fixed-point force columns (overwrite semantics).
    double* d_grad = ws.d_gradient;
    launch_uhf_forces_sp(d_grad, d_grad + ws.natm, d_grad + 2 * ws.natm, d_Ptot,
                         ws.d_W, ws.P, beta.P, ws.d_shell_alpha,
                         ws.d_shell_coeff, ws.d_shell_center, ws.d_shell_atom,
                         ws.d_shell_L, ws.d_ao_shell, ws.d_ao_cart, ws.d_atom_Z,
                         ws.d_atom_pos, ws.n, ws.n_prim, ws.natm, stream);
    cuda::store_forces(frame, d_grad, d_grad + ws.natm, d_grad + 2 * ws.natm,
                       ws.natm, stream);
    ATV_CUDA_CHECK(cudaGetLastError());
    last_force_device_ = true;
    (void)ctx;
}

void UHF::pull_energy_(cuda::Context& ctx, cuda::Frame& frame) {
    prec_t e_ev = 0;
    if (frame.scalars) {
        auto* stream = static_cast<cudaStream_t>(ctx.stream);
        ATV_CUDA_CHECK(cudaMemcpyAsync(&e_ev, &frame.scalars->energy,
                                       sizeof(prec_t), cudaMemcpyDeviceToHost,
                                       stream));
        ATV_CUDA_CHECK(cudaStreamSynchronize(stream));
    }
    energy_ha_ = static_cast<double>(e_ev) / units::kHartreeInEv;
}

void UHF::step_stages_(cuda::Context& ctx, cuda::Frame& frame, int step) {
    if (this->backend_ && &this->backend_->context() == &ctx
        && &this->backend_->frame() == &frame) {
        // The standalone Backend has an SCF Plan: prepare_run therefore
        // does not perform the MD initial-force walk.
        this->step_once(step);
    } else {
        this->step_borrowed_(ctx, frame, step);
    }
}

int UHF::drive_loop_(cuda::Context& ctx, cuda::Frame& frame, int budget,
                     bool host_current) {
    struct Guard {
        UHF* self;
        cuda::Context& context;
        cuda::LoopServices parent_loop;
        bool complete = false;
        ~Guard() {
            context.loop = parent_loop;
            self->release_engine_();
            if (!complete) {
                self->last_device_sp_ = false;
                self->last_force_device_ = false;
            }
        }
    } guard{this, ctx, ctx.loop};

    last_device_sp_ = false;
    last_force_device_ = false;
    energy_available_ = false;
    this->status_.reset();
    assemble_(ctx, frame, host_current);
    loop_->slot.cycle->begin(ctx.stream);
    ctx.loop.status = &this->status_;
    ctx.loop.metrics = &this->metrics_;
    this->status_.reset();
    this->metrics_ = ConvergenceMetrics{};

    const int cap = budget > 0 ? budget : max_iter;
    int iters = 0;
    for (int i = 0; i < cap; ++i) {
        this->mutable_outer_status().reset();
        step_stages_(ctx, frame, i);
        ++iters;
        if (this->poll_criteria_(i, frame, ctx) != ATV::Verdict::Continue)
            break;
    }
    loop_->slot.cycle->finish(ctx.stream);
    if (!this->converged())
        throw std::runtime_error("cuda::scf::UHF: SCF did not converge within "
                                 + std::to_string(cap) + " iterations");
    if (compute_force && ctx.exec.forces_requested)
        assemble_forces_(ctx, frame);
    const bool retrieve = this->backend_ && &this->backend_->context() == &ctx
                          && &this->backend_->frame() == &frame;
    if (retrieve)
        pull_energy_(ctx, frame);
    this->iterations_ = iters;
    energy_available_ = retrieve;
    last_device_sp_ = true;
    guard.complete = true;
    return iters;
}

void UHF::build_on(cuda::Context& ctx, cuda::Frame& frame) {
    this->build_borrowed_(ctx, frame);
}

int UHF::run_nested_erased(cuda::Context& ctx, cuda::Frame& frame, int budget) {
    return drive_loop_(ctx, frame, budget);
}

void UHF::destroy_on(cuda::Context& ctx, cuda::Frame& frame) {
    release_engine_();
    this->destroy_borrowed_(ctx, frame);
}

void UHF::run(cpu::Frame& frame) {
    energy_available_ = false;
    cache_host_types(frame);
    try {
        this->setup_pass(frame);
        drive_loop_(this->backend_->context(), this->backend_->frame(),
                    max_iter, /*host_current=*/true);
        this->teardown_pass(frame, compute_force);
    } catch (...) {
        energy_available_ = false;
        // Failed device candidates must never overwrite the host Frame.
        // Preserve the original failure even if resource teardown also fails.
        if (this->backend_) {
            try {
                this->backend_->sync();
                this->backend_->destroy_components();
            } catch (...) {
            }
            try {
                this->backend_->shutdown();
            } catch (...) {
            }
            this->backend_.reset();
            this->built_plugins_.clear();
        }
        throw;
    }
}

cpu::scf::GradientResult UHF::forces(cpu::Frame& frame) {
    const bool prev = compute_force;
    compute_force = true;
    try {
        run(frame);
    } catch (...) {
        compute_force = prev;
        throw;
    }
    compute_force = prev;
    const int natm = static_cast<int>(frame.nlocal());
    xt::xtensor<double, 2> f = xt::zeros<double>(
        std::array<std::size_t, 2>{static_cast<std::size_t>(natm), 3});
    auto fx = frame.span(kAtom_fx);
    auto fy = frame.span(kAtom_fy);
    auto fz = frame.span(kAtom_fz);
    for (int A = 0; A < natm; ++A) {
        f(A, 0) = static_cast<double>(fx[A]);
        f(A, 1) = static_cast<double>(fy[A]);
        f(A, 2) = static_cast<double>(fz[A]);
    }
    const double e_ev = static_cast<double>(frame.meta.get(kEnergy));
    return cpu::scf::GradientResult{std::move(f), e_ev};
}

} // namespace scf
} // namespace cuda
} // namespace ATV

#endif
