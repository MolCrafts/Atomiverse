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
#include "atomiverse/cuda/frame/types.cuh"
#include "atomiverse/cuda/scf/device_integrals.cuh"
#include "atomiverse/cuda/scf/internal/uhf_scf_cycle.cuh"

namespace ATV {
namespace cuda {
namespace scf {

/// @brief Row-major 1/2 P_MINAO, or empty when MINAO does not cover Z.
std::vector<double> uhf_minao_half(const int* Z, const double* x_A,
                                   const double* y_A, const double* z_A,
                                   int natm, cpu::scf::BasisSet basis,
                                   int charge, int n_ao);

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
        level_shift);
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

struct EngineSlot {
    std::unique_ptr<internal::UhfScfCycle> cycle;
    cuda::Context* ctx = nullptr;
    cuda::Frame* frame = nullptr;
};

class UhfStage : public cuda::Component {
  public:
    using Step = void (internal::UhfScfCycle::*)(void*, cuda::Frame&, int);
    UhfStage(EngineSlot* slot, ATV::scf::Phase phase, const char* stage_name,
             Step step)
        : slot_(slot), phase_(phase), name_(stage_name), step_(step) {}
    std::string name() const override { return name_; }
    void launch(const cuda::LaunchContext& ctx) override {
        if (!ctx.at(phase_) || !slot_ || !slot_->cycle)
            return;
        (slot_->cycle.get()->*step_)(ctx.runtime.stream, ctx.frame, ctx.step);
    }

  private:
    EngineSlot* slot_ = nullptr;
    ATV::scf::Phase phase_ = ATV::scf::Phase::fock;
    const char* name_ = "";
    Step step_ = nullptr;
};

class UhfConverged : public cuda::Criterion {
  public:
    explicit UhfConverged(EngineSlot* slot) : slot_(slot) {}
    const char* name() const override { return "scf_converged"; }
    ATV::Verdict after_iteration(const cuda::LoopState& state) override {
        if (!slot_ || !slot_->cycle)
            return ATV::Verdict::Continue;
        if (slot_->cycle->poll_converged(state.runtime.stream, state.iteration))
            return ATV::Verdict::Converged;
        return ATV::Verdict::Continue;
    }

  private:
    EngineSlot* slot_ = nullptr;
};

struct UhfLoopState {
    EngineSlot slot;
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
      energy_ha_(other.energy_ha_), iterations_(other.iterations_),
      last_force_device_(other.last_force_device_),
      last_device_sp_(other.last_device_sp_),
      host_Z_(std::move(other.host_Z_)) {
    other.energy_ha_ = 0.0;
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

void UHF::assemble_(cuda::Context& ctx, cuda::Frame& frame) {
    const int natm = frame.n_atoms();
    if (natm <= 0) {
        throw std::invalid_argument(
            "cuda::scf::UHF: device frame has no atoms");
    }
    if (static_cast<int>(host_Z_.size()) != natm) {
        host_Z_.resize(static_cast<std::size_t>(natm));
        auto* stream = static_cast<cudaStream_t>(ctx.stream);
        cudaMemcpyAsync(host_Z_.data(), frame.atoms.type,
                        sizeof(int) * static_cast<std::size_t>(natm),
                        cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);
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
    last_device_sp_ = true;
    last_force_device_ = false;
    loop_->slot.cycle->build(ctx.stream, frame);
}

void UHF::assemble_forces_(cuda::Context& ctx, cuda::Frame& frame) {
    const auto& ws = loop_->slot.cycle->workspace();
    const auto& beta = loop_->slot.cycle->beta();
    auto* stream = static_cast<cudaStream_t>(ctx.stream);
    const bool ready = ws.device_sp_shells && ws.n > 0 && ws.P && ws.C && ws.eig
                       && beta.P && beta.C && beta.eig && ws.d_shell_alpha
                       && ws.d_shell_atom && ws.d_shell_L && ws.d_ao_shell
                       && ws.d_ao_cart && ws.d_atom_pos && ws.d_W
                       && ws.d_force_Ptotal && ws.d_force_Wbeta
                       && (!kMixedPrecision || ws.d_force_gradient)
                       && frame.atoms.fx && frame.atoms.fy && frame.atoms.fz;
    if (!ready) {
        throw std::runtime_error(
            "cuda::scf::UHF: missing kernel launch_uhf_forces_sp — the device "
            "s/p force workspace is incomplete (P/C/eig/shell maps).");
    }
    const int nn = ws.n * ws.n;
    prec_t* d_Ptot = ws.d_force_Ptotal;
    prec_t* d_Wb = ws.d_force_Wbeta;
    const int threads = 256;
    const int blocks = (nn + threads - 1) / threads;
    uhf_add_mat_kernel<<<blocks, threads, 0, stream>>>(d_Ptot, ws.P, beta.P,
                                                       nn);
    ATV_CUDA_CHECK(cudaGetLastError());
    launch_build_W_uhf(ws.d_W, ws.C, ws.eig, ws.n, ws.n_occ, stream);
    launch_build_W_uhf(d_Wb, beta.C, beta.eig, ws.n, beta.n_occ, stream);
    uhf_add_mat_kernel<<<blocks, threads, 0, stream>>>(ws.d_W, ws.d_W, d_Wb,
                                                       nn);
    ATV_CUDA_CHECK(cudaGetLastError());
    launch_uhf_forces_sp(
        frame.atoms.fx, frame.atoms.fy, frame.atoms.fz, d_Ptot, ws.d_W, ws.P,
        beta.P, ws.d_shell_alpha, ws.d_shell_coeff, ws.d_shell_center,
        ws.d_shell_atom, ws.d_shell_L, ws.d_ao_shell, ws.d_ao_cart, ws.d_atom_Z,
        ws.d_atom_pos, ws.n, ws.n_prim, ws.natm, stream, ws.d_force_gradient);
    last_force_device_ = true;
    (void)ctx;
}

void UHF::pull_energy_(cuda::Context& ctx, cuda::Frame& frame) {
    double e_ev = 0;
    if (frame.scalars) {
        auto* stream = static_cast<cudaStream_t>(ctx.stream);
        ATV_CUDA_CHECK(cudaMemcpyAsync(&e_ev, &frame.scalars->energy,
                                       sizeof(e_ev), cudaMemcpyDeviceToHost,
                                       stream));
        ATV_CUDA_CHECK(cudaStreamSynchronize(stream));
    }
    energy_ha_ = static_cast<double>(e_ev) / units::kHartreeInEv;
}

void UHF::step_stages_(cuda::Context& ctx, cuda::Frame& frame, int step) {
    struct Guard {
        bool& flag;
        explicit Guard(bool& f) : flag(f) { flag = true; }
        ~Guard() { flag = false; }
    } guard(this->launching_);
    for (const auto& plan_slot : this->plan().slots) {
        if (plan_slot.ops.empty())
            continue;
        for (const auto& op : plan_slot.ops) {
            if (!op.component)
                continue;
            cuda::LaunchContext lc{ctx, frame, plan_slot.phase, step};
            op.component->launch(lc);
        }
    }
}

void UHF::seed_minao_(cuda::Context& ctx, cuda::Frame& frame) {
    const int natm = frame.n_atoms();
    if (natm <= 0 || !loop_ || !loop_->slot.cycle)
        return;
    if (static_cast<int>(host_Z_.size()) != natm)
        return;
    auto* stream = static_cast<cudaStream_t>(ctx.stream);
    std::vector<real_t> hx(static_cast<std::size_t>(natm));
    std::vector<real_t> hy(hx.size());
    std::vector<real_t> hz(hx.size());
    const auto nbyte = sizeof(real_t) * hx.size();
    cudaMemcpyAsync(hx.data(), frame.atoms.x, nbyte, cudaMemcpyDeviceToHost,
                    stream);
    cudaMemcpyAsync(hy.data(), frame.atoms.y, nbyte, cudaMemcpyDeviceToHost,
                    stream);
    cudaMemcpyAsync(hz.data(), frame.atoms.z, nbyte, cudaMemcpyDeviceToHost,
                    stream);
    cudaStreamSynchronize(stream);
    std::vector<double> x(static_cast<std::size_t>(natm));
    std::vector<double> y(x.size());
    std::vector<double> z(x.size());
    for (int i = 0; i < natm; ++i) {
        x[static_cast<std::size_t>(i)] =
            static_cast<double>(hx[static_cast<std::size_t>(i)]);
        y[static_cast<std::size_t>(i)] =
            static_cast<double>(hy[static_cast<std::size_t>(i)]);
        z[static_cast<std::size_t>(i)] =
            static_cast<double>(hz[static_cast<std::size_t>(i)]);
    }
    const int n = loop_->slot.cycle->workspace().n;
    const std::vector<double> half = uhf_minao_half(
        host_Z_.data(), x.data(), y.data(), z.data(), natm, basis, charge, n);
    if (half.empty())
        return;
    loop_->slot.cycle->seed_density(ctx.stream, half.data(), half.data(), n);
}

int UHF::drive_loop_(cuda::Context& ctx, cuda::Frame& frame, int budget) {
    struct Guard {
        UHF* self;
        ~Guard() { self->release_engine_(); }
    } guard{this};

    assemble_(ctx, frame);
    loop_->slot.cycle->begin(ctx.stream);
    seed_minao_(ctx, frame);
    const cuda::LoopServices saved = ctx.loop;
    ctx.loop.status = &this->status_;
    ctx.loop.metrics = &this->metrics_;
    this->status_.reset();
    this->metrics_ = LoopMetrics{};

    const int cap = budget > 0 ? budget : max_iter;
    int iters = 0;
    for (int i = 0; i < cap; ++i) {
        this->mutable_outer_status().reset();
        step_stages_(ctx, frame, i);
        ++iters;
        if (this->poll_handlers_(i, frame, ctx) != ATV::Verdict::Continue)
            break;
    }
    loop_->slot.cycle->finish(ctx.stream);
    if (compute_force)
        assemble_forces_(ctx, frame);
    pull_energy_(ctx, frame);
    this->iterations_ = iters;
    last_device_sp_ = true;
    ctx.loop = saved;
    return iters;
}

void UHF::build_on(cuda::Context&, cuda::Frame&) {
    if (this->plan().slots.empty())
        this->build();
}

int UHF::run_nested_erased(cuda::Context& ctx, cuda::Frame& frame, int budget) {
    return drive_loop_(ctx, frame, budget);
}

void UHF::destroy_on(cuda::Context&, cuda::Frame&) { release_engine_(); }

void UHF::run(cpu::Frame& frame) {
    cache_host_types(frame);
    this->setup_pass(frame);
    try {
        drive_loop_(this->backend_->context(), this->backend_->frame(),
                    max_iter);
    } catch (...) {
        this->teardown_pass(frame);
        throw;
    }
    this->teardown_pass(frame);
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

const SCFWorkspace& UHF::workspace() const {
    if (!loop_ || !loop_->slot.cycle) {
        throw std::runtime_error(
            "cuda::scf::UHF::workspace: no live SCF workspace. Call run() or "
            "forces(); the workspace is released when that call returns.");
    }
    return loop_->slot.cycle->workspace();
}

} // namespace scf
} // namespace cuda
} // namespace ATV

#endif
