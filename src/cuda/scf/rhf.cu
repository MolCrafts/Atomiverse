/// @file cuda/scf/rhf.cu
/// @brief cuda::scf::RHF — concrete restricted HF on the CUDA backend.
///
/// **Energy path:** device integrals only — no host ERI fill.
/// STO-3G l ≤ 1 uses `launch_*_sp`. Cartesian shells with l ≤ 3 use
/// `launch_*_lgt1` (6-31G, 6-31G*, 6-31+G*, def2-TZVP f). Spherical
/// bases with l ≤ 3 (def2-SVP, def2-TZVP) evaluate those Cartesian
/// integrals and sandwich with CartToSph on device. l ≥ 4 throws
/// `launch_eri_g`. The iteration is the Driver plan (PySCF
/// `scf.hf.kernel` / `get_veff` = J − ½K via `scf._vhf.direct`
/// `int2e_cart`).
///
/// **Forces:** STO-3G s/p analytic on device (`launch_rhf_forces_sp`).
/// Any other energy with forces requested throws
/// `launch_rhf_forces_lgt1`. Use `with_gradient(false)` for that energy,
/// or `cpu::scf::RHF` for gradients. `run(const Frame&)` does not request
/// forces and does not write the host Frame. The total is PySCF
/// `scf.hf.energy_tot` (`energy_elec` + nuclear repulsion), stored on
/// `energy()` [Hartree].

#ifdef ATV_HAS_HF

#include "atomiverse/cuda/scf/rhf.cuh"

#include <cuda_runtime.h>

#include <array>
#include <exception>
#include <stdexcept>
#include <string>
#include <vector>

#include <xtensor/containers/xtensor.hpp>

#include "atomiverse/core/frame_schema.h"
#include "atomiverse/core/units.h"
#include "atomiverse/cpu/frame/transfer.h"
#include "atomiverse/cpu/scf/basis_data/accessors.h"
#include "atomiverse/cpu/scf/dense_matrix.h"
#include "atomiverse/cpu/scf/grid.h"
#include "atomiverse/cpu/scf/internal/cartesian_shell_pack.h"
#include "atomiverse/cpu/scf/rhf.h"
#include "atomiverse/cpu/scf/scf_enums.h"
#include "atomiverse/cpu/scf/slater_determinant.h"
#include "atomiverse/cpu/xc/xc_functional.h"
#include "atomiverse/cuda/device/error.cuh"
#include "atomiverse/cuda/frame/transfer.cuh"
#include "atomiverse/cuda/frame/types.cuh"
#include "atomiverse/cuda/scf/device_integrals.cuh"
#include "atomiverse/cuda/scf/internal/nuclear_identity.cuh"
#include "atomiverse/cuda/scf/internal/rhf_scf_cycle.cuh"
#include "atomiverse/cuda/scf/mp2.cuh"

namespace ATV {
namespace cuda {
namespace scf {
namespace {

/// @brief Closed-shell occupation, or throw. Shared by the STO-3G and l>1 paths.
int require_closed_shell_occ(const int* h_Z, int natm, int charge) {
    int n_electrons = -charge;
    for (int i = 0; i < natm; ++i)
        n_electrons += h_Z[i];
    if (n_electrons < 0 || n_electrons % 2 != 0) {
        throw std::invalid_argument(
            "cuda::scf::RHF: closed-shell CUDA path requires even non-negative "
            "electron count (n_electrons="
            + std::to_string(n_electrons) + ", charge=" + std::to_string(charge)
            + ")");
    }
    return n_electrons / 2;
}

/// @brief CUDA-only RHF: molecular s/p STO-3G → device 1e/ERI + SCF (Wave 4).
///
/// Builds a full-molecule shell list from `basis_data::sp_sto3g_shells(Z)`
/// per atom (mixed Z OK). n_AO = Σ n_cart(L) with s→1, p→3.
/// @param level_shift Virtual level shift λ [Hartree]; wired on device Fock.
std::unique_ptr<internal::RhfScfCycle>
make_device_sp_sto3g_solver(const int* h_Z, int natm, int charge, double tol,
                            int max_iter, double level_shift) {
    const int n_occ = require_closed_shell_occ(h_Z, natm, charge);

    // Count shells / AOs, then fill molecular SoA from the basis table.
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

    return std::make_unique<internal::RhfScfCycle>(
        n_shells, n_ao, n_prim, alpha_mol.data(), coeff_mol.data(),
        L_mol.data(), atom_of.data(), n_occ, tol, max_iter, level_shift);
}

/// @brief Cartesian l ≤ 3 (ragged n_prim) → device 1e + ERI. No host integrals.
std::unique_ptr<internal::RhfScfCycle>
make_device_lgt1_solver(cpu::scf::BasisSet basis, const int* h_Z, int natm,
                        int charge, double tol, int max_iter,
                        double level_shift) {
    const cpu::scf::internal::CartesianShellPack pack =
        cpu::scf::internal::pack_cartesian_shells(basis, h_Z, natm);
    const int n_occ = require_closed_shell_occ(h_Z, natm, charge);
    const bool sph = cpu::scf::angular_convention(basis)
                         == cpu::scf::basis_data::AngularConvention::Spherical
                     && pack.max_l >= 2;
    return std::make_unique<internal::RhfScfCycle>(
        pack.n_shells, pack.n_ao, pack.alpha.data(), pack.coeff.data(),
        static_cast<int>(pack.alpha.size()), pack.nprim.data(),
        pack.prim_off.data(), pack.L.data(), pack.atom_of.data(), n_occ, tol,
        max_iter, level_shift, sph);
}

} // namespace

struct EngineSlot {
    std::unique_ptr<internal::RhfScfCycle> cycle;
    cuda::Context* ctx = nullptr;
    cuda::Frame* frame = nullptr;
};

class RhfStage : public cuda::Component {
  public:
    using Step = void (internal::RhfScfCycle::*)(void*, cuda::Frame&, int);

    RhfStage(EngineSlot* slot, ATV::scf::Phase phase, const char* stage_name,
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

class RhfConverged : public cuda::Criterion {
  public:
    bool contributes_to_convergence() const override { return true; }
    explicit RhfConverged(EngineSlot* slot) : slot_(slot) {}
    std::string name() const override { return "scf_converged"; }
    ATV::Verdict verdict(const cuda::LoopState& state) override {
        if (!slot_ || !slot_->cycle)
            return ATV::Verdict::Continue;
        // PySCF kernel: stop when |ΔE| and the orbital gradient are inside
        // tolerance. This path judges the density residual the check stage
        // wrote (device flag or host `workspace.converged`).
        if (slot_->cycle->poll_converged(state.runtime.stream, state.iteration))
            return ATV::Verdict::Converged;
        return ATV::Verdict::Continue;
    }

  private:
    EngineSlot* slot_ = nullptr;
};

struct RhfLoopState {
    EngineSlot slot;
    internal::NuclearIdentity identity;
    RhfStage fock;
    RhfStage mix;
    RhfStage diag;
    RhfStage density;
    RhfStage check;

    RhfLoopState()
        : fock(&slot, ATV::scf::Phase::fock, "scf_fock_stage",
               &internal::RhfScfCycle::step_fock),
          mix(&slot, ATV::scf::Phase::mix, "scf_mix_stage",
              &internal::RhfScfCycle::step_mix),
          diag(&slot, ATV::scf::Phase::diagonalize, "scf_diagonalize_stage",
               &internal::RhfScfCycle::step_diagonalize),
          density(&slot, ATV::scf::Phase::density, "scf_density_stage",
                  &internal::RhfScfCycle::step_density),
          check(&slot, ATV::scf::Phase::check, "scf_check_stage",
                &internal::RhfScfCycle::step_check) {}
};

RHF::RHF(cpu::scf::BasisSet basis_in, int charge_in, double tol_in,
         int max_iter_in, double level_shift_in, bool compute_force_in)
    : basis(basis_in), charge(charge_in), tol(tol_in), max_iter(max_iter_in),
      level_shift(level_shift_in), compute_force(compute_force_in),
      loop_(std::make_unique<RhfLoopState>()) {
    wire_stages_();
}

RHF::~RHF() { release_engine_(); }

RHF::RHF(RHF&& other) noexcept
    : cuda::Driver<ATV::scf::Phase>(std::move(other)), basis(other.basis),
      charge(other.charge), tol(other.tol), max_iter(other.max_iter),
      level_shift(other.level_shift), compute_force(other.compute_force),
      loop_(std::move(other.loop_)), energy_ha_(other.energy_ha_),
      energy_available_(other.energy_available_),
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

void RHF::wire_stages_() {
    static_assert(static_cast<int>(ATV::scf::Phase::fock) == 0);
    static_assert(static_cast<int>(ATV::scf::Phase::mix) == 1);
    static_assert(static_cast<int>(ATV::scf::Phase::diagonalize) == 2);
    static_assert(static_cast<int>(ATV::scf::Phase::density) == 3);
    static_assert(static_cast<int>(ATV::scf::Phase::check) == 4);

    this->on(ATV::scf::Phase::fock).add_ref(loop_->fock);
    this->on(ATV::scf::Phase::mix).add_ref(loop_->mix);
    this->on(ATV::scf::Phase::diagonalize).add_ref(loop_->diag);
    this->on(ATV::scf::Phase::density).add_ref(loop_->density);
    this->on(ATV::scf::Phase::check).add_ref(loop_->check);
    this->add_criterion(RhfConverged{&loop_->slot});
    this->build();
}

void RHF::cache_host_types(const cpu::Frame& frame) {
    const int n = static_cast<int>(frame.nlocal());
    if (n <= 0) {
        host_Z_.clear();
        return;
    }
    host_Z_.resize(static_cast<std::size_t>(n));
    const auto type = frame.span(kAtom_type);
    for (int i = 0; i < n; ++i)
        host_Z_[static_cast<std::size_t>(i)] = type[i];
}

void RHF::release_engine_() {
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

void RHF::assemble_(cuda::Context& ctx, cuda::Frame& frame, bool nuclear_forces,
                    bool host_current) {
    const int natm = frame.n_atoms();
    if (natm <= 0) {
        throw std::invalid_argument(
            "cuda::scf::RHF: device frame has no atoms");
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
    const int* h_Z = host_Z_.data();

    last_force_device_ = false;
    const cpu::scf::internal::CudaRhfRoute route =
        cpu::scf::internal::classify_cuda_rhf(basis, h_Z, natm);
    if (route.kind == cpu::scf::internal::CudaRhfRouteKind::MissingKernel) {
        throw std::invalid_argument(
            std::string("cuda::scf::RHF: missing kernel ")
            + route.missing_kernel
            + ". Cartesian l<=3 is launch_eri_lgt1 on device (spherical "
              "l<=3 sandwiches with CartToSph). l>=4 needs launch_eri_g. "
              "No host ERI fill — use cpu::scf::RHF for this basis.");
    }
    if (nuclear_forces
        && route.kind != cpu::scf::internal::CudaRhfRouteKind::Sto3gSp) {
        throw std::invalid_argument(
            std::string("cuda::scf::RHF::forces: missing kernel ")
            + cpu::scf::internal::kMissingKernelForcesLgt1
            + ". Device analytic gradients cover STO-3G l<=1 "
              "(launch_rhf_forces_sp) only. Cartesian l<=3 energy: "
              "with_gradient(false). Forces: "
              "cpu::scf::RHF{basis}.forces(frame).");
    }

    release_engine_();
    loop_->slot.ctx = &ctx;
    loop_->slot.frame = &frame;
    if (route.kind == cpu::scf::internal::CudaRhfRouteKind::Sto3gSp) {
        loop_->slot.cycle = make_device_sp_sto3g_solver(h_Z, natm, charge, tol,
                                                        max_iter, level_shift);
    } else {
        loop_->slot.cycle = make_device_lgt1_solver(basis, h_Z, natm, charge,
                                                    tol, max_iter, level_shift);
    }
    loop_->slot.cycle->build(ctx.stream, frame);
}

void RHF::assemble_forces_(cuda::Context& ctx, cuda::Frame& frame) {
    const auto& ws = loop_->slot.cycle->workspace();
    auto* stream = static_cast<cudaStream_t>(ctx.stream);
    const bool device_forces =
        ws.device_sp_shells && ws.n > 0 && ws.P && ws.C && ws.eig
        && ws.d_shell_alpha && ws.d_shell_atom && ws.d_shell_L && ws.d_ao_shell
        && ws.d_ao_cart && ws.d_atom_pos && ws.d_W && ws.d_gradient
        && frame.atoms.fx && frame.atoms.fy && frame.atoms.fz;
    if (!device_forces) {
        throw std::runtime_error(
            "cuda::scf::RHF: compute_force is set but the device "
            "s/p force workspace is incomplete (P/C/eig/shell maps).");
    }
    launch_build_W_rhf(ws.d_W, ws.C, ws.eig, ws.n, ws.n_occ, stream);
    // The analytic gradient is assembled in FP64 and then published to the
    // Frame's fixed-point force columns (overwrite semantics).
    double* d_grad = ws.d_gradient;
    launch_rhf_forces_sp(d_grad, d_grad + ws.natm, d_grad + 2 * ws.natm, ws.P,
                         ws.d_W, ws.d_shell_alpha, ws.d_shell_coeff,
                         ws.d_shell_center, ws.d_shell_atom, ws.d_shell_L,
                         ws.d_ao_shell, ws.d_ao_cart, ws.d_atom_Z,
                         ws.d_atom_pos, ws.n, ws.n_prim, ws.natm, stream);
    cuda::store_forces(frame, d_grad, d_grad + ws.natm, d_grad + 2 * ws.natm,
                       ws.natm, stream);
    last_force_device_ = true;
}

void RHF::pull_energy_(cuda::Context& ctx, cuda::Frame& frame) {
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

void RHF::step_stages_(cuda::Context& ctx, cuda::Frame& frame, int step) {
    if (!this->backend_ || &this->backend_->context() != &ctx
        || &this->backend_->frame() != &frame) {
        this->step_borrowed_(ctx, frame, step);
        return;
    }
    struct Guard {
        bool& flag;
        explicit Guard(bool& f) : flag(f) { flag = true; }
        ~Guard() { flag = false; }
    } guard(this->launching_);
    this->backend_->execute_phases(this->plan(), step, 0,
                                   static_cast<int>(this->plan().slots.size()));
}

int RHF::drive_loop_(cuda::Context& ctx, cuda::Frame& frame, int budget,
                     bool nuclear_forces, bool host_current, MP2* correlation,
                     bool accumulate) {
    struct Guard {
        RHF* self;
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
    assemble_(ctx, frame, nuclear_forces, host_current);
    const bool device = loop_->slot.cycle->device_integrals();
    // Correlation scratch is sized and allocated during setup, before the
    // first HF iteration. The live device orbitals/integrals never leave this
    // execution owner before the post-HF operation consumes them.
    if (correlation)
        correlation->prepare_device_(loop_->slot.cycle->workspace(), ctx, frame,
                                     accumulate);
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
        throw std::runtime_error(
            "cuda::scf::RHF: SCF did not converge within the iteration budget");
    if (nuclear_forces)
        assemble_forces_(ctx, frame);
    const bool retrieve = this->backend_ && &this->backend_->context() == &ctx
                          && &this->backend_->frame() == &frame;
    if (correlation) {
        correlation->evaluate_device_(loop_->slot.cycle->workspace(), ctx,
                                      frame, retrieve);
        if (retrieve)
            energy_ha_ = correlation->retrieve_result_(ctx);
    } else if (retrieve) {
        pull_energy_(ctx, frame);
    }
    this->iterations_ = iters;
    energy_available_ = retrieve;
    last_device_sp_ = device;
    guard.complete = true;
    return iters;
}

void RHF::build_on(cuda::Context& ctx, cuda::Frame& frame) {
    this->build_borrowed_(ctx, frame);
}

int RHF::run_nested_erased(cuda::Context& ctx, cuda::Frame& frame, int budget) {
    return drive_loop_(ctx, frame, budget,
                       compute_force && ctx.exec.forces_requested);
}

void RHF::release_backend_() {
    if (!this->backend_)
        return;
    std::exception_ptr failure;
    try {
        if (this->backend_->context().stream)
            this->backend_->sync();
        this->backend_->destroy_components();
    } catch (...) {
        failure = std::current_exception();
    }
    try {
        this->backend_->shutdown();
    } catch (...) {
        if (!failure)
            failure = std::current_exception();
    }
    this->backend_.reset();
    this->built_plugins_.clear();
    if (failure)
        std::rethrow_exception(failure);
}

void RHF::destroy_on(cuda::Context& ctx, cuda::Frame& frame) {
    release_engine_();
    this->destroy_borrowed_(ctx, frame);
}

void RHF::run(cpu::Frame& frame) {
    energy_available_ = false;
    cache_host_types(frame);
    try {
        this->setup_pass(frame);
        drive_loop_(this->backend_->context(), this->backend_->frame(),
                    max_iter, compute_force, /*host_current=*/true);
        this->teardown_pass(frame, compute_force);
    } catch (...) {
        energy_available_ = false;
        try {
            release_backend_();
        } catch (...) {
        }
        throw;
    }
}

void RHF::run(const cpu::Frame& frame) {
    energy_available_ = false;
    // Scratch so setup cannot touch the caller's columns. No retrieve:
    // to_cpu would overwrite positions, forces, and kEnergy.
    cpu::Frame scratch(frame);
    cache_host_types(scratch);
    try {
        this->setup_pass(scratch);
        drive_loop_(this->backend_->context(), this->backend_->frame(),
                    max_iter, /*nuclear_forces=*/false, /*host_current=*/true);
        release_backend_();
    } catch (...) {
        energy_available_ = false;
        try {
            release_backend_();
        } catch (...) {
        }
        throw;
    }
}

cpu::scf::GradientResult RHF::forces(cpu::Frame& frame) {
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

#endif // ATV_HAS_HF
