/// @file cuda/planewave/pw_solver.cu
/// @brief cuda::pw::PW — GPU plane-wave SCF method object.
///
/// Mirrors the CPU SCF loop in src/cpu/planewave/pw_solver.cpp stage for
/// stage. Problem assembly (`setup_pw_problem`) and the per-k table binding
/// stay on the host — they are geometry work, not per-iteration work. Both
/// halves of the iteration are device-resident:
///
///   * band side — `internal::DeviceBandSolver` (cuFFT + cuBLAS Davidson),
///   * density side — `internal::DeviceDensity` (ρ(G) → V_H → V_xc (LSDA or
///     PBE GGA) → V_eff(r) → Pulay mixing → energy/residual reductions).
///
/// The two meet only through two device pointers bound once per run
/// (`bind_grids`), so **no O(N_G) or O(N_r) density-side array crosses PCIe
/// per SCF iteration**. The claim is scoped to `DeviceDensity` on purpose —
/// the band side still transfers per iteration (see below). What the density
/// side still crosses with: the Pulay Gram matrix (≤ 8²·16 B, only once the
/// history holds ≥ 2 entries), its ≤ 8 coefficients, and 5 energy/residual
/// doubles. Each is documented at its call site in device_density.cu.
///
/// The band side is geometry-hoisted too: `build_kpoints` uploads every per-k
/// table (G→FFT map, T(G), q = k+G, KB descriptors, initial Davidson subspace)
/// once per run, so `bind_kpoint` allocates nothing, copies nothing across
/// PCIe and synchronizes nothing.
///
/// The Davidson subspace eigenproblem runs on device too (cuSOLVER `Zheevd`,
/// in place), so the n_sub² Hamiltonian and the Ritz rotation no longer round
/// trip. Still NOT device-resident (named, not claimed fixed): the
/// host-pointer `cublasDznrm2` residual norms, the n_bands eigenvalues that
/// are kernel scalar arguments, and the Zheevd `devInfo` check — all inside
/// device_solver.cu, all out of scope here.
///
/// Logging goes through `cpu::pw::internal::log_iteration_energy` on purpose:
/// `core/log.h` is CUDA-free and must not be included from a `.cu`.

#include "atomiverse/cuda/planewave/pw_solver.cuh"

#include "atomiverse/core/frame_schema.h"
#include "atomiverse/core/precision.h"
#include "atomiverse/core/units.h"
#include "atomiverse/cpu/planewave/internal/convergence.h"
#include "atomiverse/cpu/planewave/internal/nc_potential.h"
#include "atomiverse/cpu/planewave/internal/pw_setup.h"
#include "atomiverse/cpu/planewave/internal/pw_workspace.h"
#include "atomiverse/cpu/planewave/qe_units.h"
#include "atomiverse/cuda/device/error.cuh"
#include "atomiverse/cuda/frame/precision.cuh"
#include "atomiverse/cuda/frame/transfer.cuh"
#include "atomiverse/cuda/planewave/internal/device_density.cuh"
#include "atomiverse/cuda/planewave/internal/device_solver.cuh"
#include "atomiverse/cuda/planewave/internal/frozen_functional_virial.cuh"
#include "atomiverse/cuda/planewave/internal/pw_forces.cuh"
#include "atomiverse/cuda/planewave/internal/pw_stress.cuh"
#include "atomiverse/electronic/smearing.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <cstdio>
#include <exception>
#include <functional>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace ATV {
namespace cuda {
namespace pw {

namespace {
bool frozen_functional_admitted(bool spin, electronic::Smearing filling,
                                cpu::xc::XCType type,
                                const cpu::pw::internal::PWProblem& p,
                                int multiplicity) noexcept {
    const long long mag = static_cast<long long>(multiplicity) - 1;
    const bool nonempty_fixed_spin =
        spin && filling == electronic::Smearing::none && mag >= 0
        && mag < p.n_elec && (p.n_elec + mag) % 2 == 0;
    return ((!spin
             && (filling == electronic::Smearing::none
                 || filling == electronic::Smearing::fermi_dirac))
            || nonempty_fixed_spin)
           && (type == cpu::xc::XCType::LSDA || cpu::xc::is_pbe_family(type))
           && p.kg.kpoints.size() == 1 && p.gv_k.size() == 1
           && p.gv_k[0].kvec.norm_squared() == 0;
}
using Step = std::function<void(int)>;
class PwStage final : public cuda::Component {
    Step* step_;
    ATV::pw::Phase phase_;
    const char* name_;

  public:
    PwStage(Step& step, ATV::pw::Phase phase, const char* name)
        : step_(&step), phase_(phase), name_(name) {}
    std::string name() const override { return name_; }
    void launch(const cuda::LaunchContext& ctx) override {
        if (ctx.at(phase_) && *step_)
            (*step_)(static_cast<int>(ctx.step));
    }
};
class PwConverged final : public cuda::Criterion {
    const bool* gate_;

  public:
    explicit PwConverged(const bool& gate) : gate_(&gate) {}
    bool contributes_to_convergence() const override { return true; }
    std::string name() const override { return "pw_converged"; }
    ATV::Verdict verdict(const cuda::LoopState&) override {
        return *gate_ ? ATV::Verdict::Converged : ATV::Verdict::Continue;
    }
};

// Download only the real system geometry at a nested execution boundary.
// The PW basis setup is host-side; density/orbitals stay in their device state.
cpu::Frame host_geometry(const cuda::Frame& device, cudaStream_t stream) {
    const int n = device.n_atoms();
    if (n <= 0 || !device.atoms.x || !device.atoms.y || !device.atoms.z
        || !device.atoms.type || !device.box)
        throw std::invalid_argument("cuda::pw::PW: incomplete parent geometry");
    cpu::Frame host(n);
    const pos_t* primary[3] = {device.atoms.x, device.atoms.y, device.atoms.z};
    const pos_t* correction[3] = {device.atoms.x_corr, device.atoms.y_corr,
                                  device.atoms.z_corr};
    if constexpr (kPositionCorrection) {
        for (const auto* column : correction)
            if (!column)
                throw std::invalid_argument(
                    "cuda::pw::PW: missing position correction");
    }
    std::array<std::vector<pos_t>, 3> hi, lo;
    for (int d = 0; d < 3; ++d) {
        hi[d].resize(n);
        lo[d].assign(n, pos_t(0));
    }
    std::vector<int> types(n);
    geom_t cell[9];
    // Host destinations must outlive queued D2H work, including a later
    // transfer failure. Construct this guard after every destination buffer.
    struct Transfers {
        cudaStream_t stream;
        bool pending = true;
        ~Transfers() {
            if (pending)
                cudaStreamSynchronize(stream);
        }
    } transfers{stream};
    for (int d = 0; d < 3; ++d) {
        ATV_CUDA_CHECK(cudaMemcpyAsync(hi[d].data(), primary[d],
                                       n * sizeof(pos_t),
                                       cudaMemcpyDeviceToHost, stream));
        if constexpr (kPositionCorrection) {
            ATV_CUDA_CHECK(cudaMemcpyAsync(lo[d].data(), correction[d],
                                           n * sizeof(pos_t),
                                           cudaMemcpyDeviceToHost, stream));
        }
    }
    ATV_CUDA_CHECK(cudaMemcpyAsync(types.data(), device.atoms.type,
                                   n * sizeof(int), cudaMemcpyDeviceToHost,
                                   stream));
    ATV_CUDA_CHECK(cudaMemcpyAsync(cell, device.box, sizeof(cell),
                                   cudaMemcpyDeviceToHost, stream));
    ATV_CUDA_CHECK(cudaStreamSynchronize(stream));
    transfers.pending = false;
    auto x = host.span(kAtom_x), y = host.span(kAtom_y), z = host.span(kAtom_z);
    auto type = host.span(kAtom_type);
    for (int i = 0; i < n; ++i) {
        x[i] = load_position(hi[0].data(), lo[0].data(), i);
        y[i] = load_position(hi[1].data(), lo[1].data(), i);
        z[i] = load_position(hi[2].data(), lo[2].data(), i);
        type[i] = types[i];
    }
    Mat3 matrix;
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            matrix(i, j) = cell[3 * i + j];
    host.box.set_cell(matrix);
    return host;
}

struct ScalarPublication {
    double energy;
    double virial[6];
};
__global__ void accumulate_scalars(SystemScalars* scalars,
                                   ScalarPublication value, bool stress) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        scalars->energy += value.energy;
        if (stress)
            for (int k = 0; k < 6; ++k)
                scalars->stress[k] += value.virial[k];
    }
}
} // namespace

struct PwLoopState {
    Step veff, diagonalize, density, mix;
    bool gate = false;
    PwStage veff_stage{veff, ATV::pw::Phase::veff, "pw_veff_stage"};
    PwStage diag_stage{diagonalize, ATV::pw::Phase::diagonalize,
                       "pw_diagonalize_stage"};
    PwStage density_stage{density, ATV::pw::Phase::density, "pw_density_stage"};
    PwStage mix_stage{mix, ATV::pw::Phase::mix, "pw_mix_stage"};
    PwConverged criterion{gate};
    void clear() {
        veff = {};
        diagonalize = {};
        density = {};
        mix = {};
    }
};

PW::PW(std::shared_ptr<const ::ATV::electronic::NormConservingPotential> pp,
       double cutoff, int mesh, cpu::xc::XCType functional)
    : potential(std::move(pp)), ecut_ry(cutoff), k_mesh(mesh), xc(functional),
      loop_(std::make_unique<PwLoopState>()) {
    if (!potential
        || (!dynamic_cast<const cpu::pw::GthPseudopotential*>(potential.get())
            && !dynamic_cast<const cpu::pw::UpfPseudopotential*>(
                potential.get())))
        throw std::invalid_argument(
            "cuda::pw::PW: only builtin GTH/UPF NC providers are supported");
    this->on(ATV::pw::Phase::veff).add_ref(loop_->veff_stage);
    this->on(ATV::pw::Phase::diagonalize).add_ref(loop_->diag_stage);
    this->on(ATV::pw::Phase::density).add_ref(loop_->density_stage);
    this->on(ATV::pw::Phase::mix).add_ref(loop_->mix_stage);
    this->add_criterion_ref(loop_->criterion);
    cuda::Driver<ATV::pw::Phase>::build();
}
PW::~PW() {
    try {
        release_backend_();
    } catch (...) {
    }
}
PW::PW(PW&& other)
    : cuda::Driver<ATV::pw::Phase>(std::move(other)),
      potential(std::move(other.potential)), ecut_ry(other.ecut_ry),
      k_mesh(other.k_mesh), xc(other.xc), tol(other.tol),
      max_iter(other.max_iter), mix_beta(other.mix_beta),
      pulay_depth(other.pulay_depth), spin_polarized(other.spin_polarized),
      multiplicity(other.multiplicity), smearing(other.smearing),
      degauss(other.degauss), mp_order(other.mp_order),
      extra_bands(other.extra_bands), built_(other.built_),
      result_(std::move(other.result_)), loop_(std::move(other.loop_)) {
    other.built_ = false;
    other.result_ = PWResult{};
}

void PW::validate_members() const {
    if (!loop_)
        throw std::logic_error("cuda::pw::PW: moved-from method has no loop");
    // One admission rule for restricted and collinear paths; each PBE-family
    // functional selects the shared physical parameters used by its kernel.
    if (xc != cpu::xc::XCType::LSDA && !cpu::xc::is_pbe_family(xc))
        throw std::invalid_argument(
            "cuda::pw::PW::run: device XC supports LSDA, PBE, PBEsol, "
            "revPBE and RPBE");
    if (spin_polarized && multiplicity < 1)
        throw std::invalid_argument(
            "cuda::pw::PW::run: multiplicity must be >= 1 when "
            "spin_polarized");
    if (!spin_polarized && multiplicity != 1)
        throw std::invalid_argument(
            "cuda::pw::PW::run: multiplicity != 1 requires spin_polarized. "
            "Set spin_polarized=true for collinear UKS, or multiplicity=1.");
    if (smearing != ATV::electronic::Smearing::none && !(degauss > 0.0))
        throw std::invalid_argument(
            "cuda::pw::PW::run: degauss must be > 0 [Ry] when smearing is "
            "enabled");
    // k_mesh = 0 is the dangerous one: generate_mp_mesh loops zero times and
    // returns an empty grid without throwing, so run() would iterate over a
    // density that is never accumulated and report a converged e_band of 0.
    if (k_mesh < 1)
        throw std::invalid_argument(
            "cuda::pw::PW::run: k_mesh must be >= 1 (Monkhorst-Pack mesh "
            "size per dimension; 1 = Γ-point only)");
    if (!(ecut_ry > 0.0))
        throw std::invalid_argument(
            "cuda::pw::PW::run: ecut_ry must be > 0 [Ry]");
    if (!(tol > 0.0))
        throw std::invalid_argument("cuda::pw::PW::run: tol must be > 0");
    if (!(mix_beta > 0.0))
        throw std::invalid_argument("cuda::pw::PW::run: mix_beta must be > 0");
    if (pulay_depth < 1)
        throw std::invalid_argument(
            "cuda::pw::PW::run: pulay_depth must be >= 1");
    if (max_iter < 1)
        throw std::invalid_argument("cuda::pw::PW::run: max_iter must be >= 1");
}

void PW::build(const cpu::Frame& reference) {
    cpu::pw::internal::validate_potential_frame(reference, *potential);
    built_ = true;
}

void PW::release_backend_() {
    if (!this->backend_)
        return;
    std::exception_ptr failure;
    try {
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

void PW::drive_stages_(cuda::Context& ctx, cuda::Frame& frame, int budget) {
    struct Restore {
        cuda::Context& ctx;
        cuda::LoopServices saved;
        PwLoopState& loop;
        ~Restore() {
            ctx.loop = saved;
            loop.clear();
        }
    } restore{ctx, ctx.loop, *loop_};
    loop_->gate = false;
    this->status_.reset();
    this->metrics_ = ConvergenceMetrics{};
    ctx.loop.status = &this->status_;
    ctx.loop.metrics = &this->metrics_;
    const int cap = budget < 0 ? max_iter : budget;
    double previous_energy = 0;
    for (int iter = 0; iter < cap; ++iter) {
        this->mutable_outer_status().reset();
        if (this->backend_ && &this->backend_->context() == &ctx
            && &this->backend_->frame() == &frame)
            this->step_once(iter);
        else
            this->step_borrowed_(ctx, frame, iter);
        this->metrics_.energy = result_.energy_ha * units::kHartreeInEv;
        this->metrics_.energy_change =
            iter == 0 ? std::numeric_limits<double>::infinity()
                      : std::abs(this->metrics_.energy - previous_energy);
        previous_energy = this->metrics_.energy;
        if (this->poll_criteria_(iter, frame, ctx) != ATV::Verdict::Continue)
            break;
    }
    result_.converged = this->converged();
}

std::array<int, 2> PW::prepare_(const cpu::Frame& frame,
                                cpu::pw::internal::PWProblem& prob) const {
    // ── Shared problem assembly (identical to the CPU backend) ──
    // PWProblem is non-copyable (ws.gvec aliases gv0): construct in place.
    cpu::pw::internal::setup_pw_problem(frame, *potential, ecut_ry, k_mesh,
                                        prob);

    // Collinear occupation split: N↑ − N↓ = multiplicity − 1, N↑ + N↓ = n_elec.
    const int n_elec = prob.n_elec;
    const bool use_smear = (smearing != ATV::electronic::Smearing::none);
    int n_alpha = n_elec / 2;
    int n_beta = n_elec - n_alpha;
    if (spin_polarized) {
        const int mag = multiplicity - 1;
        if ((n_elec + mag) % 2 != 0 || mag > n_elec)
            throw std::invalid_argument(
                "cuda::pw::PW::run: n_elec and multiplicity have incompatible "
                "parity (need N↑,N↓ non-negative integers)");
        n_alpha = (n_elec + mag) / 2;
        n_beta = (n_elec - mag) / 2;
    } else {
        // Fixed closed-shell occupations require even n_elec. Metallic smear
        // fills fractional occupations and accepts odd valence counts (Al).
        if (!use_smear && n_elec % 2 != 0)
            throw std::invalid_argument(
                "cuda::pw::PW::run: " + std::to_string(n_elec)
                + " valence electrons is not closed-shell — the restricted "
                  "fixed-occupation path occupies every band with exactly 2 "
                  "electrons. Use spin_polarized for open shell, or set "
                  "smearing for metallic occupations.");
    }

    return {n_alpha, n_beta};
}

PW::DerivativePublication
PW::evaluate_(const cpu::Frame& frame, cpu::pw::internal::PWProblem& prob,
              const std::array<int, 2>& spin_occupations, cuda::Context& ctx,
              cuda::Frame& device, int budget,
              std::vector<std::array<double, 3>>* forces_out,
              std::array<double, 6>* stress_out, cuda::Frame* force_target) {
    struct ClearCallbacks {
        PwLoopState& loop;
        ~ClearCallbacks() { loop.clear(); }
    } clear_callbacks{*loop_};
    result_ = PWResult{};
    auto& ws = prob.ws;
    const int n_elec = prob.n_elec;
    const bool use_smear = smearing != ATV::electronic::Smearing::none;
    const int n_alpha = spin_occupations[0], n_beta = spin_occupations[1];
    result_.magnetization = spin_polarized ? multiplicity - 1 : 0;

    // Mixing members are the method's, not the workspace defaults. The device
    // mixer reads them through DeviceDensity's constructor; the workspace
    // fields are kept in sync so the host `ws` never lies about the run.
    ws.mix_beta = AccReal(mix_beta);
    ws.pulay_depth = pulay_depth;

    const int n_kpts = static_cast<int>(prob.kg.kpoints.size());
    // Band count: fixed occ uses n_elec/2 (or max spin channel); smear uses
    // ceil(n_e/2) + extra_bands so empty states sit near E_F for the μ search.
    int n_bands =
        spin_polarized ? std::max(std::max(n_alpha, n_beta), 1) : prob.n_bands;
    if (use_smear) {
        const int n_occ_ceil = std::max((n_elec + 1) / 2, 1);
        n_bands = std::max({n_bands, n_occ_ceil + std::max(extra_bands, 0), 1});
    }
    const AccReal vol_bohr = prob.vol_bohr;
    // build_kpoints() leaves ws bound to the density sphere, so ws.omega is
    // stable for the whole run; the k-loop no longer touches ws at all.
    const AccReal omega_bohr = ws.omega;

    ATV::electronic::Smearing smear_kind =
        ATV::electronic::Smearing::fermi_dirac;
    if (smearing == ATV::electronic::Smearing::fermi_dirac)
        smear_kind = ATV::electronic::Smearing::fermi_dirac;
    else if (smearing == ATV::electronic::Smearing::methfessel_paxton)
        smear_kind = ATV::electronic::Smearing::methfessel_paxton;
    else if (smearing == ATV::electronic::Smearing::gaussian)
        smear_kind = ATV::electronic::Smearing::gaussian;
    const AccReal degauss_ha =
        AccReal(degauss) * AccReal(cpu::pw::qe_units::rydberg_to_hartree);

    // No MPI k-point distribution on this backend: every rank owns every k.
    result_.n_kpts = n_kpts;
    result_.n_kpts_local = n_kpts;

    // Device state: one-time H2D of the geometry-constant G-sphere tables and
    // the atomic ρ(G) seed, then nothing scientific leaves the GPU. Density
    // borrows the execution owner's stream; the band solver uses it too,
    // so all SCF stages and final publication stay ordered on that stream.
    internal::DeviceDensity den(prob, mix_beta, pulay_depth, xc, spin_polarized,
                                n_alpha, n_beta, n_elec,
                                static_cast<cudaStream_t>(ctx.stream));
    internal::DeviceBandSolver solver(ws.fft_nx, ws.fft_ny, ws.fft_nz, n_bands,
                                      den.stream());
    solver.bind_grids(den.density_accumulator(), den.effective_potential());
    // Every per-k table (G→FFT map, T(G), q = k+G, KB descriptors, initial
    // Davidson subspace) is geometry — built and uploaded once, here.
    solver.build_kpoints(prob);
    const bool frozen_slice = frozen_functional_admitted(
        spin_polarized, smearing, xc, prob, multiplicity);
    std::unique_ptr<internal::FrozenFunctionalWorkspace> frozen;
    if (frozen_slice && (forces_out || stress_out || force_target))
        frozen = std::make_unique<internal::FrozenFunctionalWorkspace>(
            frame, prob, n_bands, xc, smearing, degauss_ha,
            spin_polarized
                ? std::optional<internal::FixedSpinOccupations>{{n_alpha,
                                                                 n_beta}}
                : std::nullopt,
            den.stream(), forces_out || force_target, stress_out != nullptr);
    if (use_smear)
        solver.ensure_ritz_cache();

    // Same convergence policy object the CPU backend uses (warmup = 2); the
    // device reduces ‖ρ − ρ_old‖₂ itself, so feed the norm overload.
    cpu::pw::internal::ConvergenceCheck conv{AccReal(tol), 2};

    // Cross-phase scalars retain the original association and spin bookkeeping.
    AccReal e_band = 0, entropy_sum = 0, n_up = 0, n_dn = 0;
    AccReal mu_ha = 0, n_occ_total = 0;
    if (spin_polarized && use_smear) {
        const int n_bs = n_bands;
        std::vector<std::vector<AccReal>> evals_sk(2 * n_kpts);
        loop_->veff = [&](int iter) {
            den.sync_total_rho();
            den.hartree();
            den.exchange_correlation_spin();
        };
        loop_->diagonalize = [&](int iter) {
            for (int s = 0; s < 2; ++s) {
                den.assemble_veff_spin(s);
                for (int ik = 0; ik < n_kpts; ++ik) {
                    solver.bind_kpoint(ik);
                    solver.solve(ws, 40);
                    auto& ev =
                        evals_sk[static_cast<std::size_t>(s * n_kpts + ik)];
                    ev.assign(ws.eigenvalues.begin(), ws.eigenvalues.end());
                    if (static_cast<int>(ev.size()) > n_bs)
                        ev.resize(static_cast<std::size_t>(n_bs));
                }
            }

            std::vector<AccReal> all_evals;
            std::vector<AccReal> all_weights;
            all_evals.reserve(static_cast<std::size_t>(2 * n_kpts * n_bs));
            all_weights.reserve(static_cast<std::size_t>(2 * n_kpts * n_bs));
            for (int s = 0; s < 2; ++s) {
                for (int ik = 0; ik < n_kpts; ++ik) {
                    const AccReal w =
                        prob.kg.weights[static_cast<std::size_t>(ik)];
                    const auto& ev =
                        evals_sk[static_cast<std::size_t>(s * n_kpts + ik)];
                    for (int b = 0; b < n_bs && b < static_cast<int>(ev.size());
                         ++b) {
                        all_evals.push_back(ev[static_cast<std::size_t>(b)]);
                        all_weights.push_back(w);
                    }
                }
            }
            mu_ha = ATV::electronic::find_fermi_level(
                all_evals, all_weights, AccReal(n_elec), degauss_ha, smear_kind,
                mp_order, AccReal(1));

            e_band = 0;
            entropy_sum = 0;
            n_up = 0;
            n_dn = 0;
            for (int s = 0; s < 2; ++s) {
                den.assemble_veff_spin(s);
                den.zero_accumulator();
                for (int ik = 0; ik < n_kpts; ++ik) {
                    solver.bind_kpoint(ik);
                    solver.solve(ws, 40);
                    const AccReal w =
                        prob.kg.weights[static_cast<std::size_t>(ik)];
                    const auto& ev =
                        evals_sk[static_cast<std::size_t>(s * n_kpts + ik)];
                    std::vector<AccReal> w_one(ev.size(), w);
                    auto fill = ATV::electronic::fill_occupations(
                        ev, w_one, mu_ha, degauss_ha, smear_kind, mp_order,
                        AccReal(1));
                    entropy_sum += fill.entropy;
                    AccReal n_s = 0;
                    const int nb = static_cast<int>(fill.occ.size());
                    for (int b = 0; b < nb; ++b) {
                        e_band += w * ev[static_cast<std::size_t>(b)]
                                  * fill.occ[static_cast<std::size_t>(b)];
                        n_s += w * fill.occ[static_cast<std::size_t>(b)];
                    }
                    if (s == 0)
                        n_up += n_s;
                    else
                        n_dn += n_s;
                    solver.accumulate_density(w, omega_bohr, fill.occ.data(),
                                              nb);
                }
                den.collect_spin_out(s);
            }
        };
        // Spin outputs are collected inside each spin's shared accumulator
        // lane above; no separate extraction is needed at this phase.
        loop_->density = {};
        loop_->mix = [&](int iter) {
            den.mix_spin(iter);
            const auto s = den.reduce_spin(static_cast<double>(vol_bohr));
            ws.E_xc = AccReal(s.e_xc);
            auto terms = cpu::pw::internal::combine_energy_terms(
                e_band, AccReal(s.e_h), AccReal(s.e_xc_int), AccReal(s.e_loc),
                ws.E_xc, prob.ew_total);
            terms.e_one =
                e_band - AccReal(2.0) * AccReal(s.e_h) - AccReal(s.e_xc_int);
            const AccReal e_ent_ha = -degauss_ha * entropy_sum;
            terms.total = terms.e_one + AccReal(s.e_h) + ws.E_xc + prob.ew_total
                          + e_ent_ha;
            cpu::pw::internal::log_iteration_energy(terms, ws, prob.ew_total,
                                                    iter);
            result_.iterations = iter + 1;
            result_.energy_ha = static_cast<double>(terms.total);
            result_.energy =
                result_.energy_ha / cpu::pw::qe_units::rydberg_to_hartree;
            result_.magnetization = static_cast<double>(n_up - n_dn);
            result_.n_elec_occ = static_cast<double>(n_up + n_dn);
            result_.e_fermi = static_cast<double>(mu_ha)
                              / cpu::pw::qe_units::rydberg_to_hartree;
            result_.e_entropy = static_cast<double>(e_ent_ha)
                                / cpu::pw::qe_units::rydberg_to_hartree;
            this->metrics_.residual = s.resid;
            loop_->gate = conv.step(AccReal(s.resid));
        };
        drive_stages_(ctx, device, budget);
    } else if (spin_polarized) {
        const int n_occ[2] = {n_alpha, n_beta};
        loop_->veff = [&](int iter) {
            den.sync_total_rho();
            den.hartree();
            den.exchange_correlation_spin();
        };
        loop_->diagonalize = [&](int iter) {
            e_band = 0;
            for (int s = 0; s < 2; ++s) {
                den.assemble_veff_spin(s);
                den.zero_accumulator();
                for (int ik = 0; ik < n_kpts; ++ik) {
                    solver.bind_kpoint(ik);
                    solver.solve(ws, 40);
                    const AccReal weight = prob.kg.weights[ik];
                    for (int b = 0; b < n_occ[s]; ++b)
                        e_band += weight * ws.eigenvalues[b]; // spin-orbital
                    // f = 1 per occupied spin-orbital (CPU accumulate_kpoint).
                    solver.accumulate_density(weight, omega_bohr, n_occ[s],
                                              1.0);
                }
                den.collect_spin_out(s);
            }
        };
        // Spin outputs are collected inside each spin's shared accumulator
        // lane above; no separate extraction is needed at this phase.
        loop_->density = {};
        loop_->mix = [&](int iter) {
            den.mix_spin(iter);
            const auto s = den.reduce_spin(static_cast<double>(vol_bohr));
            ws.E_xc = AccReal(s.e_xc);

            // Spin energy: e_one = e_band − 2 e_h − ∫(ρ↑V↑+ρ↓V↓).
            auto terms = cpu::pw::internal::combine_energy_terms(
                e_band, AccReal(s.e_h), AccReal(s.e_xc_int), AccReal(s.e_loc),
                ws.E_xc, prob.ew_total);
            terms.e_one =
                e_band - AccReal(2.0) * AccReal(s.e_h) - AccReal(s.e_xc_int);
            terms.total =
                terms.e_one + AccReal(s.e_h) + ws.E_xc + prob.ew_total;

            cpu::pw::internal::log_iteration_energy(terms, ws, prob.ew_total,
                                                    iter);

            result_.iterations = iter + 1;
            result_.energy_ha = static_cast<double>(terms.total);
            result_.energy =
                result_.energy_ha / cpu::pw::qe_units::rydberg_to_hartree;

            this->metrics_.residual = s.resid;
            loop_->gate = conv.step(AccReal(s.resid));
        };
        drive_stages_(ctx, device, budget);
    } else if (use_smear) {
        std::vector<std::vector<AccReal>> evals_k(n_kpts);
        loop_->veff = [&](int iter) {
            den.hartree();
            den.exchange_correlation();
            den.assemble_veff();
        };
        loop_->diagonalize = [&](int iter) {
            for (int ik = 0; ik < n_kpts; ++ik) {
                solver.bind_kpoint(ik);
                solver.solve(ws, 40);
                evals_k[static_cast<std::size_t>(ik)].assign(
                    ws.eigenvalues.begin(), ws.eigenvalues.end());
                if (static_cast<int>(
                        evals_k[static_cast<std::size_t>(ik)].size())
                    > n_bands)
                    evals_k[static_cast<std::size_t>(ik)].resize(
                        static_cast<std::size_t>(n_bands));
                solver.stash_ritz();
            }

            std::vector<AccReal> all_evals;
            std::vector<AccReal> all_weights;
            all_evals.reserve(static_cast<std::size_t>(n_kpts * n_bands));
            all_weights.reserve(static_cast<std::size_t>(n_kpts * n_bands));
            for (int ik = 0; ik < n_kpts; ++ik) {
                const AccReal w = prob.kg.weights[static_cast<std::size_t>(ik)];
                const auto& ev = evals_k[static_cast<std::size_t>(ik)];
                for (int b = 0; b < n_bands && b < static_cast<int>(ev.size());
                     ++b) {
                    all_evals.push_back(ev[static_cast<std::size_t>(b)]);
                    all_weights.push_back(w);
                }
            }

            mu_ha = ATV::electronic::find_fermi_level(
                all_evals, all_weights, AccReal(n_elec), degauss_ha, smear_kind,
                mp_order);

            n_occ_total = 0;
            entropy_sum = 0;
            std::vector<std::vector<AccReal>> occ_k(
                static_cast<std::size_t>(n_kpts));
            for (int ik = 0; ik < n_kpts; ++ik) {
                const AccReal w = prob.kg.weights[static_cast<std::size_t>(ik)];
                const auto& ev = evals_k[static_cast<std::size_t>(ik)];
                std::vector<AccReal> w_one(ev.size(), w);
                auto fill = ATV::electronic::fill_occupations(
                    ev, w_one, mu_ha, degauss_ha, smear_kind, mp_order);
                occ_k[static_cast<std::size_t>(ik)].resize(ev.size());
                for (std::size_t b = 0; b < ev.size(); ++b)
                    occ_k[static_cast<std::size_t>(ik)][b] =
                        ATV::electronic::smear_f_band(ev[b], mu_ha, degauss_ha,
                                                      smear_kind, mp_order);
                n_occ_total += fill.n_occ;
                entropy_sum += fill.entropy;
            }

            den.zero_accumulator();
            e_band = 0;
            for (int ik = 0; ik < n_kpts; ++ik) {
                solver.load_ritz(ik);
                const AccReal weight =
                    prob.kg.weights[static_cast<std::size_t>(ik)];
                const auto& occ = occ_k[static_cast<std::size_t>(ik)];
                const auto& ev = evals_k[static_cast<std::size_t>(ik)];
                const int nb = static_cast<int>(occ.size());
                for (int b = 0; b < nb; ++b)
                    e_band += weight * ev[static_cast<std::size_t>(b)]
                              * occ[static_cast<std::size_t>(b)];
                solver.accumulate_density(weight, omega_bohr, occ.data(), nb);
            }
        };
        loop_->density = [&](int iter) { den.collect(); };
        loop_->mix = [&](int iter) {
            den.mix(iter);
            const auto s = den.reduce(static_cast<double>(vol_bohr));
            ws.E_xc = AccReal(s.e_xc);

            const auto terms = cpu::pw::internal::combine_energy_terms(
                e_band, AccReal(s.e_h), AccReal(s.e_xc_int), AccReal(s.e_loc),
                ws.E_xc, prob.ew_total);
            // Mermin free energy: F = E_KS − σ · S  (σ = degauss in Ha).
            const AccReal e_ent_ha = -degauss_ha * entropy_sum;
            const AccReal e_ha = terms.total + e_ent_ha;

            cpu::pw::internal::log_iteration_energy(terms, ws, prob.ew_total,
                                                    iter);

            result_.iterations = iter + 1;
            result_.energy_ha = static_cast<double>(e_ha);
            result_.energy =
                result_.energy_ha / cpu::pw::qe_units::rydberg_to_hartree;
            result_.e_fermi = static_cast<double>(mu_ha)
                              / cpu::pw::qe_units::rydberg_to_hartree;
            result_.n_elec_occ = static_cast<double>(n_occ_total);
            result_.e_entropy = static_cast<double>(e_ent_ha)
                                / cpu::pw::qe_units::rydberg_to_hartree;

            this->metrics_.residual = s.resid;
            loop_->gate = conv.step(AccReal(s.resid));
        };
        drive_stages_(ctx, device, budget);
    } else {
        loop_->veff = [&](int iter) {
            den.hartree();              // PoissonSolver::step
            den.exchange_correlation(); // XCBuilder::step(ws, xc)
            den.assemble_veff();        // VeffBuilder::step
        };
        loop_->diagonalize = [&](int iter) {
            den.zero_accumulator(); // DensityBuilder::begin

            e_band = 0;
            for (int ik = 0; ik < n_kpts; ++ik) {
                // The per-k workspace rebinding that used to live here — G-sphere,
                // KB/atomic-wfc table copies, the T(G) refill — was rebuilding
                // geometry that cannot change inside the SCF. It now happens once
                // in build_kpoints(); binding is a pointer swap.
                solver.bind_kpoint(ik);
                solver.solve(ws, 40);

                const AccReal weight = prob.kg.weights[ik];
                // Eigenvalues are the only band-side quantity the host needs;
                // they are already in ws (the Davidson subspace solve is a host
                // step). No wavefunction or density array is read back.
                for (int b = 0; b < n_bands; ++b)
                    e_band += weight * ws.eigenvalues[b] * AccReal(2);

                solver.accumulate_density(weight, omega_bohr);
            }
        };
        loop_->density = [&](int iter) { den.collect(); };
        loop_->mix = [&](int iter) {
            // DensityBuilder::finish, first half
            den.mix(iter); // pulay_mix (host solve of the ≤ 8×8 DIIS system)
            const auto s = den.reduce(static_cast<double>(vol_bohr));

            // The shared logger reads ws.E_xc; every other host density-side
            // array is stale after W2 and is deliberately never read again.
            ws.E_xc = AccReal(s.e_xc);

            // The SHARED decomposition. `iteration_energy` itself cannot be
            // called: its Parseval G-loop reads ws.rho_G / ws.V_*_G, which after
            // W2 live only on the device. Its combiner half can, and does — one
            // definition of e_one / total for both backends, so the
            // catastrophic-cancellation term order cannot drift apart.
            const auto terms = cpu::pw::internal::combine_energy_terms(
                e_band, AccReal(s.e_h), AccReal(s.e_xc_int), AccReal(s.e_loc),
                ws.E_xc, prob.ew_total);

            cpu::pw::internal::log_iteration_energy(terms, ws, prob.ew_total,
                                                    iter);

            result_.iterations = iter + 1;
            result_.energy_ha = static_cast<double>(terms.total);
            result_.energy =
                result_.energy_ha / cpu::pw::qe_units::rydberg_to_hartree;

            this->metrics_.residual = s.resid;
            loop_->gate = conv.step(AccReal(s.resid));
        };
        drive_stages_(ctx, device, budget);
    }

    if (frozen) {
        const auto status =
            frozen->finish(den, solver, result_, forces_out, stress_out,
                           this->borrowed_backend_ ? &device : nullptr);
        if (status == internal::FrozenFinishResult::complete)
            return this->borrowed_backend_
                       ? DerivativePublication::device_committed
                       : DerivativePublication::ordinary;
        std::array<double, 6> legacy_sigma{};
        std::vector<std::array<double, 3>> legacy_force;
        // Unsupported finite XC domain only: retain the old same-accepted-state
        // stress then force ordering, and stage both candidates before commit.
        if (stress_out)
            legacy_sigma = internal::pw_stress_from_state(frame, *this, prob,
                                                          den, n_alpha, n_beta);
        if (forces_out || force_target)
            legacy_force = internal::pw_forces_analytic_impl(
                frame, den, solver, prob, ws, n_alpha, n_beta, use_smear,
                n_elec, smear_kind, static_cast<double>(degauss_ha), mp_order,
                nullptr);
        frozen->commit_fallback(den, result_, legacy_force, legacy_sigma,
                                this->borrowed_backend_ ? &device : nullptr);
        if (forces_out)
            *forces_out = std::move(legacy_force);
        if (stress_out)
            *stress_out = legacy_sigma;
        return this->borrowed_backend_ ? DerivativePublication::device_committed
                                       : DerivativePublication::ordinary;
    }
    // Both derivatives consume this same solve, before its device state is freed.
    // The stress capture must precede the force epilogue's refreshed potentials.
    if (stress_out)
        *stress_out = internal::pw_stress_from_state(
            frame, *this, prob, den, spin_polarized ? n_alpha : -1,
            spin_polarized ? n_beta : -1);
    if (forces_out || force_target) {
        auto force = internal::pw_forces_analytic_impl(
            frame, den, solver, prob, ws, spin_polarized ? n_alpha : -1,
            spin_polarized ? n_beta : -1, use_smear, n_elec, smear_kind,
            static_cast<double>(degauss_ha), mp_order, force_target);
        if (forces_out)
            *forces_out = std::move(force);
    }
    return DerivativePublication::ordinary;
}

PWResult& PW::standalone_(const cpu::Frame& geometry,
                          std::vector<std::array<double, 3>>* forces_out,
                          std::array<double, 6>* stress_out) {
    result_ = PWResult{};
    validate_members();
    if (stress_out && !(geometry.box.volume() > 0))
        throw std::invalid_argument(
            "cuda::pw::PW::stress: non-positive cell volume");
    if (this->borrowed_backend_)
        throw std::logic_error(
            "cuda::pw::PW: standalone entry while mounted on a parent");
    // Incorporate bindings/criteria added after construction or a prior solve.
    // In particular, verdict-only criteria need their lifecycle-only entries.
    cuda::Driver<ATV::pw::Phase>::build();
    cpu::pw::internal::PWProblem problem;
    const auto occupations = prepare_(geometry, problem);
    built_ = true;
    // Materialize the actual input system, never an empty scheduling Frame.
    // The caller's arrays and metadata are published only by its chosen entry.
    cpu::Frame scratch(geometry);
    try {
        this->setup_pass(scratch);
        evaluate_(geometry, problem, occupations, this->backend_->context(),
                  this->backend_->frame(), max_iter, forces_out, stress_out);
        release_backend_();
    } catch (...) {
        if (loop_)
            loop_->clear();
        try {
            release_backend_();
        } catch (...) {
        }
        throw;
    }
    return result_;
}
PWResult& PW::run(const cpu::Frame& frame) {
    return standalone_(frame, nullptr, nullptr);
}
PWResult& PW::run(cpu::Frame& frame) {
    PWResult& res = standalone_(frame, nullptr, nullptr);
    frame.meta.get(kEnergy) = res.energy * cpu::pw::qe_units::rydberg_to_ev;
    return res;
}
PWResult& PW::forces(cpu::Frame& frame) {
    std::vector<std::array<double, 3>> force;
    standalone_(frame, &force, nullptr);
    auto fx = frame.span(kAtom_fx), fy = frame.span(kAtom_fy),
         fz = frame.span(kAtom_fz);
    for (int i = 0; i < frame.nlocal(); ++i) {
        fx[i] = force[i][0];
        fy[i] = force[i][1];
        fz[i] = force[i][2];
    }
    frame.meta.get(kEnergy) = result_.energy * cpu::pw::qe_units::rydberg_to_ev;
    return result_;
}
std::array<double, 6> PW::stress(cpu::Frame& frame) {
    std::array<double, 6> sigma{};
    standalone_(frame, nullptr, &sigma);
    auto& virial = frame.meta.get(kStress);
    for (std::size_t k = 0; k < sigma.size(); ++k)
        virial[k] = sigma[k] * frame.box.volume();
    frame.meta.get(kEnergy) = result_.energy * cpu::pw::qe_units::rydberg_to_ev;
    return sigma;
}

void PW::build_on(cuda::Context& ctx, cuda::Frame& frame) {
    validate_members();
    this->build_borrowed_(ctx, frame);
}
void PW::destroy_on(cuda::Context& ctx, cuda::Frame& frame) {
    if (loop_)
        loop_->clear();
    this->destroy_borrowed_(ctx, frame);
}
int PW::run_nested_erased(cuda::Context& ctx, cuda::Frame& device, int budget) {
    result_ = PWResult{};
    validate_members();
    if (!this->borrowed_backend_ || ctx.exec.backend != this->borrowed_backend_
        || this->borrowed_frame_ != &device)
        throw std::logic_error(
            "cuda::pw::PW: nested entry requires a matching parent");
    if (!device.scalars
        || (ctx.exec.forces_requested
            && (!device.atoms.fx || !device.atoms.fy || !device.atoms.fz)))
        throw std::invalid_argument(
            "cuda::pw::PW: parent Frame has incomplete output columns");
    const auto stream = static_cast<cudaStream_t>(ctx.stream);
    cpu::Frame geometry = host_geometry(device, stream);
    cpu::pw::internal::PWProblem problem;
    const auto occupations = prepare_(geometry, problem);
    std::array<double, 6> sigma{};
    const auto publication =
        evaluate_(geometry, problem, occupations, ctx, device, budget, nullptr,
                  ctx.exec.stress_requested ? &sigma : nullptr,
                  ctx.exec.forces_requested ? &device : nullptr);
    if (!device.scalars)
        throw std::invalid_argument(
            "cuda::pw::PW: parent Frame has no scalars");
    if (publication != DerivativePublication::device_committed) {
        ScalarPublication value{};
        value.energy = result_.energy * cpu::pw::qe_units::rydberg_to_ev;
        for (int k = 0; k < 6; ++k)
            value.virial[k] = sigma[k] * geometry.box.volume();
        accumulate_scalars<<<1, 1, 0, stream>>>(device.scalars, value,
                                                ctx.exec.stress_requested);
        ATV_CUDA_CHECK(cudaGetLastError());
    }
    return result_.iterations;
}

} // namespace pw
} // namespace cuda
} // namespace ATV
