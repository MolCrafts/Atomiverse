/// @file cpu/pw/pw_solver.cpp
/// @brief PW implementation — orchestrates the internal step helpers
///        through the plane-wave SCF self-consistency cycle.
///
/// Low-level callers construct `PW` explicitly and execute its
/// build/run lifecycle; workflow callers use `PWEvaluator`.
///
/// Problem assembly (Ewald, G-vectors, FFT grids, V_loc, per-k KB
/// projectors / atomic wavefunctions on q = k+G, initial density) is the
/// shared `internal::setup_pw_problem()` — the same routine the CUDA
/// backend uses, so the two backends stay numerically aligned. This file
/// owns only the CPU SCF loop.
///
/// MPI: when built with `ATV_HAS_MPI` and MPI is initialized, k-points
/// are distributed round-robin across `MPI_COMM_WORLD`; the band-energy
/// scalar and the real-space density accumulation are Allreduced each
/// SCF iteration, so all ranks hold the identical mixed density and
/// take identical convergence decisions.

#include "atomiverse/cpu/planewave/pw_solver.h"

#include "atomiverse/cpu/gpw/gth_potential.h"

#include "atomiverse/core/frame_schema.h"
#include "atomiverse/core/log.h"
#include "atomiverse/core/units.h"
#include "atomiverse/cpu/planewave/internal/convergence.h"
#include "atomiverse/cpu/planewave/internal/davidson.h"
#include "atomiverse/cpu/planewave/internal/density_builder.h"
#include "atomiverse/cpu/planewave/internal/g_vector.h"
#include "atomiverse/cpu/planewave/internal/poisson.h"
#include "atomiverse/cpu/planewave/internal/pw_forces.h"
#include "atomiverse/cpu/planewave/internal/pw_setup.h"
#include "atomiverse/cpu/planewave/internal/pw_stress.h"
#include "atomiverse/cpu/planewave/internal/pw_workspace.h"
#include "atomiverse/cpu/planewave/internal/scf_loop.h"
#include "atomiverse/cpu/planewave/internal/smearing.h"
#include "atomiverse/cpu/planewave/internal/upf_reader.h"
#include "atomiverse/cpu/planewave/internal/veff_builder.h"
#include "atomiverse/cpu/planewave/internal/xc_builder.h"
#include "atomiverse/cpu/planewave/qe_units.h"

#ifdef ATV_HAS_MPI
#include <mpi.h>
#endif

#include <algorithm>
#include <cmath>
#include <complex>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace ATV {
namespace cpu {
namespace pw {

namespace {

/// @brief MPI layout for the k-point round-robin distribution.
struct MpiLayout {
    int rank = 0;
    int nprocs = 1;
    bool active = false; ///< true only when MPI is built AND initialized.
};

MpiLayout mpi_layout() {
    MpiLayout layout;
#ifdef ATV_HAS_MPI
    int initialized = 0;
    MPI_Initialized(&initialized);
    if (initialized) {
        MPI_Comm_rank(MPI_COMM_WORLD, &layout.rank);
        MPI_Comm_size(MPI_COMM_WORLD, &layout.nprocs);
        layout.active = (layout.nprocs > 1);
    }
#endif
    return layout;
}

#ifdef ATV_HAS_MPI
MPI_Datatype mpi_real_type() {
    return sizeof(AccReal) == sizeof(double) ? MPI_DOUBLE : MPI_FLOAT;
}
#endif

} // namespace

// ── PW ────────────────────────────────────────────────────────────

namespace {
void ensure_helpers(PW& self, std::unique_ptr<internal::PoissonSolver>& poisson,
                    std::unique_ptr<internal::XCBuilder>& xc,
                    std::unique_ptr<internal::VeffBuilder>& veff,
                    std::unique_ptr<internal::DensityBuilder>& density) {
    if (!poisson)
        poisson = std::make_unique<internal::PoissonSolver>();
    if (!xc)
        xc = std::make_unique<internal::XCBuilder>();
    if (!veff)
        veff = std::make_unique<internal::VeffBuilder>();
    if (!density)
        density = std::make_unique<internal::DensityBuilder>();
}
} // namespace

// Helpers are allocated on first build()/run() from current member parameters.

// Out-of-line where internal::* are complete types (required for the
// unique_ptr members over forward-declared helpers).
PW::PW() = default;
PW::~PW() = default;
PW::PW(PW&&) noexcept = default;
PW& PW::operator=(PW&&) noexcept = default;

// ── PW Driver node: multi-path stage skeleton ────────────────────────
//
// All product SCF paths (restricted / collinear × fixed-occ / metallic
// smear) wire path-local StepFns onto the same pw::Phase stages and share
// drive_stage_loop_. No free residual `for (iter) { … break; }` on product.

void PW::reset_built_driver_() {
    veff_stage_.reset();
    diag_stage_.reset();
    density_stage_.reset();
    mix_stage_.reset();
    converged_stage_.reset();
    this->pending_.clear();
    this->owned_components_.clear();
    this->plan_ = {};
    this->criteria_.clear();
    this->criteria_owned_.clear();
    this->backend_.reset();
    driver_built_ = false;
    wired_self_ = nullptr;
}

void PW::ensure_driver_built_(VeffStage::StepFn veff,
                              DiagonalizeStage::StepFn diag,
                              DensityStage::StepFn dens, MixStage::StepFn mix) {
    if (driver_built_) {
        if (wired_self_ == this)
            return;
        reset_built_driver_();
    }

    veff_stage_ = std::make_unique<VeffStage>(std::move(veff));
    diag_stage_ = std::make_unique<DiagonalizeStage>(std::move(diag));
    density_stage_ = std::make_unique<DensityStage>(std::move(dens));
    mix_stage_ = std::make_unique<MixStage>(std::move(mix));
    converged_stage_ = std::make_unique<PwConverged>(loop_converged_);

    this->on(ATV::pw::Phase::veff).add_ref(*veff_stage_);
    this->on(ATV::pw::Phase::diagonalize).add_ref(*diag_stage_);
    this->on(ATV::pw::Phase::density).add_ref(*density_stage_);
    this->on(ATV::pw::Phase::mix).add_ref(*mix_stage_);
    this->add_criterion_ref(*converged_stage_);

    this->set_serial_on_replica();
    cpu::Driver<ATV::pw::Phase>::build();
    driver_built_ = true;
    wired_self_ = this;
}

bool PW::drive_stage_loop_(int max_iter_in, PWResult& res) {
    // Stages must already be wired via ensure_driver_built_ (lambdas capture
    // the run() prologue). Serial whole-molecule loop — empty Frame is fine
    // under set_serial_on_replica.
    loop_converged_ = false;
    if (converged_stage_)
        converged_stage_->reset();

    cpu::Frame loop_frame;
    setup_pass(loop_frame);

    int executed = 0;
    for (int iter = 0; iter < max_iter_in; ++iter) {
        mutable_outer_status().reset();
        step_once(iter);
        ++executed;
        res.iterations = executed;
        if (poll_criteria_(iter) != ATV::Verdict::Continue)
            break;
    }
    teardown_pass();

    const bool conv = this->converged() || loop_converged_;
    res.converged = conv;
    return conv;
}

void PW::build(const cpu::Frame& reference) {
    if (!poisson_) {
        poisson_ = std::make_unique<internal::PoissonSolver>();
        xc_ = std::make_unique<internal::XCBuilder>();
        veff_ = std::make_unique<internal::VeffBuilder>();
        density_ = std::make_unique<internal::DensityBuilder>();
    }
    // Validate the pseudopotential up front so a bad file or an unknown GTH
    // name fails at build() rather than mid-SCF. setup_pw_problem() rebuilds
    // it at each run().
    if (upf_path.empty() == gth_name.empty())
        throw std::invalid_argument(
            "cpu::pw::PW::build: set exactly one of upf_path (a "
            "norm-conserving UPF file) or gth_name (an in-tree analytic GTH "
            "parameterization such as \"GTH-PBE-q4\"). A plane-wave run has "
            "no default pseudopotential.");
    if (!upf_path.empty()) {
        (void)internal::read_upf_v1(upf_path);
    } else {
        const int* __restrict sp_type = reference.span(kAtom_type);
        if (reference.nlocal() > 0)
            (void)gpw::GthPotential(static_cast<molrs::Element>(sp_type[0]),
                                    gth_name);
    }
    built_ = true;
}

PWResult& PW::run(const cpu::Frame& frame) {
    // Lazy build — convenience for callers that skip explicit build().
    if (!built_)
        build(frame);

    result_ = PWResult{};

    if (spin_polarized) {
        if (multiplicity < 1)
            throw std::invalid_argument(
                "PW::run: multiplicity must be >= 1 when spin_polarized");
    }

    const auto layout = mpi_layout();
    const bool log_rank0 = (layout.rank == 0);

    // ── Shared problem assembly (identical to the CUDA backend) ──
    internal::PWProblem prob;
    internal::setup_pw_problem(frame, upf_path, gth_name, ecut_ry, k_mesh,
                               prob);
    internal::PWWorkspace& ws = prob.ws;
    auto& gv0 = prob.gv0;
    const int n_kpts = static_cast<int>(prob.kg.kpoints.size());
    // Round-robin distribution observables (the scaling gate reads these):
    // rank r owns k-points {r, r+nprocs, ...} — the exact count of every
    // `for (ik = rank; ik < n_kpts; ik += nprocs)` loop below. Serial /
    // MPI-uninitialized layouts are rank 0 of 1, so all k are local.
    result_.n_kpts = n_kpts;
    result_.n_kpts_local =
        n_kpts / layout.nprocs + (layout.rank < n_kpts % layout.nprocs ? 1 : 0);
    const int n_elec = prob.n_elec;
    const AccReal vol_bohr = prob.vol_bohr;
    ws.mix_beta = AccReal(mix_beta);
    ws.pulay_depth = pulay_depth;

    // Declared here rather than beside the band count below because the
    // occupation split needs it: metallic smearing is what makes an odd
    // valence-electron count legal on the restricted path.
    const bool use_smear = (smearing != Smearing::none);

    // Collinear occupation split: N↑ − N↓ = multiplicity − 1, N↑ + N↓ = n_elec.
    int n_alpha = n_elec / 2;
    int n_beta = n_elec - n_alpha;
    if (spin_polarized) {
        const int mag = multiplicity - 1; // 2S
        if ((n_elec + mag) % 2 != 0 || mag > n_elec)
            throw std::invalid_argument(
                "PW::run: n_elec and multiplicity have incompatible parity "
                "(need N↑,N↓ non-negative integers)");
        n_alpha = (n_elec + mag) / 2;
        n_beta = (n_elec - mag) / 2;
        ws.nspin = 2;
        result_.magnetization = static_cast<double>(mag);
        const std::size_t ng = static_cast<std::size_t>(ws.n_g);
        ws.rho_spin_G.resize(2);
        ws.rho_spin_G_old.resize(2);
        ws.V_xc_spin_G.resize(2);
        for (int s = 0; s < 2; ++s) {
            ws.rho_spin_G[s].resize({ng});
            ws.rho_spin_G_old[s].resize({ng});
            ws.V_xc_spin_G[s].resize({ng});
        }
        // Seed spin densities by splitting the atomic total density.
        const AccReal fa =
            (n_elec > 0) ? AccReal(n_alpha) / AccReal(n_elec) : AccReal(0.5);
        const AccReal fb =
            (n_elec > 0) ? AccReal(n_beta) / AccReal(n_elec) : AccReal(0.5);
        for (int ig = 0; ig < ws.n_g; ++ig) {
            ws.rho_spin_G[0][ig] = fa * ws.rho_G[ig];
            ws.rho_spin_G[1][ig] = fb * ws.rho_G[ig];
            ws.rho_spin_G_old[0][ig] = ws.rho_spin_G[0][ig];
            ws.rho_spin_G_old[1][ig] = ws.rho_spin_G[1][ig];
        }
    } else {
        // Same gate the force and stress paths apply, for the same reason:
        // without smearing the restricted path derives its band count as
        // n_elec / 2, so an odd count would drop an electron and converge on
        // a different system than the one asked for. Refusing an energy the
        // gradient paths already refuse is what keeps the three in step.
        if (!use_smear)
            internal::require_closed_shell(n_elec, "PW::run");
        ws.nspin = 1;
        result_.magnetization = 0;
    }

    const int n_bands_a = std::max(n_alpha, 1);
    const int n_bands_b = std::max(n_beta, 1);
    const int n_bands =
        spin_polarized ? std::max(n_bands_a, n_bands_b) : prob.n_bands;
    // Restricted band count stays n_elec/2 from setup.
    const int n_bands_r = spin_polarized ? n_bands : prob.n_bands;
    if (use_smear && !(degauss > 0.0))
        throw std::invalid_argument(
            "PW::run: degauss must be > 0 [Ry] when smearing is enabled");
    int n_bands_eff = n_bands_r;
    if (use_smear) {
        // Empty states near E_F: ceil(N_e/2) + extra_bands (per spin channel
        // when collinear; closed-shell band when restricted).
        const int n_occ_ceil = std::max((prob.n_elec + 1) / 2, 1);
        n_bands_eff = n_occ_ceil + std::max(extra_bands, 0);
        if (spin_polarized)
            n_bands_eff = std::max(n_bands_eff, n_bands_r);
    }
    const AccReal degauss_ha =
        AccReal(degauss) * AccReal(qe_units::rydberg_to_hartree);
    internal::SmearKind smear_kind = internal::SmearKind::fermi_dirac;
    if (smearing == Smearing::fermi_dirac)
        smear_kind = internal::SmearKind::fermi_dirac;
    else if (smearing == Smearing::methfessel_paxton)
        smear_kind = internal::SmearKind::methfessel_paxton;
    else if (smearing == Smearing::gaussian)
        smear_kind = internal::SmearKind::gaussian;

    if (log_rank0)
        ATV_INFO("[pw] start atoms={} elec={} bands={} ecut_ry={} {} "
                 "k_mesh={} ranks={} spin={} mult={} Na={} Nb={} smear={}",
                 frame.nlocal(), n_elec, n_bands_eff, ecut_ry,
                 upf_path.empty() ? gth_name.c_str() : "upf", k_mesh,
                 layout.nprocs, spin_polarized ? 2 : 1, multiplicity, n_alpha,
                 n_beta, use_smear ? 1 : 0);

    // ── Step-helper preparation (grid-sized scratch) ──
    xc_->prepare(ws);
    veff_->prepare(ws);
    density_->prepare(ws);
    density_->reset(ws);
    // Davidson subspace: collinear smear uses n_bands_eff states per spin;
    // fixed-occ collinear uses max(N↑,N↓); restricted uses n_bands_eff.
    const int n_dav =
        spin_polarized ? (use_smear ? n_bands_eff : n_bands_r) : n_bands_eff;
    if (!davidson_ || davidson_bands_ != n_dav) {
        davidson_ = std::make_unique<internal::DavidsonSolver>(
            n_dav, AccReal(tol) * AccReal(0.1), 40);
        davidson_bands_ = n_dav;
    }
    davidson_->prepare(ws);
    internal::ConvergenceCheck convergence{AccReal(tol), 2};

    // ── SCF loop (all product paths → Driver stages) ──
    // Path-local StepFns capture prologue state; drive_stage_loop_ owns
    // iteration + PwConverged. Residual free `for (iter) { break; }` is gone.
    if (use_smear && spin_polarized) {
        // Collinear UKS + metallic smearing: shared μ over spin-orbitals,
        // f∈[0,1] per (spin,k,band); M = N↑−N↓ is self-consistent.
        const int n_bs = n_bands_eff;
        std::vector<xt::xarray<std::complex<AccReal>>> wfc_sk(
            static_cast<std::size_t>(2 * n_kpts));
        std::vector<std::vector<AccReal>> evals_sk(
            static_cast<std::size_t>(2 * n_kpts));
        xt::xarray<std::complex<AccReal>> rho_out[2];
        AccReal e_band = 0;
        AccReal mu_ha = AccReal(0);
        AccReal n_occ = AccReal(0);
        AccReal n_up = AccReal(0);
        AccReal n_dn = AccReal(0);
        AccReal entropy_sum = AccReal(0);
        int iter_log = 0;

        reset_built_driver_();
        ensure_driver_built_(
            /*veff=*/
            [&]() {
                for (int ig = 0; ig < ws.n_g; ++ig)
                    ws.rho_G[ig] = ws.rho_spin_G[0][ig] + ws.rho_spin_G[1][ig];
                poisson_->step(ws);
                xc_->step_spin(ws, xc);
                if (davidson_bands_ != n_bs) {
                    davidson_ = std::make_unique<internal::DavidsonSolver>(
                        n_bs, AccReal(tol) * AccReal(0.1), 40);
                    davidson_bands_ = n_bs;
                    davidson_->prepare(ws);
                }
            },
            /*diag=*/
            [&]() {
                auto* saved_gvec = ws.gvec;
                for (int s = 0; s < 2; ++s) {
                    // V_xc_σ / V_eff_σ live on the density G-sphere.
                    ws.gvec = saved_gvec;
                    ws.n_g = gv0.count();
                    for (int ig = 0; ig < ws.n_g; ++ig)
                        ws.V_xc_G[ig] = ws.V_xc_spin_G[s][ig];
                    veff_->step(ws);

                    for (int ik = layout.rank; ik < n_kpts;
                         ik += layout.nprocs) {
                        auto& gvk = prob.gv_k[ik];
                        ws.gvec = &gvk;
                        ws.n_g = gvk.count();
                        ws.T_G.resize({static_cast<std::size_t>(ws.n_g)});
                        for (int ig = 0; ig < ws.n_g; ++ig)
                            ws.T_G[ig] = gvk.ekin[ig];
                        if (!prob.kb_proj_k.empty())
                            ws.kb_proj = prob.kb_proj_k[ik];
                        if (!prob.atom_wfc_k.empty())
                            ws.atom_wfc = prob.atom_wfc_k[ik];

                        davidson_->step(ws);

                        const std::size_t sk =
                            static_cast<std::size_t>(s * n_kpts + ik);
                        evals_sk[sk].assign(ws.eigenvalues.begin(),
                                            ws.eigenvalues.end());
                        if (static_cast<int>(evals_sk[sk].size()) > n_bs)
                            evals_sk[sk].resize(static_cast<std::size_t>(n_bs));
                        wfc_sk[sk] = ws.wfc;
                    }
                }
                ws.gvec = saved_gvec;
                ws.n_g = gv0.count();

#ifdef ATV_HAS_MPI
                if (layout.active) {
                    std::vector<AccReal> pack(
                        static_cast<std::size_t>(2 * n_kpts * n_bs),
                        AccReal(0));
                    for (int s = 0; s < 2; ++s) {
                        for (int ik = layout.rank; ik < n_kpts;
                             ik += layout.nprocs) {
                            const auto& ev = evals_sk[static_cast<std::size_t>(
                                s * n_kpts + ik)];
                            for (int b = 0;
                                 b < n_bs && b < static_cast<int>(ev.size());
                                 ++b)
                                pack[static_cast<std::size_t>(
                                    (s * n_kpts + ik) * n_bs + b)] = ev[b];
                        }
                    }
                    MPI_Allreduce(MPI_IN_PLACE, pack.data(), 2 * n_kpts * n_bs,
                                  mpi_real_type(), MPI_SUM, MPI_COMM_WORLD);
                    for (int s = 0; s < 2; ++s) {
                        for (int ik = 0; ik < n_kpts; ++ik) {
                            auto& ev = evals_sk[static_cast<std::size_t>(
                                s * n_kpts + ik)];
                            ev.resize(static_cast<std::size_t>(n_bs));
                            for (int b = 0; b < n_bs; ++b)
                                ev[static_cast<std::size_t>(b)] =
                                    pack[static_cast<std::size_t>(
                                        (s * n_kpts + ik) * n_bs + b)];
                        }
                    }
                }
#endif
            },
            /*density=*/
            [&]() {
                auto* saved_gvec = ws.gvec;
                std::vector<AccReal> all_evals;
                std::vector<AccReal> all_weights;
                all_evals.reserve(static_cast<std::size_t>(2 * n_kpts * n_bs));
                all_weights.reserve(
                    static_cast<std::size_t>(2 * n_kpts * n_bs));
                for (int s = 0; s < 2; ++s) {
                    for (int ik = 0; ik < n_kpts; ++ik) {
                        const AccReal w =
                            prob.kg.weights[static_cast<std::size_t>(ik)];
                        const auto& ev =
                            evals_sk[static_cast<std::size_t>(s * n_kpts + ik)];
                        for (int b = 0;
                             b < n_bs && b < static_cast<int>(ev.size()); ++b) {
                            all_evals.push_back(
                                ev[static_cast<std::size_t>(b)]);
                            all_weights.push_back(w);
                        }
                    }
                }

                mu_ha = internal::find_fermi_level(
                    all_evals, all_weights, AccReal(prob.n_elec), degauss_ha,
                    smear_kind, mp_order, AccReal(1));

                n_occ = AccReal(0);
                n_up = AccReal(0);
                n_dn = AccReal(0);
                entropy_sum = AccReal(0);
                std::vector<std::vector<AccReal>> occ_sk(
                    static_cast<std::size_t>(2 * n_kpts));

                for (int s = 0; s < 2; ++s) {
                    for (int ik = 0; ik < n_kpts; ++ik) {
                        const AccReal w =
                            prob.kg.weights[static_cast<std::size_t>(ik)];
                        const auto& ev =
                            evals_sk[static_cast<std::size_t>(s * n_kpts + ik)];
                        std::vector<AccReal> w_one(ev.size(), w);
                        auto fill = internal::fill_occupations(
                            ev, w_one, mu_ha, degauss_ha, smear_kind, mp_order,
                            AccReal(1));
                        occ_sk[static_cast<std::size_t>(s * n_kpts + ik)] =
                            fill.occ;
                        n_occ += fill.n_occ;
                        entropy_sum += fill.entropy;
                        AccReal n_s = AccReal(0);
                        for (std::size_t b = 0; b < fill.occ.size(); ++b)
                            n_s += w * fill.occ[b];
                        if (s == 0)
                            n_up += n_s;
                        else
                            n_dn += n_s;
                    }
                }

                e_band = 0;
                for (int s = 0; s < 2; ++s) {
                    density_->begin(ws);
                    for (int ik = layout.rank; ik < n_kpts;
                         ik += layout.nprocs) {
                        auto& gvk = prob.gv_k[ik];
                        ws.gvec = &gvk;
                        ws.n_g = gvk.count();
                        const std::size_t sk =
                            static_cast<std::size_t>(s * n_kpts + ik);
                        ws.wfc = wfc_sk[sk];
                        ws.eigenvalues.resize({evals_sk[sk].size()});
                        for (std::size_t b = 0; b < evals_sk[sk].size(); ++b)
                            ws.eigenvalues[b] = evals_sk[sk][b];

                        const AccReal weight =
                            prob.kg.weights[static_cast<std::size_t>(ik)];
                        const auto& occ = occ_sk[sk];
                        const int nb = static_cast<int>(occ.size());
                        for (int b = 0; b < nb; ++b)
                            e_band += weight * ws.eigenvalues[b]
                                      * occ[static_cast<std::size_t>(b)];

                        density_->accumulate_kpoint(ws, weight, nb, occ.data());
                    }
                    ws.gvec = saved_gvec;
                    ws.n_g = gv0.count();

#ifdef ATV_HAS_MPI
                    if (layout.active) {
                        MPI_Allreduce(MPI_IN_PLACE, density_->accumulation(),
                                      ws.nr, mpi_real_type(), MPI_SUM,
                                      MPI_COMM_WORLD);
                    }
#endif
                    density_->extract(ws, rho_out[s]);
                }

#ifdef ATV_HAS_MPI
                if (layout.active) {
                    MPI_Allreduce(MPI_IN_PLACE, &e_band, 1, mpi_real_type(),
                                  MPI_SUM, MPI_COMM_WORLD);
                }
#endif
            },
            /*mix=*/
            [&]() {
                for (int s = 0; s < 2; ++s) {
                    for (int ig = 0; ig < ws.n_g; ++ig) {
                        ws.rho_spin_G_old[s][ig] = ws.rho_spin_G[s][ig];
                        ws.rho_spin_G[s][ig] =
                            ws.rho_spin_G_old[s][ig]
                            + AccReal(mix_beta)
                                  * (rho_out[s][ig] - ws.rho_spin_G_old[s][ig]);
                    }
                }
                for (int ig = 0; ig < ws.n_g; ++ig) {
                    ws.rho_G_old[ig] = ws.rho_G[ig];
                    ws.rho_G[ig] = ws.rho_spin_G[0][ig] + ws.rho_spin_G[1][ig];
                }

                AccReal dr = 0;
                for (int ig = 0; ig < ws.n_g; ++ig) {
                    auto d = ws.rho_G[ig] - ws.rho_G_old[ig];
                    dr += d.real() * d.real() + d.imag() * d.imag();
                }
                loop_converged_ = (std::sqrt(dr) < AccReal(tol));
                result_.magnetization = static_cast<double>(n_up - n_dn);
                result_.n_elec_occ = static_cast<double>(n_occ);
                result_.e_fermi =
                    static_cast<double>(mu_ha) / qe_units::rydberg_to_hartree;

                auto terms = internal::iteration_energy(
                    ws, e_band, prob.ew_total, vol_bohr);
                terms.e_one =
                    e_band - AccReal(2.0) * terms.e_h - ws.e_xc_int_spin;
                terms.total = terms.e_one + terms.e_h + ws.E_xc + prob.ew_total
                              + ws.E_aug;
                const AccReal e_ent_ha = -degauss_ha * entropy_sum;
                terms.total += e_ent_ha;
                result_.energy_ha = static_cast<double>(terms.total);
                result_.energy =
                    result_.energy_ha / qe_units::rydberg_to_hartree;
                result_.e_entropy = static_cast<double>(e_ent_ha)
                                    / qe_units::rydberg_to_hartree;
                internal::log_iteration_energy(terms, ws, prob.ew_total,
                                               iter_log);
                ++iter_log;
            });

        const bool conv = drive_stage_loop_(max_iter, result_);
        if (conv) {
            if (log_rank0)
                ATV_INFO("[pw] converged iter={} energy_ry={:.8f} "
                         "e_fermi_ry={:.6f} n_occ={:.6f} mag={:.4f}",
                         result_.iterations, result_.energy, result_.e_fermi,
                         result_.n_elec_occ, result_.magnetization);
        } else if (log_rank0) {
            ATV_WARN("[pw] not converged after {} iters energy_ry={:.8f}",
                     max_iter, result_.energy);
        }

    } else if (use_smear) {
        // Restricted metallic-smearing SCF via Driver stages.
        std::vector<xt::xarray<std::complex<AccReal>>> wfc_k(
            static_cast<std::size_t>(n_kpts));
        std::vector<std::vector<AccReal>> evals_k(
            static_cast<std::size_t>(n_kpts));
        AccReal e_band = 0;
        AccReal mu_ha = AccReal(0);
        AccReal n_occ = AccReal(0);
        AccReal entropy_sum = AccReal(0);
        int iter_log = 0;

        reset_built_driver_();
        ensure_driver_built_(
            /*veff=*/
            [&]() {
                poisson_->step(ws);
                xc_->step(ws, xc);
                veff_->step(ws);
            },
            /*diag=*/
            [&]() {
                auto* saved_gvec = ws.gvec;
                for (int ik = layout.rank; ik < n_kpts; ik += layout.nprocs) {
                    auto& gvk = prob.gv_k[ik];
                    ws.gvec = &gvk;
                    ws.n_g = gvk.count();
                    ws.T_G.resize({static_cast<std::size_t>(ws.n_g)});
                    for (int ig = 0; ig < ws.n_g; ++ig)
                        ws.T_G[ig] = gvk.ekin[ig];
                    if (!prob.kb_proj_k.empty())
                        ws.kb_proj = prob.kb_proj_k[ik];
                    if (!prob.atom_wfc_k.empty())
                        ws.atom_wfc = prob.atom_wfc_k[ik];

                    davidson_->step(ws);

                    evals_k[static_cast<std::size_t>(ik)].assign(
                        ws.eigenvalues.begin(), ws.eigenvalues.end());
                    if (static_cast<int>(
                            evals_k[static_cast<std::size_t>(ik)].size())
                        > n_bands_eff)
                        evals_k[static_cast<std::size_t>(ik)].resize(
                            static_cast<std::size_t>(n_bands_eff));
                    wfc_k[static_cast<std::size_t>(ik)] = ws.wfc;
                }
                ws.gvec = saved_gvec;
                ws.n_g = gv0.count();

#ifdef ATV_HAS_MPI
                if (layout.active) {
                    std::vector<AccReal> pack(
                        static_cast<std::size_t>(n_kpts * n_bands_eff),
                        AccReal(0));
                    for (int ik = layout.rank; ik < n_kpts;
                         ik += layout.nprocs) {
                        const auto& ev = evals_k[static_cast<std::size_t>(ik)];
                        for (int b = 0;
                             b < n_bands_eff && b < static_cast<int>(ev.size());
                             ++b)
                            pack[static_cast<std::size_t>(ik * n_bands_eff
                                                          + b)] = ev[b];
                    }
                    MPI_Allreduce(MPI_IN_PLACE, pack.data(),
                                  n_kpts * n_bands_eff, mpi_real_type(),
                                  MPI_SUM, MPI_COMM_WORLD);
                    for (int ik = 0; ik < n_kpts; ++ik) {
                        evals_k[static_cast<std::size_t>(ik)].resize(
                            static_cast<std::size_t>(n_bands_eff));
                        for (int b = 0; b < n_bands_eff; ++b)
                            evals_k[static_cast<std::size_t>(ik)][b] =
                                pack[static_cast<std::size_t>(ik * n_bands_eff
                                                              + b)];
                    }
                }
#endif
            },
            /*density=*/
            [&]() {
                auto* saved_gvec = ws.gvec;
                std::vector<AccReal> all_evals;
                std::vector<AccReal> all_weights;
                all_evals.reserve(
                    static_cast<std::size_t>(n_kpts * n_bands_eff));
                all_weights.reserve(
                    static_cast<std::size_t>(n_kpts * n_bands_eff));

                for (int ik = 0; ik < n_kpts; ++ik) {
                    const AccReal w =
                        prob.kg.weights[static_cast<std::size_t>(ik)];
                    const auto& ev = evals_k[static_cast<std::size_t>(ik)];
                    for (int b = 0;
                         b < n_bands_eff && b < static_cast<int>(ev.size());
                         ++b) {
                        all_evals.push_back(ev[static_cast<std::size_t>(b)]);
                        all_weights.push_back(w);
                    }
                }

                mu_ha = internal::find_fermi_level(
                    all_evals, all_weights, AccReal(prob.n_elec), degauss_ha,
                    smear_kind, mp_order);
                n_occ = AccReal(0);
                entropy_sum = AccReal(0);
                std::vector<std::vector<AccReal>> occ_k(
                    static_cast<std::size_t>(n_kpts));

                for (int ik = 0; ik < n_kpts; ++ik) {
                    const AccReal w =
                        prob.kg.weights[static_cast<std::size_t>(ik)];
                    const auto& ev = evals_k[static_cast<std::size_t>(ik)];
                    std::vector<AccReal> w_one(ev.size(), w);
                    auto fill = internal::fill_occupations(
                        ev, w_one, mu_ha, degauss_ha, smear_kind, mp_order);
                    occ_k[static_cast<std::size_t>(ik)].resize(ev.size());
                    for (std::size_t b = 0; b < ev.size(); ++b)
                        occ_k[static_cast<std::size_t>(ik)][b] =
                            internal::smear_f_band(ev[b], mu_ha, degauss_ha,
                                                   smear_kind, mp_order);
                    n_occ += fill.n_occ;
                    entropy_sum += fill.entropy;
                }

                density_->begin(ws);
                e_band = 0;
                for (int ik = layout.rank; ik < n_kpts; ik += layout.nprocs) {
                    auto& gvk = prob.gv_k[ik];
                    ws.gvec = &gvk;
                    ws.n_g = gvk.count();
                    ws.wfc = wfc_k[static_cast<std::size_t>(ik)];
                    ws.eigenvalues.resize(
                        {evals_k[static_cast<std::size_t>(ik)].size()});
                    for (std::size_t b = 0;
                         b < evals_k[static_cast<std::size_t>(ik)].size(); ++b)
                        ws.eigenvalues[b] =
                            evals_k[static_cast<std::size_t>(ik)][b];

                    const AccReal weight =
                        prob.kg.weights[static_cast<std::size_t>(ik)];
                    const auto& occ = occ_k[static_cast<std::size_t>(ik)];
                    const int nb = static_cast<int>(occ.size());
                    for (int b = 0; b < nb; ++b)
                        e_band += weight * ws.eigenvalues[b]
                                  * occ[static_cast<std::size_t>(b)];

                    density_->accumulate_kpoint(ws, weight, nb, occ.data());
                }
                ws.gvec = saved_gvec;
                ws.n_g = gv0.count();

#ifdef ATV_HAS_MPI
                if (layout.active) {
                    MPI_Allreduce(MPI_IN_PLACE, density_->accumulation(), ws.nr,
                                  mpi_real_type(), MPI_SUM, MPI_COMM_WORLD);
                    MPI_Allreduce(MPI_IN_PLACE, &e_band, 1, mpi_real_type(),
                                  MPI_SUM, MPI_COMM_WORLD);
                }
#endif
            },
            /*mix=*/
            [&]() {
                density_->finish(ws);
                loop_converged_ = convergence.step(ws);

                const auto terms = internal::iteration_energy(
                    ws, e_band, prob.ew_total, vol_bohr);
                AccReal e_ha = terms.total;
                AccReal e_ent_ha = AccReal(0);
                e_ent_ha = -degauss_ha * entropy_sum;
                e_ha += e_ent_ha;
                result_.energy_ha = static_cast<double>(e_ha);
                result_.energy =
                    result_.energy_ha / qe_units::rydberg_to_hartree;
                result_.e_fermi =
                    static_cast<double>(mu_ha) / qe_units::rydberg_to_hartree;
                result_.n_elec_occ = static_cast<double>(n_occ);
                result_.e_entropy = static_cast<double>(e_ent_ha)
                                    / qe_units::rydberg_to_hartree;

                internal::log_iteration_energy(terms, ws, prob.ew_total,
                                               iter_log);
                ++iter_log;
            });

        const bool conv = drive_stage_loop_(max_iter, result_);
        if (conv) {
            if (log_rank0)
                ATV_INFO("[pw] converged iter={} energy_ry={:.8f} "
                         "e_fermi_ry={:.6f} n_occ={:.6f}",
                         result_.iterations, result_.energy, result_.e_fermi,
                         result_.n_elec_occ);
        } else if (log_rank0) {
            ATV_WARN("[pw] not converged after {} iters energy_ry={:.8f}",
                     max_iter, result_.energy);
        }

    } else if (spin_polarized) {
        // Collinear fixed-occupation SCF via Driver stages.
        AccReal e_band = 0;
        xt::xarray<std::complex<AccReal>> rho_out[2];
        int iter_log = 0;
        const int n_occ[2] = {n_alpha, n_beta};

        reset_built_driver_();
        ensure_driver_built_(
            /*veff=*/
            [&]() {
                for (int ig = 0; ig < ws.n_g; ++ig)
                    ws.rho_G[ig] = ws.rho_spin_G[0][ig] + ws.rho_spin_G[1][ig];
                poisson_->step(ws);
                xc_->step_spin(ws, xc);
                if (davidson_bands_ != n_bands_r) {
                    davidson_ = std::make_unique<internal::DavidsonSolver>(
                        n_bands_r, AccReal(tol) * AccReal(0.1), 40);
                    davidson_bands_ = n_bands_r;
                    davidson_->prepare(ws);
                }
            },
            /*diag=*/
            [&]() {
                auto* saved_gvec = ws.gvec;
                e_band = 0;
                for (int s = 0; s < 2; ++s) {
                    for (int ig = 0; ig < ws.n_g; ++ig)
                        ws.V_xc_G[ig] = ws.V_xc_spin_G[s][ig];
                    veff_->step(ws);

                    density_->begin(ws);
                    for (int ik = layout.rank; ik < n_kpts;
                         ik += layout.nprocs) {
                        auto& gvk = prob.gv_k[ik];
                        ws.gvec = &gvk;
                        ws.n_g = gvk.count();
                        ws.T_G.resize({static_cast<std::size_t>(ws.n_g)});
                        for (int ig = 0; ig < ws.n_g; ++ig)
                            ws.T_G[ig] = gvk.ekin[ig];
                        if (!prob.kb_proj_k.empty())
                            ws.kb_proj = prob.kb_proj_k[ik];
                        if (!prob.atom_wfc_k.empty())
                            ws.atom_wfc = prob.atom_wfc_k[ik];

                        davidson_->step(ws);

                        const AccReal weight = prob.kg.weights[ik];
                        for (int b = 0; b < n_occ[s]; ++b)
                            e_band += weight * ws.eigenvalues[b];

                        density_->accumulate_kpoint(ws, weight, n_occ[s],
                                                    AccReal(1));

                        ws.gvec = saved_gvec;
                        ws.n_g = gv0.count();
                    }

#ifdef ATV_HAS_MPI
                    if (layout.active) {
                        MPI_Allreduce(MPI_IN_PLACE, density_->accumulation(),
                                      ws.nr, mpi_real_type(), MPI_SUM,
                                      MPI_COMM_WORLD);
                    }
#endif
                    density_->extract(ws, rho_out[s]);
                }

#ifdef ATV_HAS_MPI
                if (layout.active) {
                    MPI_Allreduce(MPI_IN_PLACE, &e_band, 1, mpi_real_type(),
                                  MPI_SUM, MPI_COMM_WORLD);
                }
#endif
            },
            /*density=*/
            [&]() {
                // Pre-mix bookkeeping hook (spin extract already done).
            },
            /*mix=*/
            [&]() {
                for (int s = 0; s < 2; ++s) {
                    for (int ig = 0; ig < ws.n_g; ++ig) {
                        ws.rho_spin_G_old[s][ig] = ws.rho_spin_G[s][ig];
                        ws.rho_spin_G[s][ig] =
                            ws.rho_spin_G_old[s][ig]
                            + AccReal(mix_beta)
                                  * (rho_out[s][ig] - ws.rho_spin_G_old[s][ig]);
                    }
                }
                for (int ig = 0; ig < ws.n_g; ++ig) {
                    ws.rho_G_old[ig] = ws.rho_G[ig];
                    ws.rho_G[ig] = ws.rho_spin_G[0][ig] + ws.rho_spin_G[1][ig];
                }

                AccReal dr = 0;
                for (int ig = 0; ig < ws.n_g; ++ig) {
                    auto d = ws.rho_G[ig] - ws.rho_G_old[ig];
                    dr += d.real() * d.real() + d.imag() * d.imag();
                }
                loop_converged_ = (std::sqrt(dr) < AccReal(tol));

                auto terms = internal::iteration_energy(
                    ws, e_band, prob.ew_total, vol_bohr);
                terms.e_one =
                    e_band - AccReal(2.0) * terms.e_h - ws.e_xc_int_spin;
                terms.total = terms.e_one + terms.e_h + ws.E_xc + prob.ew_total
                              + ws.E_aug;
                result_.energy_ha = static_cast<double>(terms.total);
                result_.energy =
                    result_.energy_ha / qe_units::rydberg_to_hartree;
                internal::log_iteration_energy(terms, ws, prob.ew_total,
                                               iter_log);
                ++iter_log;
            });

        const bool conv = drive_stage_loop_(max_iter, result_);
        if (conv) {
            if (log_rank0)
                ATV_INFO("[pw] converged iter={} energy_ry={:.8f} mag={}",
                         result_.iterations, result_.energy,
                         result_.magnetization);
        } else if (log_rank0) {
            ATV_WARN("[pw] not converged after {} iters energy_ry={:.8f}",
                     max_iter, result_.energy);
        }

    } else {
        // ── Restricted fixed-occupation path — Driver stages ──
        AccReal e_band = 0;
        int iter_log = 0;

        reset_built_driver_();
        ensure_driver_built_(
            /*veff=*/
            [&]() {
                poisson_->step(ws);
                xc_->step(ws, xc);
                veff_->step(ws);
            },
            /*diag=*/
            [&]() {
                auto* saved_gvec = ws.gvec;
                density_->begin(ws);
                e_band = 0;
                for (int ik = layout.rank; ik < n_kpts; ik += layout.nprocs) {
                    auto& gvk = prob.gv_k[ik];
                    ws.gvec = &gvk;
                    ws.n_g = gvk.count();
                    ws.T_G.resize({static_cast<std::size_t>(ws.n_g)});
                    for (int ig = 0; ig < ws.n_g; ++ig)
                        ws.T_G[ig] = gvk.ekin[ig];
                    if (!prob.kb_proj_k.empty())
                        ws.kb_proj = prob.kb_proj_k[ik];
                    if (!prob.atom_wfc_k.empty())
                        ws.atom_wfc = prob.atom_wfc_k[ik];

                    davidson_->step(ws);

                    const AccReal weight = prob.kg.weights[ik];
                    for (int b = 0; b < n_bands_r; ++b)
                        e_band += weight * ws.eigenvalues[b] * AccReal(2);

                    density_->accumulate_kpoint(ws, weight, n_bands_r);

                    ws.gvec = saved_gvec;
                    ws.n_g = gv0.count();
                }

#ifdef ATV_HAS_MPI
                if (layout.active) {
                    MPI_Allreduce(MPI_IN_PLACE, density_->accumulation(), ws.nr,
                                  mpi_real_type(), MPI_SUM, MPI_COMM_WORLD);
                    MPI_Allreduce(MPI_IN_PLACE, &e_band, 1, mpi_real_type(),
                                  MPI_SUM, MPI_COMM_WORLD);
                }
#endif
            },
            /*density=*/
            [&]() {
                // Pre-mix bookkeeping hook (accumulation already done).
            },
            /*mix=*/
            [&]() {
                density_->finish(ws);
                loop_converged_ = convergence.step(ws);

                const auto terms = internal::iteration_energy(
                    ws, e_band, prob.ew_total, vol_bohr);
                result_.energy_ha = static_cast<double>(terms.total);
                result_.energy =
                    result_.energy_ha / qe_units::rydberg_to_hartree;
                internal::log_iteration_energy(terms, ws, prob.ew_total,
                                               iter_log);
                ++iter_log;
            });

        const bool conv = drive_stage_loop_(max_iter, result_);
        if (conv) {
            if (log_rank0)
                ATV_INFO("[pw] converged iter={} energy_ry={:.8f} mag={}",
                         result_.iterations, result_.energy,
                         result_.magnetization);
        } else if (log_rank0) {
            ATV_WARN("[pw] not converged after {} iters energy_ry={:.8f}",
                     max_iter, result_.energy);
        }
    }

    return result_;
}

std::vector<std::array<double, 3>> pw_forces_analytic(const cpu::Frame& frame,
                                                      const PW& method,
                                                      double* energy_ry_out) {
    return internal::pw_forces_analytic_impl(frame, method, energy_ry_out);
}

std::array<double, 6> pw_stress(const cpu::Frame& frame, const PW& method,
                                double* energy_ry_out) {
    return internal::pw_stress_analytic_impl(frame, method, energy_ry_out);
}

std::array<double, 6> pw_stress_fd(const cpu::Frame& frame, const PW& method,
                                   double h) {
    return internal::pw_stress_fd_impl(frame, method, h);
}

std::vector<std::array<double, 3>>
pw_forces_fd(const cpu::Frame& frame, const PW& method, double h_ang) {
    if (h_ang <= 0.0)
        throw std::invalid_argument("pw_forces_fd: h_ang must be > 0");
    const int n = frame.nlocal();
    if (n <= 0)
        throw std::invalid_argument("pw_forces_fd: frame has no atoms");

    // Working copy: only positions are perturbed; type/mass/box stay fixed.
    cpu::Frame work = frame;

    auto x = work.span(kAtom_x);
    auto y = work.span(kAtom_y);
    auto z = work.span(kAtom_z);
    StorageReal* axes[3] = {x.data(), y.data(), z.data()};

    std::vector<std::array<double, 3>> forces(static_cast<std::size_t>(n),
                                              {0.0, 0.0, 0.0});
    // Energy is in Ry; convert dE/dR to eV/Å for the Frame force convention.
    const double ry_to_ev = qe_units::rydberg_to_ev;

    for (int a = 0; a < n; ++a) {
        for (int k = 0; k < 3; ++k) {
            const StorageReal r0 = axes[k][a];

            axes[k][a] = r0 + static_cast<StorageReal>(h_ang);
            PW plus = internal::copy_pw_params(method);
            plus.build(work);
            const double e_plus = plus.run(work).energy; // Ry

            axes[k][a] = r0 - static_cast<StorageReal>(h_ang);
            PW minus = internal::copy_pw_params(method);
            minus.build(work);
            const double e_minus = minus.run(work).energy;

            axes[k][a] = r0;
            // F = -dE/dR with E in eV, R in Å
            forces[static_cast<std::size_t>(a)][static_cast<std::size_t>(k)] =
                -(e_plus - e_minus) * ry_to_ev / (2.0 * h_ang);
        }
    }
    return forces;
}

} // namespace pw
} // namespace cpu
} // namespace ATV
