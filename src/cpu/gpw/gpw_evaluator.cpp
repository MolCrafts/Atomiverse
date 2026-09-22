/// @file src/cpu/gpw/gpw_evaluator.cpp
/// @brief Implementation of the GPW energy evaluator component (see header).

#include "atomiverse/cpu/gpw/gpw_evaluator.h"

#include <cmath>
#include <complex>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <vector>

#include <xtensor/containers/xtensor.hpp>
#include <xtensor/views/xview.hpp>

#include "atomiverse/core/frame_schema.h"
#include "atomiverse/core/units.h"
#include "atomiverse/cpu/gpw/internal/gpw_forces.h"
#include "atomiverse/cpu/gpw/internal/gpw_stress.h"
#include "atomiverse/cpu/gpw/internal/kpoints.h"
#include "atomiverse/cpu/linalg/density.h"
#include "atomiverse/cpu/linalg/eigensolve.h"
#include "atomiverse/cpu/linalg/matrix.h"

namespace ATV {
namespace cpu {
namespace gpw {

namespace {
const Kind& find_kind(const std::vector<Kind>& kinds, int z) {
    for (const auto& k : kinds)
        if (static_cast<int>(k.element) == z)
            return k;
    throw std::runtime_error(
        "GPWEvaluator: no basis/potential configured for Z="
        + std::to_string(z));
}

/// @brief Refuse a cell GPW cannot represent.
///
/// GPW carries the cell as three edge lengths and builds its lattice images
/// from them, so a sheared frame would be solved on `diag(h)` — a *different*
/// Bravais lattice. The Gamma-point Bloch sum would then run over the wrong
/// images and every lattice-sum quantity (S, T, V_pp, V_nl, rho) would be wrong
/// at O(shear), not O(shear^2), while `Box::volume()` still reported |det h|.
/// There is no regime in which that is an approximation of the requested
/// system, so it is refused rather than approximated.
///
/// @param box Frame cell [Angstrom].
/// @param who Call site name, for the message.
/// @throws std::invalid_argument if `box` has any non-zero off-diagonal entry.
void require_orthorhombic(const cpu::Box& box, const char* who) {
    if (box.is_orthorhombic())
        return;
    const Mat3& h = box.cell();
    std::string msg = std::string(who)
                      + ": GPW supports orthorhombic cells only, but the frame "
                        "cell is sheared (";
    bool first = true;
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j) {
            if (i == j || h(i, j) == 0.0)
                continue;
            if (!first)
                msg += ", ";
            first = false;
            msg += "h(" + std::to_string(i) + "," + std::to_string(j)
                   + ")=" + std::to_string(h(i, j));
        }
    msg += " [Angstrom]). Supply a diagonal cell matrix (e.g. "
           "cpu::Box::orthorhombic(a, b, c)), or use a method that handles a "
           "general cell.";
    throw std::invalid_argument(msg);
}
} // namespace

namespace {
/// Assemble the GpwAtom set at the frame's current geometry (Å → Bohr).
std::vector<GpwAtom> assemble_atoms(const cpu::Frame& frame,
                                    const GPWEvaluator& self) {
    const int n = frame.nlocal();
    const int* __restrict sp_type = frame.span(kAtom_type);
    const StorageReal* __restrict sp_x = frame.span(kAtom_x);
    const StorageReal* __restrict sp_y = frame.span(kAtom_y);
    const StorageReal* __restrict sp_z = frame.span(kAtom_z);
    std::vector<GpwAtom> atoms;
    atoms.reserve(static_cast<std::size_t>(n));
    for (int i = 0; i < n; ++i) {
        const Kind& k = find_kind(self.kinds, sp_type[i]);
        const double pos[3] = {sp_x[i] * units::kAngstromInBohr,
                               sp_y[i] * units::kAngstromInBohr,
                               sp_z[i] * units::kAngstromInBohr};
        atoms.push_back(GpwAtom(pos, k.basis, k.pot, sp_type[i]));
    }
    return atoms;
}

/// Publish the two CPMD electronic energies [eV] on the frame, so Observers,
/// Thermo and `cpu::md::CpmdConserved` see the adiabaticity signal that used
/// to be reachable only through `last_electronic_ke_ev()`. Called from every
/// CPMD branch (Γ and multi-k, RKS and collinear UKS) plus the cold seed,
/// which publishes zeros — the electronic degrees of freedom have not moved
/// yet.
/// The keys are open-schema (not pre-inserted by `Frame(int)`), so the first
/// launch inserts and later launches overwrite (the `kDipole` pattern).
/// @param frame Frame to write.
/// @param te_ev Fictitious electronic kinetic energy \(T_e\) [eV].
/// @param etherm_ev Electronic Nosé–Hoover thermostat energy [eV].
void publish_electronic_energies(cpu::Frame& frame, double te_ev,
                                 double etherm_ev) {
    if (!frame.meta.contains(kEnergyElectronKinetic.key))
        frame.meta.insert(kEnergyElectronKinetic, te_ev);
    else
        frame.meta.set(kEnergyElectronKinetic, te_ev);
    if (!frame.meta.contains(kEnergyElectronThermostat.key))
        frame.meta.insert(kEnergyElectronThermostat, etherm_ev);
    else
        frame.meta.set(kEnergyElectronThermostat, etherm_ev);
}

/// Complex Bloch density from occupied coefficients:
/// \(P = f\,C_{\mathrm{occ}} C_{\mathrm{occ}}^\dagger\) (occupancy f per
/// orbital: 2 for closed-shell RKS, 1 per spin channel). Shared by the
/// multi-k CPMD fixed-density step and the multi-k linearized-XL warm step.
/// @param C Bloch coefficients (nb × m, columns = orbitals). @param n_occ
/// Occupied column count (≤ m). @param occ Occupancy f (dimensionless).
/// @return Hermitian nb × nb density (dimensionless).
linalg::CplxMatrix density_from_cocc(const linalg::CplxMatrix& C, int n_occ,
                                     double occ) {
    using Cplx = std::complex<double>;
    const int nb = static_cast<int>(C.shape(0));
    linalg::CplxMatrix P = xt::zeros<Cplx>(
        {static_cast<std::size_t>(nb), static_cast<std::size_t>(nb)});
    for (int o = 0; o < n_occ; ++o)
        for (int mu = 0; mu < nb; ++mu) {
            const Cplx cmu = occ * C(mu, o);
            for (int nu = 0; nu < nb; ++nu)
                P(mu, nu) += cmu * std::conj(C(nu, o));
        }
    return P;
}

/// Row-major flat occupied block (the `GpwResult::C_occ_k` layout the k-path
/// force/stress assembly consumes).
/// @param C Bloch coefficients (nb × m). @param n_occ Occupied columns.
/// @return Flat nb × n_occ complex block, row-major.
std::vector<std::complex<double>> flatten_occ(const linalg::CplxMatrix& C,
                                              int n_occ) {
    const int nb = static_cast<int>(C.shape(0));
    std::vector<std::complex<double>> flat(static_cast<std::size_t>(nb)
                                           * static_cast<std::size_t>(n_occ));
    for (int mu = 0; mu < nb; ++mu)
        for (int o = 0; o < n_occ; ++o)
            flat[static_cast<std::size_t>(mu) * static_cast<std::size_t>(n_occ)
                 + static_cast<std::size_t>(o)] = C(mu, o);
    return flat;
}

/// Accumulate a Voigt stress [Hartree/Bohr³] onto the frame's `kStress`
/// [eV/Å³]. Shared by the converged-SCF stress block and the linearized-XL
/// shadow-density stress on both warm branches (Γ and multi-k).
/// @param frame Frame to write. @param sigma_ha Voigt σ [Hartree/Bohr³].
void add_stress_to_frame(cpu::Frame& frame,
                         const std::array<double, 6>& sigma_ha) {
    // 1 Ha/Bohr³ = kHartreeInEv / (kBohrInAngstrom)³ eV/Å³
    const double ha_bohr3_to_ev_a3 =
        ATV::units::kHartreeInEv
        / (ATV::units::kBohrInAngstrom * ATV::units::kBohrInAngstrom
           * ATV::units::kBohrInAngstrom);
    // kStress is the EXTENSIVE Voigt virial [eV] (see core/frame_schema.h): it
    // is `+=`-accumulated by every evaluator and MPI-summed over ranks, and
    // both of those are only defined for an extensive quantity. GPW computes
    // the intensive σ = −(1/Ω)∂E/∂ε, so the volume goes back in here — the
    // method-level Ha/Bohr³ return stays intensive.
    const double vol = frame.box.volume();
    auto& st = frame.meta.get(kStress);
    for (int k = 0; k < 6; ++k)
        st[static_cast<std::size_t>(k)] +=
            sigma_ha[static_cast<std::size_t>(k)] * ha_bohr3_to_ev_a3 * vol;
}
} // namespace

void GPWEvaluator::set_timestep(double dt_fs) {
    if (!(dt_fs > 0.0) || !std::isfinite(dt_fs))
        throw std::invalid_argument(
            "GPWEvaluator::set_timestep: nuclear timestep must be finite and "
            "> 0 (fs), got "
            + std::to_string(dt_fs));
    cpmd_dt_fs_ = dt_fs;
}

void GPWEvaluator::build(const cpu::BuildContext& ctx) {
    require_orthorhombic(ctx.frame.box, "GPWEvaluator::build");

    // Validate that every atom in the frame has a configured element, and
    // count valence electrons for the closed-shell occupation.
    const int n_build = ctx.frame.nlocal();
    const int* __restrict sp_type_build = ctx.frame.span(kAtom_type);
    double z_val_total = 0.0;
    for (int i = 0; i < n_build; ++i)
        z_val_total +=
            static_cast<double>(find_kind(kinds, sp_type_build[i]).pot.zion);

    // Assemble the solver from the build-time frame. Product path is the
    // `use_ot` member; env `ATV_GPW_OT` remains a diagnostic force-on.
    GpwDft cfg;
    cfg.cell[0] = ctx.frame.box.length(0) * units::kAngstromInBohr;
    cfg.cell[1] = ctx.frame.box.length(1) * units::kAngstromInBohr;
    cfg.cell[2] = ctx.frame.box.length(2) * units::kAngstromInBohr;
    cfg.cutoff_ry = cutoff_ry;
    cfg.ngrids = ngrids;
    cfg.rel_cutoff = rel_cutoff;
    cfg.n_alpha = n_alpha;
    cfg.n_beta = n_beta;
    if (n_alpha + n_beta > 0) {
        // UKS: occupations are explicit; n_occ unused.
        cfg.n_occ = 0;
    } else {
        cfg.n_occ = static_cast<int>(std::lround(z_val_total / 2.0));
    }
    cfg.max_iter = max_iter;
    cfg.tol = tol;
    cfg.mix = mix;
    cfg.xc = xc;
    cfg.nonperiodic = nonperiodic;
    cfg.d3 = d3;
    cfg.d3bj = d3bj;
    cfg.d3_atm = d3_atm;
    cfg.d3_rcut_bohr = d3_rcut_bohr;
    cfg.d4 = d4;
    cfg.d4_atm = d4_atm;
    cfg.ts = ts;
    cfg.xdm = xdm;
    cfg.mbd = mbd;
    for (int d = 0; d < 3; ++d)
        cfg.mbd_k_mesh[d] = mbd_k_mesh[d];
    cfg.vdw_df = vdw_df;
    cfg.vdw_df2 = vdw_df2;
    cfg.rvv10 = rvv10;
    cfg.hirshfeld = hirshfeld;
    cfg.use_ot = use_ot || (std::getenv("ATV_GPW_OT") != nullptr);
    cfg.plus_u = plus_u;
    cfg.atom_spin_guess = atom_spin_guess;
    for (int d = 0; d < 3; ++d)
        cfg.k_mesh[d] = k_mesh[d];
    cfg.k_time_reversal = k_time_reversal;
    cfg.smearing = smearing;
    cfg.degauss = degauss;
    cfg.mp_order = mp_order;
    cfg.extra_bands = extra_bands;

    const bool multi_k = (k_mesh[0] != 1 || k_mesh[1] != 1 || k_mesh[2] != 1);

    // XL-BOMD guards (no-fallback): auxiliary-density guess on the
    // diagonalization path only.
    //   Γ RKS: one total-P real propagator
    //   Γ UKS: dual α/β real propagators
    //   multi-k RKS: one complex propagator per k (seeded on first SCF)
    //   multi-k UKS: dual complex propagators per k (P_k^α, P_k^β)
    if (xl_bomd_linearized && !xl_bomd)
        throw std::invalid_argument(
            "GPWEvaluator: xl_bomd_linearized requires xl_bomd=true "
            "(zero-SCF shadow forces ride on the XL auxiliary density)");
    if (xl_bomd) {
        if (cfg.use_ot)
            throw std::invalid_argument(
                "GPWEvaluator: xl_bomd requires the diagonalization SCF path "
                "(use_ot parameterizes orbitals, not the density)");
        if (cpmd)
            throw std::invalid_argument(
                "GPWEvaluator: xl_bomd and cpmd are mutually exclusive "
                "(XL is SCF-converged auxiliary density; CP is fictitious "
                "orbital dynamics)");
        if (second_gen_cp)
            throw std::invalid_argument(
                "GPWEvaluator: xl_bomd and second_gen_cp are mutually "
                "exclusive (XL = Niklasson aux density; SGCP = ASPC + "
                "partial SCF corrector)");
        if (xl_bomd_linearized) {
            // Zero-SCF linearized forces on the diagonalization path:
            // Γ RKS (total P), Γ UKS (dual P_α/P_β), multi-k RKS (complex
            // P_k per k), multi-k UKS (dual P_k^α/P_k^β per k).
            if (!compute_forces)
                throw std::invalid_argument(
                    "GPWEvaluator: xl_bomd_linearized requires "
                    "compute_forces=true (shadow forces are the product)");
            // compute_stress is product: each warm branch evaluates the
            // analytic fixed-P stress at the shadow payload (P = D = q[n],
            // W from the H[n] eigenpairs) — zero-SCF preserved. Only the
            // multi-SCF strain-FD oracle is refused: it re-converges the
            // SCF at every strained cell, which is the Born–Oppenheimer
            // stress, not the shadow functional.
            if (compute_stress && stress_use_fd)
                throw std::invalid_argument(
                    "GPWEvaluator: xl_bomd_linearized refuses stress_use_fd "
                    "(the multi-SCF strain-FD oracle re-converges the SCF "
                    "under strain — Born–Oppenheimer stress, not the "
                    "zero-SCF shadow functional; use the analytic fixed-P "
                    "path, stress_use_fd=false)");
        }
        // Validates xl_bomd_k ∈ [3, 9] via the real constructor even when
        // multi-k uses XlBomdCplx (same Table I).
        if (multi_k) {
            xl_.reset();
            xl_beta_.reset();
            xl_k_.clear(); // sized after the first multi-k SCF (n_kpts known)
            xl_k_beta_.clear();
            (void)electronic::XlBomd(xl_bomd_k); // throw early on bad K
        } else {
            xl_.emplace(xl_bomd_k);
            if (n_alpha + n_beta > 0)
                xl_beta_.emplace(xl_bomd_k);
            else
                xl_beta_.reset();
            xl_k_.clear();
            xl_k_beta_.clear();
        }
    } else {
        xl_.reset();
        xl_beta_.reset();
        xl_k_.clear();
        xl_k_beta_.clear();
    }

    // Car–Parrinello guards: RKS Γ, collinear Γ-UKS, multi-k RKS, or
    // multi-k collinear UKS (dual-spin Bloch C_k^σ).
    if (cpmd) {
        if (cfg.use_ot)
            throw std::invalid_argument(
                "GPWEvaluator: cpmd requires the diagonalization SCF path for "
                "the cold seed (use_ot is refused — no silent OT-CP hybrid)");
        if (second_gen_cp)
            throw std::invalid_argument(
                "GPWEvaluator: cpmd and second_gen_cp are mutually exclusive "
                "(CPMD = fictitious mass; SGCP = ASPC + partial SCF)");
        if (!(cpmd_mu > 0.0) || !std::isfinite(cpmd_mu))
            throw std::invalid_argument(
                "GPWEvaluator: cpmd_mu must be finite and > 0 (a.u. mass)");
        // The electronic step rides the nuclear clock, so the dt is PUSHED by
        // the MD integrator (`cpu::TimestepProvider` → `set_timestep`), never
        // re-declared here. No push ⇒ either there is no MD loop around this
        // evaluator, or its integrator publishes nothing: both are broken CP.
        if (!(cpmd_dt_fs_ > 0.0))
            throw std::runtime_error(
                "GPWEvaluator: cpmd requires an MD integrator publishing the "
                "timestep (TimestepProvider); CPMD outside MD is meaningless");
        if (!compute_forces)
            throw std::invalid_argument(
                "GPWEvaluator: cpmd requires compute_forces=true (nuclear "
                "forces from the fixed-density electronic state)");
        // Every CPMD branch of `launch` returns before the stress block, so a
        // stress request would be silently dropped rather than refused.
        if (compute_stress)
            throw std::runtime_error(
                "GPWEvaluator: CPMD stress path is not productized; "
                "variable-cell CP is deferred");
        // UKS CPMD: alpha is the primary channel (mirrors XL dual setup).
        if (n_alpha + n_beta > 0 && n_alpha < 1)
            throw std::invalid_argument(
                "GPWEvaluator: cpmd UKS requires n_alpha >= 1");
        if (cpmd_thermostat) {
            if (!(cpmd_ekin_target_ha > 0.0)
                || !std::isfinite(cpmd_ekin_target_ha))
                throw std::invalid_argument(
                    "GPWEvaluator: cpmd_thermostat requires "
                    "cpmd_ekin_target_ha finite and > 0 (Ha)");
            if (!(cpmd_tau_e > 0.0) || !std::isfinite(cpmd_tau_e))
                throw std::invalid_argument(
                    "GPWEvaluator: cpmd_thermostat requires cpmd_tau_e "
                    "finite and > 0 (fs)");
        }
        if (multi_k) {
            // Complex propagators sized after the first multi-k SCF (n_kpts):
            // RKS one per k; collinear UKS one per (k, spin).
            cpmd_.reset();
            cpmd_beta_.reset();
            cpmd_k_.clear();
            cpmd_k_beta_.clear();
            (void)electronic::CpmdCplx(cpmd_mu); // early mu validation
        } else {
            cpmd_.emplace(cpmd_mu);
            cpmd_->thermostat = cpmd_thermostat;
            cpmd_->ekin_target_ha = cpmd_ekin_target_ha;
            cpmd_->tau_e_fs = cpmd_tau_e;
            if (n_alpha + n_beta > 0 && n_beta > 0) {
                cpmd_beta_.emplace(cpmd_mu);
                cpmd_beta_->thermostat = cpmd_thermostat;
                cpmd_beta_->ekin_target_ha = cpmd_ekin_target_ha;
                cpmd_beta_->tau_e_fs = cpmd_tau_e;
            } else {
                cpmd_beta_.reset();
            }
            cpmd_k_.clear();
            cpmd_k_beta_.clear();
        }
        last_te_ev_ = 0.0;
        last_etherm_ev_ = 0.0;
    } else {
        if (cpmd_thermostat)
            throw std::invalid_argument(
                "GPWEvaluator: cpmd_thermostat requires cpmd=true");
        cpmd_.reset();
        cpmd_beta_.reset();
        cpmd_k_.clear();
        cpmd_k_beta_.clear();
        last_te_ev_ = 0.0;
        last_etherm_ev_ = 0.0;
    }

    // Second-generation CP (Kühne PRL 2007): ASPC + partial SCF corrector.
    // Γ RKS: one total-P real ASPC (DIIS or OT corrector).
    // Γ UKS: dual α/β real ASPC (DIIS or OT).
    // multi-k RKS: one complex ASPC per k (DIIS or OT — the k-OT manifold
    //   warm-starts from natural orbitals of the predicted P_k).
    // multi-k UKS: dual complex ASPC per (k, σ) (DIIS or OT — dual-spin
    //   per-k OT manifolds warm-start from natural orbitals of the
    //   predicted P_k^σ).
    if (second_gen_cp) {
        if (!(second_gen_cp_k >= 0))
            throw std::invalid_argument(
                "GPWEvaluator: second_gen_cp_k must be >= 0 (ASPC order K)");
        if (!(second_gen_cp_max_iter >= 1))
            throw std::invalid_argument(
                "GPWEvaluator: second_gen_cp_max_iter must be >= 1 "
                "(corrector SCF budget)");
        if (!compute_forces)
            throw std::invalid_argument(
                "GPWEvaluator: second_gen_cp requires compute_forces=true "
                "(forces at the post-corrector density)");
        if (multi_k) {
            // Validate K early; k-histories sized after the first multi-k SCF
            // (n_kpts known).
            (void)electronic::ASPCPredictorCplx(second_gen_cp_k);
            sgcp_aspc_.reset();
            sgcp_aspc_beta_.reset();
            sgcp_aspc_k_.clear();
            sgcp_aspc_k_beta_.clear();
        } else {
            sgcp_aspc_.emplace(second_gen_cp_k);
            if (n_alpha + n_beta > 0)
                sgcp_aspc_beta_.emplace(second_gen_cp_k);
            else
                sgcp_aspc_beta_.reset();
            sgcp_aspc_k_.clear();
            sgcp_aspc_k_beta_.clear();
        }
    } else {
        sgcp_aspc_.reset();
        sgcp_aspc_beta_.reset();
        sgcp_aspc_k_.clear();
        sgcp_aspc_k_beta_.clear();
    }

    solver_ = std::make_unique<GpwDft>(std::move(cfg));
    solver_->build(assemble_atoms(ctx.frame, *this));
}

void GPWEvaluator::launch(const cpu::LaunchContext& ctx) {
    cpu::Frame& frame = ctx.frame;
    const int n = frame.nlocal();

    // Re-checked every launch, not only at build: this is the variable-cell
    // path, and TriclinicCellRelax and ParrinelloRahmanNPT both write a general
    // cell matrix back to the Frame between launches. A build-time-only guard
    // would let those silently reduce the cell to its diagonal.
    require_orthorhombic(frame.box, "GPWEvaluator::launch");

    // Variable cell: re-target the solver to the frame's current orthorhombic
    // box. GpwDft::run rebuilds every geometry-dependent structure (grids,
    // lattice images, S/T/Vpp) from `cell` per call, so a box change between
    // launches is well-defined — cell-relax drivers rescale box freely.
    for (int d = 0; d < 3; ++d) {
        const double now = frame.box.length(d) * units::kAngstromInBohr;
        if (!(now > 0.0))
            throw std::runtime_error(
                "GPWEvaluator::launch: non-positive box edge");
        solver_->cell[d] = now;
    }

    const auto atoms = assemble_atoms(frame, *this);

    // ── Car–Parrinello path ──────────────────────────────────────────────
    // Seed: one full SCF + BO analytic forces, then hand occupied C to Cpmd.
    // Later steps: fixed-density KS eval (max_iter=0) + forces at that
    // density + electronic Verlet. Nuclear PE is E_KS only; T_e is published
    // on frame.meta[kEnergyElectronKinetic] (and last_electronic_ke_ev())
    // for cpu::md::CpmdConserved to assemble the CP conserved energy.
    // RKS Γ: one real propagator (occ=2). UKS Γ: dual α/β (occ=1 each).
    // Multi-k RKS: one CpmdCplx per sampled k (occ=2). Multi-k UKS: one
    // CpmdCplx per (k, spin), occ=1 each (β set empty when n_beta == 0).
    const bool cpmd_uks_k = (n_alpha + n_beta > 0);
    const bool cpmd_multi_k = !cpmd_k_.empty() && cpmd_k_[0].ready() && [&]() {
        for (const auto& ch : cpmd_k_)
            if (!ch.ready())
                return false;
        if (cpmd_uks_k && n_beta > 0) {
            if (cpmd_k_beta_.size() != cpmd_k_.size())
                return false;
            for (const auto& ch : cpmd_k_beta_)
                if (!ch.ready())
                    return false;
        }
        return true;
    }();
    const bool cpmd_uks = cpmd_ && (n_alpha + n_beta > 0);
    const bool cpmd_ready =
        cpmd_multi_k
        || (cpmd_
            && (cpmd_uks
                    ? (cpmd_->ready()
                       && (n_beta < 1 || (cpmd_beta_ && cpmd_beta_->ready())))
                    : cpmd_->ready()));
    if (cpmd_ready && cpmd_multi_k) {
        // Multi-k CPMD: fixed-density {P_k} (RKS, occ 2) or {P_k^α, P_k^β}
        // (collinear UKS, occ 1 per spin) → forces → per-(k, σ) electronic VV.
        using Cplx = std::complex<double>;
        const int nk = static_cast<int>(cpmd_k_.size());
        const bool have_beta_k = cpmd_uks_k && !cpmd_k_beta_.empty();
        auto fixed_density_eval_k = [&]() {
            cpmd_guess_p_k_.resize(static_cast<std::size_t>(nk));
            if (cpmd_uks_k)
                cpmd_guess_p_k_beta_.resize(static_cast<std::size_t>(nk));
            for (int ik = 0; ik < nk; ++ik) {
                const auto& Ca = cpmd_k_[static_cast<std::size_t>(ik)].C();
                if (cpmd_uks_k) {
                    cpmd_guess_p_k_[static_cast<std::size_t>(ik)] =
                        density_from_cocc(Ca, n_alpha, /*occ=*/1.0);
                    // Half-metal edge (n_beta == 0): zero β density.
                    cpmd_guess_p_k_beta_[static_cast<std::size_t>(ik)] =
                        have_beta_k
                            ? density_from_cocc(
                                  cpmd_k_beta_[static_cast<std::size_t>(ik)]
                                      .C(),
                                  n_beta, /*occ=*/1.0)
                            : linalg::CplxMatrix(
                                  xt::zeros<Cplx>({Ca.shape(0), Ca.shape(0)}));
                } else {
                    cpmd_guess_p_k_[static_cast<std::size_t>(ik)] =
                        density_from_cocc(Ca, solver_->n_occ, /*occ=*/2.0);
                }
            }
            const int saved_max = solver_->max_iter;
            solver_->max_iter = 0;
            try {
                last_ = solver_->run(
                    atoms, /*guess_C=*/nullptr, /*guess_P=*/nullptr,
                    /*guess_P_beta=*/nullptr, &cpmd_guess_p_k_,
                    cpmd_uks_k ? &cpmd_guess_p_k_beta_ : nullptr);
            } catch (...) {
                solver_->max_iter = saved_max;
                throw;
            }
            solver_->max_iter = saved_max;
        };

        // S_k is geometry-only: build it standalone, align the propagated
        // coefficients against it, then evaluate the KS matrices ONCE at the
        // aligned density. `GpwDft::overlap_k` runs the same phased assembly
        // the solve does, so these are the very S_k the eval would report.
        const std::vector<linalg::CplxMatrix> S_k = solver_->overlap_k(atoms);
        if (static_cast<int>(S_k.size()) != nk)
            throw std::runtime_error(
                "GPWEvaluator: multi-k CPMD standalone overlap_k size != nk");
        for (int ik = 0; ik < nk; ++ik) {
            cpmd_k_[static_cast<std::size_t>(ik)].align_to_overlap(
                S_k[static_cast<std::size_t>(ik)]);
            if (have_beta_k)
                cpmd_k_beta_[static_cast<std::size_t>(ik)].align_to_overlap(
                    S_k[static_cast<std::size_t>(ik)]);
        }
        fixed_density_eval_k();
        if (static_cast<int>(last_.H_ks_k.size()) != nk)
            throw std::runtime_error(
                "GPWEvaluator: multi-k CPMD fixed-density missing H_ks_k");
        if (cpmd_uks_k && static_cast<int>(last_.H_ks_k_beta.size()) != nk)
            throw std::runtime_error(
                "GPWEvaluator: multi-k UKS CPMD fixed-density missing "
                "H_ks_k_beta");

        frame.meta.get(kEnergy) += last_.e_total * ATV::units::kHartreeInEv;
        frame.meta.get(kEnergyElectronic) +=
            last_.e_total * ATV::units::kHartreeInEv;

        if (compute_forces) {
            // Populate C_occ_k(_beta) + eps_k(_beta) = diag(C† H^σ C) for
            // the k-summed force assembly (occ 2 RKS; occ 1 per spin UKS).
            auto fill_channel = [&](const linalg::CplxMatrix& C,
                                    const linalg::CplxMatrix& H, int n_occ,
                                    std::vector<Cplx>& flat_out,
                                    std::vector<double>& eps_out) {
                const int nb = static_cast<int>(C.shape(0));
                flat_out = flatten_occ(C, n_occ);
                // ε_i ≈ Re (C† H C)_ii  (BO surface → exact eigenvalues).
                std::vector<double> eps(static_cast<std::size_t>(n_occ), 0.0);
                for (int o = 0; o < n_occ; ++o) {
                    Cplx s(0, 0);
                    for (int mu = 0; mu < nb; ++mu) {
                        Cplx hco(0, 0);
                        for (int nu = 0; nu < nb; ++nu)
                            hco += H(mu, nu) * C(nu, o);
                        s += std::conj(C(mu, o)) * hco;
                    }
                    eps[static_cast<std::size_t>(o)] = s.real();
                }
                eps_out = std::move(eps);
            };
            last_.C_occ_k.resize(static_cast<std::size_t>(nk));
            last_.eps_k.resize(static_cast<std::size_t>(nk));
            if (cpmd_uks_k) {
                last_.C_occ_k_beta.resize(static_cast<std::size_t>(nk));
                last_.eps_k_beta.resize(static_cast<std::size_t>(nk));
            }
            for (int ik = 0; ik < nk; ++ik) {
                fill_channel(cpmd_k_[static_cast<std::size_t>(ik)].C(),
                             last_.H_ks_k[static_cast<std::size_t>(ik)],
                             cpmd_uks_k ? n_alpha : solver_->n_occ,
                             last_.C_occ_k[static_cast<std::size_t>(ik)],
                             last_.eps_k[static_cast<std::size_t>(ik)]);
                if (cpmd_uks_k) {
                    if (have_beta_k) {
                        fill_channel(
                            cpmd_k_beta_[static_cast<std::size_t>(ik)].C(),
                            last_.H_ks_k_beta[static_cast<std::size_t>(ik)],
                            n_beta,
                            last_.C_occ_k_beta[static_cast<std::size_t>(ik)],
                            last_.eps_k_beta[static_cast<std::size_t>(ik)]);
                    } else {
                        last_.C_occ_k_beta[static_cast<std::size_t>(ik)]
                            .clear();
                        last_.eps_k_beta[static_cast<std::size_t>(ik)].clear();
                    }
                }
            }
            last_.converged = true;
            const auto F = gpw_forces_from_scf(atoms, *solver_, last_);
            AccReal* __restrict sp_fx = frame.span(kAtom_fx);
            AccReal* __restrict sp_fy = frame.span(kAtom_fy);
            AccReal* __restrict sp_fz = frame.span(kAtom_fz);
            for (int i = 0; i < n; ++i) {
                sp_fx[i] += F[static_cast<std::size_t>(i)][0]
                            * ATV::units::kHartreePerBohrInEvPerAngstrom;
                sp_fy[i] += F[static_cast<std::size_t>(i)][1]
                            * ATV::units::kHartreePerBohrInEvPerAngstrom;
                sp_fz[i] += F[static_cast<std::size_t>(i)][2]
                            * ATV::units::kHartreePerBohrInEvPerAngstrom;
            }
        }

        // Electronic steps: T_e = Σ_k w_k Σ_σ T_e^{(k,σ)}. With the
        // electronic Nose-Hoover engaged, each (k, σ) channel carries its
        // own NH variable and E_therm = Σ_k w_k Σ_σ E_therm^{(k,σ)} — the
        // same w_k the Lagrangian gives the fictitious KE and the band
        // energy, so the extended conserved energy stays consistent.
        double te_ha = 0.0;
        double eth_ha = 0.0;
        for (int ik = 0; ik < nk; ++ik) {
            const double wk =
                (ik < static_cast<int>(last_.k_weights.size()))
                    ? last_.k_weights[static_cast<std::size_t>(ik)]
                    : (1.0 / static_cast<double>(nk));
            cpmd_k_[static_cast<std::size_t>(ik)].advance(
                last_.H_ks_k[static_cast<std::size_t>(ik)],
                S_k[static_cast<std::size_t>(ik)], cpmd_dt_fs_,
                /*occ=*/cpmd_uks_k ? 1.0 : 2.0);
            double te_k = cpmd_k_[static_cast<std::size_t>(ik)].kinetic_ha();
            double eth_k =
                cpmd_k_[static_cast<std::size_t>(ik)].thermostat_energy_ha();
            if (have_beta_k) {
                cpmd_k_beta_[static_cast<std::size_t>(ik)].advance(
                    last_.H_ks_k_beta[static_cast<std::size_t>(ik)],
                    S_k[static_cast<std::size_t>(ik)], cpmd_dt_fs_,
                    /*occ=*/1.0);
                te_k += cpmd_k_beta_[static_cast<std::size_t>(ik)].kinetic_ha();
                eth_k += cpmd_k_beta_[static_cast<std::size_t>(ik)]
                             .thermostat_energy_ha();
            }
            te_ha += wk * te_k;
            eth_ha += wk * eth_k;
        }
        last_te_ev_ = te_ha * ATV::units::kHartreeInEv;
        last_etherm_ev_ = eth_ha * ATV::units::kHartreeInEv;
        publish_electronic_energies(frame, last_te_ev_, last_etherm_ev_);
        return;
    }
    if (cpmd_ready) {
        auto fixed_density_eval_rks = [&](const linalg::RealMatrix& C_occ) {
            const int nb = static_cast<int>(C_occ.shape(0));
            const int n_occ = static_cast<int>(C_occ.shape(1));
            const linalg::RealMatrix P =
                linalg::build_density_matrix(C_occ, n_occ, nb, 2.0);
            const int saved_max = solver_->max_iter;
            solver_->max_iter = 0;
            try {
                last_ = solver_->run(atoms, /*guess_C=*/nullptr, &P);
            } catch (...) {
                solver_->max_iter = saved_max;
                throw;
            }
            solver_->max_iter = saved_max;
        };
        auto fixed_density_eval_uks = [&](const linalg::RealMatrix& Ca,
                                          const linalg::RealMatrix& Cb) {
            const int nb =
                static_cast<int>((Ca.size() > 0 ? Ca.shape(0) : Cb.shape(0)));
            const int na = Ca.size() > 0 ? static_cast<int>(Ca.shape(1)) : 0;
            const int nb_spin =
                Cb.size() > 0 ? static_cast<int>(Cb.shape(1)) : 0;
            const linalg::RealMatrix Pa =
                na > 0 ? linalg::build_density_matrix(Ca, na, nb, 1.0)
                       : linalg::RealMatrix(
                             xt::zeros<double>({static_cast<std::size_t>(nb),
                                                static_cast<std::size_t>(nb)}));
            const linalg::RealMatrix Pb =
                nb_spin > 0 ? linalg::build_density_matrix(Cb, nb_spin, nb, 1.0)
                            : linalg::RealMatrix(xt::zeros<double>(
                                  {static_cast<std::size_t>(nb),
                                   static_cast<std::size_t>(nb)}));
            const int saved_max = solver_->max_iter;
            solver_->max_iter = 0;
            try {
                last_ = solver_->run(atoms, /*guess_C=*/nullptr, &Pa, &Pb);
            } catch (...) {
                solver_->max_iter = saved_max;
                throw;
            }
            solver_->max_iter = saved_max;
        };

        // S is geometry-only: build it standalone, align, then evaluate the
        // fixed-density KS matrix ONCE at the aligned coefficients.
        // `GpwDft::overlap` runs the same Γ assembly the solve does.
        const linalg::RealMatrix S = solver_->overlap(atoms);

        if (cpmd_uks) {
            // Empty beta C when n_beta==0 (half-metal edge).
            const linalg::RealMatrix empty_c;
            cpmd_->align_to_overlap(S);
            if (cpmd_beta_ && cpmd_beta_->ready())
                cpmd_beta_->align_to_overlap(S);
            const linalg::RealMatrix& Ca2 = cpmd_->C();
            const linalg::RealMatrix& Cb2 =
                (cpmd_beta_ && cpmd_beta_->ready()) ? cpmd_beta_->C() : empty_c;
            fixed_density_eval_uks(Ca2, Cb2);

            frame.meta.get(kEnergy) += last_.e_total * ATV::units::kHartreeInEv;
            frame.meta.get(kEnergyElectronic) +=
                last_.e_total * ATV::units::kHartreeInEv;

            if (compute_forces) {
                if (last_.H_ks.size() == 0 || last_.H_ks_beta.size() == 0)
                    throw std::runtime_error(
                        "GPWEvaluator: UKS CPMD force path needs H_ks and "
                        "H_ks_beta from fixed-density eval");
                const auto F = gpw_forces_at_density(
                    atoms, *solver_, last_.P, Ca2, last_.H_ks, last_.P_beta,
                    Cb2, last_.H_ks_beta);
                AccReal* __restrict sp_fx = frame.span(kAtom_fx);
                AccReal* __restrict sp_fy = frame.span(kAtom_fy);
                AccReal* __restrict sp_fz = frame.span(kAtom_fz);
                for (int i = 0; i < n; ++i) {
                    sp_fx[i] += F[static_cast<std::size_t>(i)][0]
                                * ATV::units::kHartreePerBohrInEvPerAngstrom;
                    sp_fy[i] += F[static_cast<std::size_t>(i)][1]
                                * ATV::units::kHartreePerBohrInEvPerAngstrom;
                    sp_fz[i] += F[static_cast<std::size_t>(i)][2]
                                * ATV::units::kHartreePerBohrInEvPerAngstrom;
                }
            }

            // Dual electronic steps (occ=1 per spin).
            double te_ha = 0.0;
            double eth_ha = 0.0;
            cpmd_->advance(last_.H_ks, S, cpmd_dt_fs_, /*occ=*/1.0);
            te_ha += cpmd_->kinetic_ha();
            eth_ha += cpmd_->thermostat_energy_ha();
            if (cpmd_beta_ && cpmd_beta_->ready()) {
                cpmd_beta_->advance(last_.H_ks_beta, S, cpmd_dt_fs_,
                                    /*occ=*/1.0);
                te_ha += cpmd_beta_->kinetic_ha();
                eth_ha += cpmd_beta_->thermostat_energy_ha();
            }
            last_te_ev_ = te_ha * ATV::units::kHartreeInEv;
            last_etherm_ev_ = eth_ha * ATV::units::kHartreeInEv;
            publish_electronic_energies(frame, last_te_ev_, last_etherm_ev_);
            return;
        }

        // Closed-shell RKS CPMD.
        cpmd_->align_to_overlap(S);
        fixed_density_eval_rks(cpmd_->C());

        frame.meta.get(kEnergy) += last_.e_total * ATV::units::kHartreeInEv;
        frame.meta.get(kEnergyElectronic) +=
            last_.e_total * ATV::units::kHartreeInEv;

        if (compute_forces) {
            const auto F = gpw_forces_at_density(atoms, *solver_, last_.P,
                                                 cpmd_->C(), last_.H_ks);
            AccReal* __restrict sp_fx = frame.span(kAtom_fx);
            AccReal* __restrict sp_fy = frame.span(kAtom_fy);
            AccReal* __restrict sp_fz = frame.span(kAtom_fz);
            for (int i = 0; i < n; ++i) {
                sp_fx[i] += F[static_cast<std::size_t>(i)][0]
                            * ATV::units::kHartreePerBohrInEvPerAngstrom;
                sp_fy[i] += F[static_cast<std::size_t>(i)][1]
                            * ATV::units::kHartreePerBohrInEvPerAngstrom;
                sp_fz[i] += F[static_cast<std::size_t>(i)][2]
                            * ATV::units::kHartreePerBohrInEvPerAngstrom;
            }
        }

        cpmd_->advance(last_.H_ks, S, cpmd_dt_fs_, /*occ=*/2.0);
        last_te_ev_ = cpmd_->kinetic_ha() * ATV::units::kHartreeInEv;
        last_etherm_ev_ =
            cpmd_->thermostat_energy_ha() * ATV::units::kHartreeInEv;
        publish_electronic_energies(frame, last_te_ev_, last_etherm_ev_);
        return;
    }

    // ── Second-generation Car–Parrinello (ASPC + partial SCF) ───────────
    // Cold steps (insufficient history): full max_iter SCF, record P.
    // Warm steps: ASPC predict → SCF with corrector budget → forces at that
    // density → Kolafa correct + record (Kühne PRL 98, 066401 (2007)).
    // Γ RKS: total real P. Γ UKS: dual real P_α / P_β.
    // multi-k RKS: complex P_k per k. multi-k UKS: dual P_k^α / P_k^β.
    if (second_gen_cp) {
        const bool multi_k_sgcp =
            (k_mesh[0] != 1 || k_mesh[1] != 1 || k_mesh[2] != 1);
        const bool uks_sgcp = (n_alpha + n_beta > 0);

        // ── multi-k complex ASPC on P_k ────────────────────────────────
        if (multi_k_sgcp) {
            const std::vector<linalg::CplxMatrix>* guess_p_k = nullptr;
            const std::vector<linalg::CplxMatrix>* guess_p_k_beta = nullptr;
            std::vector<linalg::CplxMatrix> P_pred_k;
            std::vector<linalg::CplxMatrix> P_pred_k_beta;
            bool have_pred = false;
            if (!sgcp_aspc_k_.empty()
                && static_cast<int>(sgcp_aspc_k_.size()) > 0
                && sgcp_aspc_k_[0].history_size() >= second_gen_cp_k + 1) {
                // All k share the MD step count; require every channel warm.
                have_pred = true;
                P_pred_k.resize(sgcp_aspc_k_.size());
                for (std::size_t ik = 0; ik < sgcp_aspc_k_.size(); ++ik) {
                    if (!sgcp_aspc_k_[ik].predict(P_pred_k[ik])) {
                        have_pred = false;
                        break;
                    }
                }
                if (have_pred && uks_sgcp) {
                    if (sgcp_aspc_k_beta_.size() != sgcp_aspc_k_.size())
                        have_pred = false;
                    else {
                        P_pred_k_beta.resize(sgcp_aspc_k_beta_.size());
                        for (std::size_t ik = 0; ik < sgcp_aspc_k_beta_.size();
                             ++ik) {
                            if (!sgcp_aspc_k_beta_[ik].predict(
                                    P_pred_k_beta[ik])) {
                                have_pred = false;
                                break;
                            }
                        }
                    }
                }
                if (have_pred) {
                    sgcp_guess_p_k_ = P_pred_k;
                    guess_p_k = &sgcp_guess_p_k_;
                    if (uks_sgcp) {
                        sgcp_guess_p_k_beta_ = P_pred_k_beta;
                        guess_p_k_beta = &sgcp_guess_p_k_beta_;
                    }
                }
            }

            const int saved_max = solver_->max_iter;
            if (have_pred)
                solver_->max_iter = second_gen_cp_max_iter;
            try {
                last_ = solver_->run(atoms, /*guess_C=*/nullptr,
                                     /*guess_P=*/nullptr,
                                     /*guess_P_beta=*/nullptr, guess_p_k,
                                     guess_p_k_beta);
            } catch (...) {
                solver_->max_iter = saved_max;
                throw;
            }
            solver_->max_iter = saved_max;

            auto reseat_k =
                [&](std::vector<electronic::ASPCPredictorCplx>& xs) {
                    if (static_cast<int>(xs.size()) != last_.n_kpts) {
                        xs.clear();
                        xs.reserve(static_cast<std::size_t>(last_.n_kpts));
                        for (int ik = 0; ik < last_.n_kpts; ++ik)
                            xs.emplace_back(second_gen_cp_k);
                    }
                };

            const bool have_pk =
                last_.n_kpts > 0
                && static_cast<int>(last_.P_k.size()) == last_.n_kpts;
            const bool have_pk_b =
                !uks_sgcp
                || (have_pk
                    && static_cast<int>(last_.P_k_beta.size()) == last_.n_kpts);
            if (have_pk && have_pk_b) {
                reseat_k(sgcp_aspc_k_);
                for (int ik = 0; ik < last_.n_kpts; ++ik) {
                    const auto& Pk = last_.P_k[static_cast<std::size_t>(ik)];
                    auto& aspc = sgcp_aspc_k_[static_cast<std::size_t>(ik)];
                    if (have_pred
                        && static_cast<int>(P_pred_k.size()) == last_.n_kpts) {
                        linalg::CplxMatrix P_corr;
                        aspc.correct(Pk, P_pred_k[static_cast<std::size_t>(ik)],
                                     P_corr);
                        aspc.record(P_corr);
                    } else {
                        aspc.record(Pk);
                    }
                }
                if (uks_sgcp) {
                    reseat_k(sgcp_aspc_k_beta_);
                    for (int ik = 0; ik < last_.n_kpts; ++ik) {
                        const auto& Pk =
                            last_.P_k_beta[static_cast<std::size_t>(ik)];
                        auto& aspc =
                            sgcp_aspc_k_beta_[static_cast<std::size_t>(ik)];
                        if (have_pred
                            && static_cast<int>(P_pred_k_beta.size())
                                   == last_.n_kpts) {
                            linalg::CplxMatrix P_corr;
                            aspc.correct(
                                Pk, P_pred_k_beta[static_cast<std::size_t>(ik)],
                                P_corr);
                            aspc.record(P_corr);
                        } else {
                            aspc.record(Pk);
                        }
                    }
                } else {
                    sgcp_aspc_k_beta_.clear();
                }
            } else {
                for (auto& a : sgcp_aspc_k_)
                    a.reset();
                for (auto& a : sgcp_aspc_k_beta_)
                    a.reset();
            }

            frame.meta.get(kEnergy) += last_.e_total * ATV::units::kHartreeInEv;
            frame.meta.get(kEnergyElectronic) +=
                last_.e_total * ATV::units::kHartreeInEv;

            if (compute_forces) {
                // Multi-k forces at the post-corrector density. Partial SCF
                // may leave converged=false; SGCP semantics are forces at
                // the corrector state (same as Γ-UKS SGCP).
                if (!(last_.n_kpts > 0
                      && static_cast<int>(last_.C_occ_k.size()) == last_.n_kpts
                      && static_cast<int>(last_.eps_k.size()) == last_.n_kpts
                      && static_cast<int>(last_.P_k.size()) == last_.n_kpts))
                    throw std::runtime_error(
                        "GPWEvaluator: multi-k second_gen_cp force path needs "
                        "P_k / C_occ_k / eps_k from the corrector SCF");
                if (uks_sgcp
                    && (static_cast<int>(last_.P_k_beta.size()) != last_.n_kpts
                        || static_cast<int>(last_.C_occ_k_beta.size())
                               != last_.n_kpts
                        || static_cast<int>(last_.eps_k_beta.size())
                               != last_.n_kpts))
                    throw std::runtime_error(
                        "GPWEvaluator: multi-k UKS second_gen_cp force path "
                        "needs dual-spin k payloads from the corrector SCF");
                GpwResult force_scf = last_;
                force_scf.converged = true;
                const auto F = gpw_forces_from_scf(atoms, *solver_, force_scf);
                AccReal* __restrict sp_fx = frame.span(kAtom_fx);
                AccReal* __restrict sp_fy = frame.span(kAtom_fy);
                AccReal* __restrict sp_fz = frame.span(kAtom_fz);
                for (int i = 0; i < n; ++i) {
                    sp_fx[i] += F[static_cast<std::size_t>(i)][0]
                                * ATV::units::kHartreePerBohrInEvPerAngstrom;
                    sp_fy[i] += F[static_cast<std::size_t>(i)][1]
                                * ATV::units::kHartreePerBohrInEvPerAngstrom;
                    sp_fz[i] += F[static_cast<std::size_t>(i)][2]
                                * ATV::units::kHartreePerBohrInEvPerAngstrom;
                }
            }
            return;
        }

        // ── Γ real ASPC ────────────────────────────────────────────────
        if (!sgcp_aspc_)
            throw std::logic_error(
                "GPWEvaluator: second_gen_cp Γ path missing ASPC state");
        xt::xtensor<double, 2> P_pred;
        xt::xtensor<double, 2> P_pred_beta;
        const bool have_pred_a = sgcp_aspc_->predict(P_pred);
        bool have_pred_b = false;
        if (uks_sgcp && sgcp_aspc_beta_)
            have_pred_b = sgcp_aspc_beta_->predict(P_pred_beta);
        // Both channels must be warm together (shared MD step count).
        const bool have_pred =
            uks_sgcp ? (have_pred_a && have_pred_b) : have_pred_a;
        const ATV::cpu::linalg::RealMatrix* guess_c =
            (last_.C.size() > 0) ? &last_.C : nullptr;
        const ATV::cpu::linalg::RealMatrix* guess_p = nullptr;
        const ATV::cpu::linalg::RealMatrix* guess_p_beta = nullptr;
        const int saved_max = solver_->max_iter;
        if (have_pred) {
            // Layout convert: ASPC uses default-layout xtensor; GPW is
            // column-major RealMatrix.
            const auto n0 = P_pred.shape(0);
            const auto n1 = P_pred.shape(1);
            sgcp_guess_p_ = xt::zeros<double>({n0, n1});
            for (std::size_t i = 0; i < n0; ++i)
                for (std::size_t j = 0; j < n1; ++j)
                    sgcp_guess_p_(i, j) = P_pred(i, j);
            guess_p = &sgcp_guess_p_;
            if (uks_sgcp) {
                const auto m0 = P_pred_beta.shape(0);
                const auto m1 = P_pred_beta.shape(1);
                sgcp_guess_p_beta_ = xt::zeros<double>({m0, m1});
                for (std::size_t i = 0; i < m0; ++i)
                    for (std::size_t j = 0; j < m1; ++j)
                        sgcp_guess_p_beta_(i, j) = P_pred_beta(i, j);
                guess_p_beta = &sgcp_guess_p_beta_;
            }
            solver_->max_iter = second_gen_cp_max_iter;
        }
        try {
            last_ = solver_->run(atoms, guess_c, guess_p, guess_p_beta);
        } catch (...) {
            solver_->max_iter = saved_max;
            throw;
        }
        solver_->max_iter = saved_max;

        auto record_channel =
            [](electronic::ASPCPredictor& aspc, const linalg::RealMatrix& Pmat,
               bool do_correct, const xt::xtensor<double, 2>& P_pred_in) {
                xt::xtensor<double, 2> P_scf =
                    xt::zeros<double>({Pmat.shape(0), Pmat.shape(1)});
                for (std::size_t i = 0; i < Pmat.shape(0); ++i)
                    for (std::size_t j = 0; j < Pmat.shape(1); ++j)
                        P_scf(i, j) = Pmat(i, j);
                if (do_correct) {
                    xt::xtensor<double, 2> P_corr;
                    aspc.correct(P_scf, P_pred_in, P_corr);
                    aspc.record(P_corr);
                } else {
                    aspc.record(P_scf);
                }
            };

        if (last_.P.size() > 0) {
            if (!uks_sgcp) {
                // RKS force-at-density needs H_ks / S when the SCF stored
                // orbitals without the fixed-density payload. DIIS path can
                // re-eval with max_iter=0; OT forbids fixed-density eval —
                // OT exit must export H_ks/S itself (second-gen-cpmd-ot).
                if (last_.H_ks.size() == 0 || last_.S.size() == 0) {
                    if (use_ot || solver_->use_ot)
                        throw std::runtime_error(
                            "GPWEvaluator: second_gen_cp OT corrector did not "
                            "export H_ks/S (required for forces at density)");
                    const int saved = solver_->max_iter;
                    solver_->max_iter = 0;
                    try {
                        auto tmp =
                            solver_->run(atoms, /*guess_C=*/nullptr, &last_.P);
                        last_.H_ks = std::move(tmp.H_ks);
                        last_.S = std::move(tmp.S);
                    } catch (...) {
                        solver_->max_iter = saved;
                        throw;
                    }
                    solver_->max_iter = saved;
                }
            }
            record_channel(*sgcp_aspc_, last_.P, have_pred, P_pred);
            if (uks_sgcp && sgcp_aspc_beta_) {
                if (last_.P_beta.size() > 0)
                    record_channel(*sgcp_aspc_beta_, last_.P_beta, have_pred,
                                   P_pred_beta);
                else
                    sgcp_aspc_beta_->reset();
            }
        } else {
            sgcp_aspc_->reset();
            if (sgcp_aspc_beta_)
                sgcp_aspc_beta_->reset();
        }

        frame.meta.get(kEnergy) += last_.e_total * ATV::units::kHartreeInEv;
        frame.meta.get(kEnergyElectronic) +=
            last_.e_total * ATV::units::kHartreeInEv;

        if (compute_forces) {
            std::vector<std::array<double, 3>> F;
            if (uks_sgcp) {
                // UKS: forces at the post-corrector density (partial SCF may
                // not hit tol). gpw_forces_from_scf requires the converged
                // flag; SGCP semantics are forces at the corrector state.
                if (last_.P.size() == 0 || last_.C.size() == 0
                    || last_.eps.empty() || last_.P_beta.size() == 0
                    || last_.C_beta.size() == 0 || last_.eps_beta.empty())
                    throw std::runtime_error(
                        "GPWEvaluator: second_gen_cp UKS force path needs "
                        "P/C/eps for both spins from the corrector SCF");
                GpwResult force_scf = last_;
                force_scf.converged = true;
                F = gpw_forces_from_scf(atoms, *solver_, force_scf);
            } else if (use_ot || solver_->use_ot) {
                // OT corrector: prefer forces-from-scf when C/eps present
                // (partial OT may leave converged=false). Fall back to
                // density path when H_ks was exported.
                if (last_.P.size() > 0 && last_.C.size() > 0
                    && !last_.eps.empty()) {
                    GpwResult force_scf = last_;
                    force_scf.converged = true;
                    F = gpw_forces_from_scf(atoms, *solver_, force_scf);
                } else if (last_.C.size() > 0 && last_.P.size() > 0
                           && last_.H_ks.size() > 0) {
                    const int nb = static_cast<int>(last_.C.shape(0));
                    const int n_occ = solver_->n_occ;
                    linalg::RealMatrix C_occ =
                        xt::zeros<double>({static_cast<std::size_t>(nb),
                                           static_cast<std::size_t>(n_occ)});
                    for (int mu = 0; mu < nb; ++mu)
                        for (int i = 0; i < n_occ; ++i)
                            C_occ(mu, i) = last_.C(mu, i);
                    F = gpw_forces_at_density(atoms, *solver_, last_.P, C_occ,
                                              last_.H_ks);
                } else {
                    throw std::runtime_error(
                        "GPWEvaluator: second_gen_cp OT force path needs "
                        "P/C(/eps or H_ks) from the corrector SCF");
                }
            } else {
                if (last_.C.size() == 0 || last_.P.size() == 0
                    || last_.H_ks.size() == 0)
                    throw std::runtime_error(
                        "GPWEvaluator: second_gen_cp force path needs P, C, "
                        "H_ks from the corrector SCF");
                const int nb = static_cast<int>(last_.C.shape(0));
                const int n_occ = solver_->n_occ;
                linalg::RealMatrix C_occ =
                    xt::zeros<double>({static_cast<std::size_t>(nb),
                                       static_cast<std::size_t>(n_occ)});
                for (int mu = 0; mu < nb; ++mu)
                    for (int i = 0; i < n_occ; ++i)
                        C_occ(mu, i) = last_.C(mu, i);
                F = gpw_forces_at_density(atoms, *solver_, last_.P, C_occ,
                                          last_.H_ks);
            }
            AccReal* __restrict sp_fx = frame.span(kAtom_fx);
            AccReal* __restrict sp_fy = frame.span(kAtom_fy);
            AccReal* __restrict sp_fz = frame.span(kAtom_fz);
            for (int i = 0; i < n; ++i) {
                sp_fx[i] += F[static_cast<std::size_t>(i)][0]
                            * ATV::units::kHartreePerBohrInEvPerAngstrom;
                sp_fy[i] += F[static_cast<std::size_t>(i)][1]
                            * ATV::units::kHartreePerBohrInEvPerAngstrom;
                sp_fz[i] += F[static_cast<std::size_t>(i)][2]
                            * ATV::units::kHartreePerBohrInEvPerAngstrom;
            }
        }
        return;
    }

    // ── Zero-SCF linearized XL-BOMD (shadow Born–Oppenheimer forces) ────
    // After the cold full-SCF seed (xl history ready): one H[n] build from
    // the auxiliary density n, one diagonalization → D = q[n], shadow
    // energy E = Tr(H D) − Tr(H n) + E_KS[n], forces at (D, H[n]). No
    // iterative SCF (iters = 0). Γ UKS is spin-diagonal: dual auxiliary
    // densities n^α/n^β, one dual-KS build, one diagonalization per spin
    // (occ 1), E = Σ_σ [Tr(H^σ D^σ) − Tr(H^σ n^σ)] + E_KS[n^α,n^β].
    // Cold seed falls through to the SCF-XL path below (first advance
    // seeds the histories with D_scf / P^σ_scf).
    const bool lin_uks = (n_alpha + n_beta > 0);
    if (xl_bomd_linearized && xl_ && xl_->ready()
        && (!lin_uks || (xl_beta_ && xl_beta_->ready()))) {
        const linalg::RealMatrix& n_aux = xl_->aux();
        const linalg::RealMatrix* n_aux_beta =
            lin_uks ? &xl_beta_->aux() : nullptr;
        const int saved_max = solver_->max_iter;
        solver_->max_iter = 0;
        try {
            last_ =
                solver_->run(atoms, /*guess_C=*/nullptr, &n_aux, n_aux_beta);
        } catch (...) {
            solver_->max_iter = saved_max;
            throw;
        }
        solver_->max_iter = saved_max;

        if (last_.H_ks.size() == 0 || last_.S.size() == 0 || last_.P.size() == 0
            || (lin_uks
                && (last_.H_ks_beta.size() == 0 || last_.P_beta.size() == 0)))
            throw std::runtime_error(
                "GPWEvaluator: xl_bomd_linearized fixed-density step needs "
                "H_ks (+H_ks_beta for UKS), S, and P from max_iter=0");

        const int nb = static_cast<int>(last_.H_ks.shape(0));

        // Shadow energy trace: Tr(H M) over the AO basis.
        auto tr_HM = [nb](const linalg::RealMatrix& H,
                          const linalg::RealMatrix& M) {
            double t = 0.0;
            for (int i = 0; i < nb; ++i)
                for (int j = 0; j < nb; ++j)
                    t += H(static_cast<std::size_t>(i),
                           static_cast<std::size_t>(j))
                         * M(static_cast<std::size_t>(j),
                             static_cast<std::size_t>(i));
            return t;
        };
        auto occupied_block = [nb](const linalg::RealMatrix& C, int n_occ) {
            linalg::RealMatrix C_occ =
                xt::zeros<double>({static_cast<std::size_t>(nb),
                                   static_cast<std::size_t>(n_occ)});
            for (int mu = 0; mu < nb; ++mu)
                for (int i = 0; i < n_occ; ++i)
                    C_occ(static_cast<std::size_t>(mu),
                          static_cast<std::size_t>(i)) =
                        C(static_cast<std::size_t>(mu),
                          static_cast<std::size_t>(i));
            return C_occ;
        };
        const double e_ks_n = last_.e_total;
        std::vector<std::array<double, 3>> F;

        if (!lin_uks) {
            const int n_occ = solver_->n_occ;
            if (n_occ < 1)
                throw std::runtime_error(
                    "GPWEvaluator: xl_bomd_linearized needs n_occ >= 1");

            // One diagonalization of H[n] → D = q[n] (zero iterative SCF).
            auto [eig, C] =
                linalg::diagonalize_fock_generalized(last_.H_ks, last_.S);
            const linalg::RealMatrix D =
                linalg::build_density_matrix(C, n_occ, nb, 2.0);

            // Shadow energy: Tr(H D) − Tr(H n) + E_KS[n]; E_KS at D=n.
            last_.e_total =
                tr_HM(last_.H_ks, D) - tr_HM(last_.H_ks, last_.P) + e_ks_n;

            last_.P = D;
            last_.C = C;
            last_.eps.assign(eig.begin(), eig.end());
            last_.converged = true; // stationary for the shadow functional
            last_.iters = 0;        // zero iterative SCF

            xl_->advance(D);

            if (compute_forces) {
                // Forces of U(R,n) at fixed n: density D, KS matrix H[n].
                F = gpw_forces_at_density(atoms, *solver_, D,
                                          occupied_block(C, n_occ), last_.H_ks);
            }
        } else {
            // Γ UKS: one generalized diagonalization per spin channel.
            auto [eig_a, Ca] =
                linalg::diagonalize_fock_generalized(last_.H_ks, last_.S);
            auto [eig_b, Cb] =
                linalg::diagonalize_fock_generalized(last_.H_ks_beta, last_.S);
            const linalg::RealMatrix Da =
                linalg::build_density_matrix(Ca, n_alpha, nb, 1.0);
            const linalg::RealMatrix Db =
                linalg::build_density_matrix(Cb, n_beta, nb, 1.0);

            // Dual-spin shadow energy — recovers E_KS at D^σ = n^σ.
            last_.e_total = (tr_HM(last_.H_ks, Da) - tr_HM(last_.H_ks, last_.P))
                            + (tr_HM(last_.H_ks_beta, Db)
                               - tr_HM(last_.H_ks_beta, last_.P_beta))
                            + e_ks_n;

            // Löwdin ⟨S²⟩ at the shadow densities (same formula as the
            // converged-SCF store): s_z(s_z+1) + N_β − Σ|C_αᵀ S C_β|².
            const double sz = 0.5 * static_cast<double>(n_alpha - n_beta);
            double ov = 0.0;
            if (n_alpha > 0 && n_beta > 0) {
                // SCb = S·C_β_occ (nb × n_beta), then Sab = C_α_occᵀ·SCb.
                linalg::RealMatrix SCb =
                    xt::zeros<double>({static_cast<std::size_t>(nb),
                                       static_cast<std::size_t>(n_beta)});
                for (int mu = 0; mu < nb; ++mu)
                    for (int nu = 0; nu < nb; ++nu) {
                        const double s = last_.S(static_cast<std::size_t>(mu),
                                                 static_cast<std::size_t>(nu));
                        for (int j = 0; j < n_beta; ++j)
                            SCb(static_cast<std::size_t>(mu),
                                static_cast<std::size_t>(j)) +=
                                s
                                * Cb(static_cast<std::size_t>(nu),
                                     static_cast<std::size_t>(j));
                    }
                for (int i = 0; i < n_alpha; ++i)
                    for (int j = 0; j < n_beta; ++j) {
                        double sab = 0.0;
                        for (int mu = 0; mu < nb; ++mu)
                            sab += Ca(static_cast<std::size_t>(mu),
                                      static_cast<std::size_t>(i))
                                   * SCb(static_cast<std::size_t>(mu),
                                         static_cast<std::size_t>(j));
                        ov += sab * sab;
                    }
            }
            last_.s2 = sz * (sz + 1.0) + static_cast<double>(n_beta) - ov;

            last_.P = Da;
            last_.P_beta = Db;
            last_.C = Ca;
            last_.C_beta = Cb;
            last_.eps.assign(eig_a.begin(), eig_a.end());
            last_.eps_beta.assign(eig_b.begin(), eig_b.end());
            last_.converged = true; // stationary for the shadow functional
            last_.iters = 0;        // zero iterative SCF

            xl_->advance(Da);
            xl_beta_->advance(Db);

            if (compute_forces) {
                // Forces of U(R,n^α,n^β) at fixed n^σ: densities D^σ, KS
                // matrices H^σ[n] (dual-spin fixed-density overload).
                F = gpw_forces_at_density(
                    atoms, *solver_, Da, occupied_block(Ca, n_alpha),
                    last_.H_ks, Db, occupied_block(Cb, n_beta),
                    last_.H_ks_beta);
            }
        }

        frame.meta.get(kEnergy) += last_.e_total * ATV::units::kHartreeInEv;
        frame.meta.get(kEnergyElectronic) +=
            last_.e_total * ATV::units::kHartreeInEv;

        if (compute_forces) {
            AccReal* __restrict sp_fx = frame.span(kAtom_fx);
            AccReal* __restrict sp_fy = frame.span(kAtom_fy);
            AccReal* __restrict sp_fz = frame.span(kAtom_fz);
            for (int i = 0; i < n; ++i) {
                sp_fx[i] += F[static_cast<std::size_t>(i)][0]
                            * ATV::units::kHartreePerBohrInEvPerAngstrom;
                sp_fy[i] += F[static_cast<std::size_t>(i)][1]
                            * ATV::units::kHartreePerBohrInEvPerAngstrom;
                sp_fz[i] += F[static_cast<std::size_t>(i)][2]
                            * ATV::units::kHartreePerBohrInEvPerAngstrom;
            }
        }

        if (compute_stress) {
            // Shadow-density stress: analytic fixed-P Lagrangian at the warm
            // payload (P = D = q[n], W from the H[n] eigenpairs) — the cell
            // analogue of gpw_forces_at_density. No reference SCF (zero-SCF
            // preserved); stress_use_fd is refused at build().
            last_stress_ha_ = gpw_stress(atoms, *solver_, last_,
                                         stress_strain_h, stress_axes);
            add_stress_to_frame(frame, last_stress_ha_);
        }
        return;
    }

    // ── Multi-k zero-SCF linearized XL-BOMD (shadow forces on the k path) ──
    // After the cold multi-k SCF seed (one complex XL history per sampled k,
    // dual per spin for UKS): one KS build at the auxiliary densities {n_k}
    // (fixed-density, max_iter = 0), one generalized diagonalization per
    // (k, σ) → D_k^σ = q[n], k-weighted shadow energy
    //   E = Σ_k w_k Σ_σ Re Tr[H_k^σ[n] (D_k^σ − n_k^σ)] + E_KS[n],
    // and forces from the (D_k, ε_k, C_k^occ) payload on the k-path
    // assembly. No iterative SCF (iters = 0). The cold seed falls through
    // to the SCF-XL path below (its first advance seeds each per-k history
    // with P_k^σ_scf).
    const bool lin_k_ready = [&]() {
        if (!xl_bomd_linearized || xl_k_.empty())
            return false;
        for (const auto& x : xl_k_)
            if (!x.ready())
                return false;
        if (lin_uks) {
            if (xl_k_beta_.size() != xl_k_.size())
                return false;
            for (const auto& x : xl_k_beta_)
                if (!x.ready())
                    return false;
        }
        return true;
    }();
    if (lin_k_ready) {
        const int nk = static_cast<int>(xl_k_.size());
        xl_guess_p_k_.resize(static_cast<std::size_t>(nk));
        for (int ik = 0; ik < nk; ++ik)
            xl_guess_p_k_[static_cast<std::size_t>(ik)] =
                xl_k_[static_cast<std::size_t>(ik)].aux();
        if (lin_uks) {
            xl_guess_p_k_beta_.resize(static_cast<std::size_t>(nk));
            for (int ik = 0; ik < nk; ++ik)
                xl_guess_p_k_beta_[static_cast<std::size_t>(ik)] =
                    xl_k_beta_[static_cast<std::size_t>(ik)].aux();
        }
        const int saved_max = solver_->max_iter;
        solver_->max_iter = 0;
        try {
            last_ = solver_->run(atoms, /*guess_C=*/nullptr,
                                 /*guess_P=*/nullptr, /*guess_P_beta=*/nullptr,
                                 &xl_guess_p_k_,
                                 lin_uks ? &xl_guess_p_k_beta_ : nullptr);
        } catch (...) {
            solver_->max_iter = saved_max;
            throw;
        }
        solver_->max_iter = saved_max;

        if (last_.n_kpts != nk || static_cast<int>(last_.H_ks_k.size()) != nk
            || static_cast<int>(last_.S_k.size()) != nk
            || static_cast<int>(last_.k_weights.size()) != nk
            || (lin_uks && static_cast<int>(last_.H_ks_k_beta.size()) != nk))
            throw std::runtime_error(
                "GPWEvaluator: multi-k xl_bomd_linearized fixed-density step "
                "needs H_ks_k (+H_ks_k_beta for UKS), S_k and k_weights from "
                "max_iter=0");

        const double e_ks_n = last_.e_total;
        // Re Tr(H M) over one complex Bloch pair (shadow trace term).
        auto tr_HM = [](const linalg::CplxMatrix& H,
                        const linalg::CplxMatrix& M) {
            const int m = static_cast<int>(H.shape(0));
            double t = 0.0;
            for (int i = 0; i < m; ++i)
                for (int j = 0; j < m; ++j)
                    t += (H(static_cast<std::size_t>(i),
                            static_cast<std::size_t>(j))
                          * M(static_cast<std::size_t>(j),
                              static_cast<std::size_t>(i)))
                             .real();
            return t;
        };

        const int n_occ_a = lin_uks ? n_alpha : solver_->n_occ;
        const int n_occ_b = lin_uks ? n_beta : 0;
        if (n_occ_a < 1)
            throw std::runtime_error(
                lin_uks ? "GPWEvaluator: multi-k UKS xl_bomd_linearized needs "
                          "n_alpha >= 1"
                        : "GPWEvaluator: multi-k xl_bomd_linearized needs "
                          "n_occ >= 1");

        last_.eps_k.assign(static_cast<std::size_t>(nk), {});
        last_.C_occ_k.assign(static_cast<std::size_t>(nk), {});
        last_.P_k.assign(static_cast<std::size_t>(nk), linalg::CplxMatrix{});
        if (lin_uks) {
            last_.eps_k_beta.assign(static_cast<std::size_t>(nk), {});
            last_.C_occ_k_beta.assign(static_cast<std::size_t>(nk), {});
            last_.P_k_beta.assign(static_cast<std::size_t>(nk),
                                  linalg::CplxMatrix{});
        }

        double shadow = 0.0;
        for (int ik = 0; ik < nk; ++ik) {
            const std::size_t k = static_cast<std::size_t>(ik);
            const double wk = last_.k_weights[k];
            const internal::KOrth orth(last_.S_k[k], solver_->lindep_threshold);
            auto [eps_a, Ca] = internal::diagonalize_k(last_.H_ks_k[k], orth);
            if (static_cast<int>(Ca.shape(1)) < n_occ_a)
                throw std::runtime_error(
                    "GPWEvaluator: multi-k xl_bomd_linearized: occupied "
                    "count exceeds the independent basis directions at a "
                    "k-point");
            const linalg::CplxMatrix Da =
                density_from_cocc(Ca, n_occ_a, lin_uks ? 1.0 : 2.0);
            shadow += wk
                      * (tr_HM(last_.H_ks_k[k], Da)
                         - tr_HM(last_.H_ks_k[k], xl_guess_p_k_[k]));
            last_.eps_k[k] = std::move(eps_a);
            last_.C_occ_k[k] = flatten_occ(Ca, n_occ_a);
            last_.P_k[k] = Da;
            if (lin_uks) {
                auto [eps_b, Cb] =
                    internal::diagonalize_k(last_.H_ks_k_beta[k], orth);
                if (static_cast<int>(Cb.shape(1)) < n_occ_b)
                    throw std::runtime_error(
                        "GPWEvaluator: multi-k UKS xl_bomd_linearized: "
                        "occupied beta count exceeds the independent basis "
                        "directions at a k-point");
                const linalg::CplxMatrix Db =
                    density_from_cocc(Cb, n_occ_b, 1.0);
                shadow +=
                    wk
                    * (tr_HM(last_.H_ks_k_beta[k], Db)
                       - tr_HM(last_.H_ks_k_beta[k], xl_guess_p_k_beta_[k]));
                last_.eps_k_beta[k] = std::move(eps_b);
                last_.C_occ_k_beta[k] = flatten_occ(Cb, n_occ_b);
                last_.P_k_beta[k] = Db;
            }
            // XL advance(D) per k (per spin): the aux for the next step.
            xl_k_[k].advance(last_.P_k[k]);
            if (lin_uks)
                xl_k_beta_[k].advance(last_.P_k_beta[k]);
        }
        last_.e_total = e_ks_n + shadow;
        last_.converged = true; // stationary for the shadow functional
        last_.iters = 0;        // zero iterative SCF

        frame.meta.get(kEnergy) += last_.e_total * ATV::units::kHartreeInEv;
        frame.meta.get(kEnergyElectronic) +=
            last_.e_total * ATV::units::kHartreeInEv;

        if (compute_forces) {
            // Forces of U(R,{n_k}) at fixed n: the k-path assembly at the
            // (D_k, ε_k, C_k^occ) payload — since C_k are exact
            // eigenvectors of H_k[n], W = C (C†HC) C† (the Γ product's
            // off-BO energy-weighted density) reduces to Σ f ε C C†.
            const auto F = gpw_forces_from_scf(atoms, *solver_, last_);
            AccReal* __restrict sp_fx = frame.span(kAtom_fx);
            AccReal* __restrict sp_fy = frame.span(kAtom_fy);
            AccReal* __restrict sp_fz = frame.span(kAtom_fz);
            for (int i = 0; i < n; ++i) {
                sp_fx[i] += F[static_cast<std::size_t>(i)][0]
                            * ATV::units::kHartreePerBohrInEvPerAngstrom;
                sp_fy[i] += F[static_cast<std::size_t>(i)][1]
                            * ATV::units::kHartreePerBohrInEvPerAngstrom;
                sp_fz[i] += F[static_cast<std::size_t>(i)][2]
                            * ATV::units::kHartreePerBohrInEvPerAngstrom;
            }
        }

        if (compute_stress) {
            // Shadow-density stress on the k path: k-summed fixed-{P_k,W_k}
            // Lagrangian at the warm payload (C_occ_k / eps_k just built) —
            // no reference SCF, zero-SCF preserved.
            last_stress_ha_ = gpw_stress(atoms, *solver_, last_,
                                         stress_strain_h, stress_axes);
            add_stress_to_frame(frame, last_stress_ha_);
        }
        return;
    }

    // BOMD / FD-phonon warm-start: reuse previous MOs and spin densities.
    // GpwDft::run accepts optional guess_C; empty last_.C → core guess.
    // Multi-k has no real C warm-start (Bloch C_k live on the k path).
    //
    // UKS critical path: guess_C alone seeds *both* spins from α-like
    // columns (see GpwDft::run), which is a cold open-shell start. Pass
    // the converged P_α / P_β so successive force evaluations (Phonon
    // displacements, UKS AIMD) re-enter near the previous solution.
    const ATV::cpu::linalg::RealMatrix* guess = nullptr;
    if (last_.C.size() > 0 && last_.n_kpts == 0) {
        guess = &last_.C;
    }
    // XL-BOMD: propagated auxiliary density is the SCF initial-density
    // guess (Niklasson dissipative recursion, seeded on the first step).
    // Γ RKS: total P. Γ UKS: P_α / P_β. multi-k RKS: P_k. multi-k UKS:
    // dual P_k^α / P_k^β.
    // (Linearized warm path returned above when aux is ready.)
    const ATV::cpu::linalg::RealMatrix* guess_p =
        (xl_ && xl_->ready()) ? &xl_->aux() : nullptr;
    const ATV::cpu::linalg::RealMatrix* guess_p_beta =
        (xl_beta_ && xl_beta_->ready()) ? &xl_beta_->aux() : nullptr;
    if (guess_p == nullptr && last_.n_kpts == 0 && last_.P.size() > 0
        && last_.converged) {
        if (n_alpha + n_beta > 0) {
            if (last_.P_beta.size() > 0) {
                guess_p = &last_.P;
                guess_p_beta = &last_.P_beta;
            }
        } else {
            guess_p = &last_.P;
        }
    }
    const std::vector<ATV::cpu::linalg::CplxMatrix>* guess_p_k = nullptr;
    const std::vector<ATV::cpu::linalg::CplxMatrix>* guess_p_k_beta = nullptr;
    if (!xl_k_.empty() && xl_k_[0].ready()) {
        xl_guess_p_k_.resize(xl_k_.size());
        for (std::size_t ik = 0; ik < xl_k_.size(); ++ik)
            xl_guess_p_k_[ik] = xl_k_[ik].aux();
        guess_p_k = &xl_guess_p_k_;
        if (!xl_k_beta_.empty() && xl_k_beta_[0].ready()
            && xl_k_beta_.size() == xl_k_.size()) {
            xl_guess_p_k_beta_.resize(xl_k_beta_.size());
            for (std::size_t ik = 0; ik < xl_k_beta_.size(); ++ik)
                xl_guess_p_k_beta_[ik] = xl_k_beta_[ik].aux();
            guess_p_k_beta = &xl_guess_p_k_beta_;
        }
    }
    // Plain multi-k BOMD warm start (no XL aux): the previous step's
    // converged P_k (UKS: both P_k^alpha and P_k^beta — the solver rejects a
    // partial spin guess) seeds the SCF, the k-path analogue of the Γ
    // guess_C/guess_P block above and of CP2K's `EXTRAPOLATION USE_PREV_P`.
    // Pointers into `last_` stay valid through `run` — the result is copied
    // into `last_` only after it returns (same lifetime story as `&last_.C`).
    if (guess_p_k == nullptr && last_.n_kpts > 0 && last_.converged
        && static_cast<int>(last_.P_k.size()) == last_.n_kpts) {
        if (n_alpha + n_beta > 0) {
            if (static_cast<int>(last_.P_k_beta.size()) == last_.n_kpts) {
                guess_p_k = &last_.P_k;
                guess_p_k_beta = &last_.P_k_beta;
            }
        } else {
            guess_p_k = &last_.P_k;
        }
    }
    last_ = solver_->run(atoms, guess, guess_p, guess_p_beta, guess_p_k,
                         guess_p_k_beta);
    if (xl_) {
        if (last_.converged && last_.P.size() > 0) {
            xl_->advance(last_.P);
            if (xl_beta_) {
                if (last_.P_beta.size() > 0)
                    xl_beta_->advance(last_.P_beta);
                else
                    xl_beta_->reset();
            }
        } else {
            // Discontinuity: never propagate an unconverged D.
            xl_->reset();
            if (xl_beta_)
                xl_beta_->reset();
        }
    }
    // Multi-k XL: advance one complex propagator per k from SCF P_k
    // (RKS total or UKS α). Multi-k UKS also advances β from P_k_beta.
    if (xl_bomd && last_.n_kpts > 0) {
        const bool uks_k = (n_alpha + n_beta > 0);
        const bool have_a =
            last_.converged
            && static_cast<int>(last_.P_k.size()) == last_.n_kpts
            && last_.n_kpts > 0;
        const bool have_b =
            !uks_k
            || (have_a
                && static_cast<int>(last_.P_k_beta.size()) == last_.n_kpts);
        if (have_a && have_b) {
            auto reseat = [&](std::vector<electronic::XlBomdCplx>& xs) {
                if (static_cast<int>(xs.size()) != last_.n_kpts) {
                    xs.clear();
                    xs.reserve(static_cast<std::size_t>(last_.n_kpts));
                    for (int ik = 0; ik < last_.n_kpts; ++ik)
                        xs.emplace_back(xl_bomd_k);
                }
            };
            reseat(xl_k_);
            for (int ik = 0; ik < last_.n_kpts; ++ik)
                xl_k_[static_cast<std::size_t>(ik)].advance(
                    last_.P_k[static_cast<std::size_t>(ik)]);
            if (uks_k) {
                reseat(xl_k_beta_);
                for (int ik = 0; ik < last_.n_kpts; ++ik)
                    xl_k_beta_[static_cast<std::size_t>(ik)].advance(
                        last_.P_k_beta[static_cast<std::size_t>(ik)]);
            } else {
                xl_k_beta_.clear();
            }
        } else {
            for (auto& xl : xl_k_)
                xl.reset();
            for (auto& xl : xl_k_beta_)
                xl.reset();
        }
    }
    frame.meta.get(kEnergy) += last_.e_total * ATV::units::kHartreeInEv;
    frame.meta.get(kEnergyElectronic) +=
        last_.e_total * ATV::units::kHartreeInEv;

    if (compute_forces) {
        // Prefer forces at the density just converged (no second SCF). UKS has
        // no guess_C warm-start, so a re-SCF would double wall; RKS cold
        // re-SCF is kept only when the energy result lacks orbitals (should
        // not happen after a normal converged run). Multi-k uses C_occ_k.
        std::vector<std::array<double, 3>> F;
        const bool gamma_payload = last_.converged && last_.P.size() > 0
                                   && last_.C.size() > 0 && !last_.eps.empty();
        const bool k_payload =
            last_.converged && last_.n_kpts > 0
            && static_cast<int>(last_.C_occ_k.size()) == last_.n_kpts
            && static_cast<int>(last_.eps_k.size()) == last_.n_kpts;
        if (gamma_payload || k_payload) {
            F = gpw_forces_from_scf(atoms, *solver_, last_);
        } else {
            // XL mode warm-starts the force-pass SCF from the MO coefficients
            // just converged at this exact geometry. Default mode keeps the
            // cold solve unchanged.
            F = gpw_forces_analytic(atoms, *solver_,
                                    (xl_ && last_.C.size() > 0) ? &last_.C
                                                                : nullptr);
        }
        // [Hartree/Bohr]; one SCF + gradient (was 6N+1 FD)
        AccReal* __restrict sp_fx = frame.span(kAtom_fx);
        AccReal* __restrict sp_fy = frame.span(kAtom_fy);
        AccReal* __restrict sp_fz = frame.span(kAtom_fz);
        for (int i = 0; i < n; ++i) {
            sp_fx[i] += F[static_cast<std::size_t>(i)][0]
                        * ATV::units::kHartreePerBohrInEvPerAngstrom;
            sp_fy[i] += F[static_cast<std::size_t>(i)][1]
                        * ATV::units::kHartreePerBohrInEvPerAngstrom;
            sp_fz[i] += F[static_cast<std::size_t>(i)][2]
                        * ATV::units::kHartreePerBohrInEvPerAngstrom;
        }
    }

    // CPMD cold seed after the first full SCF + BO forces.
    // Multi-k RKS: seed one CpmdCplx per sampled k from Bloch C_k / C_occ_k.
    // Multi-k UKS: one per (k, spin) — α from C_k / C_occ_k (n_alpha
    // columns), β from C_k_beta / C_occ_k_beta (n_beta columns).
    if (cpmd && cpmd_k_.empty()
        && (k_mesh[0] != 1 || k_mesh[1] != 1 || k_mesh[2] != 1)) {
        if (!last_.converged || last_.n_kpts < 1)
            throw std::runtime_error(
                "GPWEvaluator: multi-k cpmd seed requires a converged multi-k "
                "SCF");
        const int nk = last_.n_kpts;
        const bool seed_uks = (n_alpha + n_beta > 0);
        const int n_occ_a = seed_uks ? n_alpha : solver_->n_occ;
        if (n_occ_a < 1)
            throw std::runtime_error(
                seed_uks ? "GPWEvaluator: multi-k UKS CPMD seed requires "
                           "n_alpha >= 1"
                         : "GPWEvaluator: multi-k RKS CPMD seed requires "
                           "n_occ >= 1");
        // One channel of per-k propagators from full C_k (preferred) or the
        // flat occupied block; `tag` names the channel in error messages.
        auto seed_channel =
            [&](const std::vector<linalg::CplxMatrix>& C_k_full,
                const std::vector<std::vector<std::complex<double>>>& C_occ_k,
                int n_occ, const char* tag) {
                const bool have_C_k = static_cast<int>(C_k_full.size()) == nk
                                      && C_k_full[0].size() > 0;
                const bool have_C_occ = static_cast<int>(C_occ_k.size()) == nk
                                        && !C_occ_k[0].empty();
                if (!have_C_k && !have_C_occ)
                    throw std::runtime_error(
                        std::string("GPWEvaluator: multi-k cpmd seed missing ")
                        + tag);
                std::vector<electronic::CpmdCplx> out;
                out.reserve(static_cast<std::size_t>(nk));
                for (int ik = 0; ik < nk; ++ik) {
                    linalg::CplxMatrix C_occ;
                    if (have_C_k) {
                        const auto& Cfull =
                            C_k_full[static_cast<std::size_t>(ik)];
                        const int nb = static_cast<int>(Cfull.shape(0));
                        if (static_cast<int>(Cfull.shape(1)) < n_occ)
                            throw std::runtime_error(
                                std::string("GPWEvaluator: multi-k cpmd "
                                            "seed: ")
                                + tag + " columns < occupied count");
                        C_occ = xt::zeros<std::complex<double>>(
                            {static_cast<std::size_t>(nb),
                             static_cast<std::size_t>(n_occ)});
                        for (int mu = 0; mu < nb; ++mu)
                            for (int o = 0; o < n_occ; ++o)
                                C_occ(mu, o) = Cfull(mu, o);
                    } else {
                        const auto& flat =
                            C_occ_k[static_cast<std::size_t>(ik)];
                        // Infer nb from flat size / n_occ.
                        if (static_cast<int>(flat.size()) % n_occ != 0)
                            throw std::runtime_error(
                                std::string("GPWEvaluator: multi-k cpmd "
                                            "seed: ")
                                + tag
                                + " size not divisible by the occupied "
                                  "count");
                        const int nb = static_cast<int>(flat.size()) / n_occ;
                        C_occ = xt::zeros<std::complex<double>>(
                            {static_cast<std::size_t>(nb),
                             static_cast<std::size_t>(n_occ)});
                        for (int mu = 0; mu < nb; ++mu)
                            for (int o = 0; o < n_occ; ++o)
                                C_occ(mu, o) =
                                    flat[static_cast<std::size_t>(mu) * n_occ
                                         + static_cast<std::size_t>(o)];
                    }
                    electronic::CpmdCplx ch(cpmd_mu);
                    // Electronic Nose-Hoover per (k, sigma) channel — each
                    // regulates its own T_e toward the full K0 (the Γ-UKS
                    // convention; published totals stay k-weighted).
                    ch.thermostat = cpmd_thermostat;
                    ch.ekin_target_ha = cpmd_ekin_target_ha;
                    ch.tau_e_fs = cpmd_tau_e;
                    ch.seed(C_occ);
                    out.push_back(std::move(ch));
                }
                return out;
            };
        cpmd_k_ =
            seed_channel(last_.C_k, last_.C_occ_k, n_occ_a,
                         seed_uks ? "C_k / C_occ_k (alpha)" : "C_k / C_occ_k");
        cpmd_k_beta_.clear();
        if (seed_uks && n_beta > 0)
            cpmd_k_beta_ =
                seed_channel(last_.C_k_beta, last_.C_occ_k_beta, n_beta,
                             "C_k_beta / C_occ_k_beta (beta)");
        // Seed step: the electronic coefficients have not been propagated, so
        // both electronic energies are exactly 0 (a re-`build()` after a warm
        // run must reset the thermostat readback too, not leave it stale).
        last_te_ev_ = 0.0;
        last_etherm_ev_ = 0.0;
        publish_electronic_energies(frame, last_te_ev_, last_etherm_ev_);
    } else if (cpmd_ && !cpmd_->ready()) {
        if (!last_.converged || last_.C.size() == 0)
            throw std::runtime_error(
                "GPWEvaluator: cpmd seed requires a converged SCF with "
                "occupied MO coefficients");
        const int nb = static_cast<int>(last_.C.shape(0));
        if (n_alpha + n_beta > 0) {
            // Collinear UKS: seed α (and β when present) occupied blocks.
            if (n_alpha < 1)
                throw std::runtime_error(
                    "GPWEvaluator: UKS CPMD seed requires n_alpha >= 1 "
                    "(alpha propagator is primary)");
            if (static_cast<int>(last_.C.shape(1)) < n_alpha)
                throw std::runtime_error(
                    "GPWEvaluator: UKS CPMD seed: alpha MO count < n_alpha");
            linalg::RealMatrix Ca =
                xt::zeros<double>({static_cast<std::size_t>(nb),
                                   static_cast<std::size_t>(n_alpha)});
            for (int mu = 0; mu < nb; ++mu)
                for (int i = 0; i < n_alpha; ++i)
                    Ca(mu, i) = last_.C(mu, i);
            cpmd_->seed(Ca);
            if (n_beta > 0) {
                if (!cpmd_beta_)
                    throw std::runtime_error(
                        "GPWEvaluator: UKS CPMD missing beta propagator");
                if (last_.C_beta.size() == 0
                    || static_cast<int>(last_.C_beta.shape(1)) < n_beta)
                    throw std::runtime_error(
                        "GPWEvaluator: UKS CPMD seed requires beta MO "
                        "coefficients");
                linalg::RealMatrix Cb =
                    xt::zeros<double>({static_cast<std::size_t>(nb),
                                       static_cast<std::size_t>(n_beta)});
                for (int mu = 0; mu < nb; ++mu)
                    for (int i = 0; i < n_beta; ++i)
                        Cb(mu, i) = last_.C_beta(mu, i);
                cpmd_beta_->seed(Cb);
            }
        } else {
            if (solver_->n_occ < 1)
                throw std::runtime_error(
                    "GPWEvaluator: RKS CPMD seed requires n_occ >= 1");
            const int n_occ = solver_->n_occ;
            linalg::RealMatrix C_occ =
                xt::zeros<double>({static_cast<std::size_t>(nb),
                                   static_cast<std::size_t>(n_occ)});
            for (int mu = 0; mu < nb; ++mu)
                for (int i = 0; i < n_occ; ++i)
                    C_occ(mu, i) = last_.C(mu, i);
            cpmd_->seed(C_occ);
        }
        // Seed step: see the multi-k branch above — both terms are 0 here.
        last_te_ev_ = 0.0;
        last_etherm_ev_ = 0.0;
        publish_electronic_energies(frame, last_te_ev_, last_etherm_ev_);
    }

    if (compute_stress) {
        // Product default: Hellmann–Feynman fixed-P analytic stress (one SCF
        // already done above + fixed-P energy samples). Optional multi-SCF
        // strain-FD oracle via stress_use_fd. Warm-start from last MOs; skip
        // masked axes (variable-cell relax of constrained boxes).
        const ATV::cpu::linalg::RealMatrix* stress_guess =
            last_.C.size() > 0 ? &last_.C : nullptr;
        last_stress_ha_ = stress_use_fd
                              ? gpw_stress_fd(atoms, *solver_, stress_strain_h,
                                              stress_axes, stress_guess)
                              : gpw_stress(atoms, *solver_, stress_strain_h,
                                           stress_axes, stress_guess);
        add_stress_to_frame(frame, last_stress_ha_);
    }
}

} // namespace gpw
} // namespace cpu
} // namespace ATV
