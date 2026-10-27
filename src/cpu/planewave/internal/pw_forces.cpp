/// @file cpu/planewave/internal/pw_forces.cpp
/// @brief Analytic plane-wave ionic forces (local HF + Ewald + KB nonlocal).

#include "atomiverse/cpu/planewave/internal/pw_forces.h"

#include "atomiverse/core/frame_schema.h"
#include "atomiverse/core/log.h"
#include "atomiverse/core/mat3.h"
#include "atomiverse/core/units.h"
#include "atomiverse/cpu/planewave/internal/davidson.h"
#include "atomiverse/cpu/planewave/internal/density_builder.h"
#include "atomiverse/cpu/planewave/internal/ewald.h"
#include "atomiverse/cpu/planewave/internal/nc_potential.h"
#include "atomiverse/cpu/planewave/internal/poisson.h"
#include "atomiverse/cpu/planewave/internal/pw_setup.h"
#include "atomiverse/cpu/planewave/internal/pw_subspace.h"
#include "atomiverse/cpu/planewave/internal/pw_workspace.h"
#include "atomiverse/cpu/planewave/internal/restricted_scf.h"
#include "atomiverse/cpu/planewave/internal/scf_loop.h"
#include "atomiverse/cpu/planewave/internal/veff_builder.h"
#include "atomiverse/cpu/planewave/internal/xc_builder.h"
#include "atomiverse/cpu/planewave/qe_units.h"
#include "atomiverse/electronic/smearing.h"

#ifdef ATV_HAS_MPI
#include <mpi.h>
#endif

#include <algorithm>
#include <cmath>
#include <complex>
#include <span>
#include <stdexcept>
#include <vector>
#include <xtensor/containers/xarray.hpp>

namespace ATV {
namespace cpu {
namespace pw {

namespace internal {
namespace {

#ifdef ATV_HAS_MPI
MPI_Datatype mpi_real_type() {
    return sizeof(AccReal) == sizeof(double) ? MPI_DOUBLE : MPI_FLOAT;
}
#endif

struct MpiLayout {
    int rank = 0;
    int nprocs = 1;
    bool active = false;
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

std::complex<AccReal> im_l_factor(int l) {
    if (l == 1)
        return {0, -1};
    if (l == 2)
        return {-1, 0};
    return {1, 0};
}

/// Local Hellmann–Feynman forces [Hartree/Bohr] from rho(G), the
/// canonical NC radial form factor and each atomic phase.
void accumulate_local_forces(const PWWorkspace& ws, const GVectorList& gv,
                             std::span<const std::complex<AccReal>> rho,
                             int n_atoms, const std::vector<AccReal>& nc_vg,
                             std::vector<Vec3>& F_ha_bohr) {
    const AccReal omega = ws.omega;

    for (int ig = 0; ig < ws.n_g; ++ig) {
        AccReal g2 = gv.g[ig].norm_squared();
        if (g2 < AccReal(1e-16))
            continue;
        AccReal Gx = gv.g[ig][0], Gy = gv.g[ig][1], Gz = gv.g[ig][2];
        AccReal rho_r = rho[ig].real();
        AccReal rho_i = rho[ig].imag();

        for (int ia = 0; ia < n_atoms; ++ia) {
            AccReal ax = ws.atom_pos_bohr[static_cast<std::size_t>(ia)][0];
            AccReal ay = ws.atom_pos_bohr[static_cast<std::size_t>(ia)][1];
            AccReal az = ws.atom_pos_bohr[static_cast<std::size_t>(ia)][2];
            AccReal phase = Gx * ax + Gy * ay + Gz * az;
            AccReal cosph = std::cos(phase);
            AccReal sinph = std::sin(phase);

            const AccReal vg_atom = nc_vg[static_cast<std::size_t>(ig)];
            AccReal scale = omega * vg_atom * (rho_r * sinph + rho_i * cosph);
            F_ha_bohr[static_cast<std::size_t>(ia)] =
                F_ha_bohr[static_cast<std::size_t>(ia)]
                + Vec3{scale * Gx, scale * Gy, scale * Gz};
        }
    }
}

/// Nonlocal KB forces for one k-point (ws.wfc, ws.kb_proj, ws.gvec bound).
///
/// E_nl = Σ_{b,a,lm} w_k f_b D_l |c_lm|^{2} with f_b = `occ_b` (scalar `occ`
/// for fixed closed shell / spin, or per-band metallic occupations in
/// \([0,2]\)), c = Σ_G β* ψ, β ∝ exp(−i q·R).
/// F_Iα = −∂E/∂R_Iα = −Σ 2 D Re(c* ∂c/∂R_α),  ∂c/∂R_α = i Σ q_α β* ψ.
void accumulate_nl_forces_k(const PWWorkspace& ws, AccReal weight, int n_bands,
                            const AccReal* occ_per_band,
                            std::vector<Vec3>& F_ha_bohr) {
    if (ws.kb_proj.empty() || !ws.gvec || !occ_per_band)
        return;

    const int ng = ws.n_g;
    const int nb = std::min(n_bands, static_cast<int>(ws.wfc.shape(0)));
    const AccReal four_pi_over_sqrt_omega =
        AccReal(4.0 * M_PI) / std::sqrt(ws.omega);
    auto& gv = *ws.gvec;

    for (int b = 0; b < nb; ++b) {
        const AccReal f_b = occ_per_band[static_cast<std::size_t>(b)];
        if (std::abs(f_b) < AccReal(1e-14))
            continue;
        const AccReal pref = weight * f_b;
        for (size_t ia = 0; ia < ws.kb_proj.size(); ++ia) {
            const auto& atom_proj = ws.kb_proj[ia];
            AccReal ax = ws.atom_pos_bohr[ia][0];
            AccReal ay = ws.atom_pos_bohr[ia][1];
            AccReal az = ws.atom_pos_bohr[ia][2];

            for (size_t ip = 0; ip < atom_proj.size(); ++ip) {
                const int l = atom_proj[ip].l;
                const AccReal D = atom_proj[ip].d;
                if (std::abs(D) < AccReal(1e-16))
                    continue;
                const int nm = n_m_for_l(l);
                const auto im_l = im_l_factor(l);

                std::vector<std::complex<AccReal>> c_lm(
                    static_cast<std::size_t>(nm), {0, 0});
                // s_α[m] = Σ_G q_α conj(β) ψ  so ∂c/∂R_α = i s_α
                std::vector<std::complex<AccReal>> s_x(nm, {0, 0});
                std::vector<std::complex<AccReal>> s_y(nm, {0, 0});
                std::vector<std::complex<AccReal>> s_z(nm, {0, 0});
                std::vector<AccReal> ylm(static_cast<std::size_t>(nm));

                for (int ig = 0; ig < ng; ++ig) {
                    AccReal fl_val = atom_proj[ip].f_G[ig];
                    if (std::abs(fl_val) < AccReal(1e-30))
                        continue;
                    AccReal qx = gv.g[ig][0] + gv.kvec[0];
                    AccReal qy = gv.g[ig][1] + gv.kvec[1];
                    AccReal qz = gv.g[ig][2] + gv.kvec[2];
                    AccReal qr = std::sqrt(qx * qx + qy * qy + qz * qz);
                    AccReal phase = qx * ax + qy * ay + qz * az;
                    AccReal cos_p = std::cos(phase), sin_p = std::sin(phase);

                    compute_ylm(qx, qy, qz, qr, l, ylm.data());

                    const auto psi = ws.wfc(b, ig);
                    for (int m = 0; m < nm; ++m) {
                        std::complex<AccReal> beta =
                            four_pi_over_sqrt_omega * im_l
                            * ylm[static_cast<std::size_t>(m)] * fl_val
                            * std::complex<AccReal>(cos_p, -sin_p);
                        std::complex<AccReal> beta_star_psi =
                            std::conj(beta) * psi;
                        c_lm[static_cast<std::size_t>(m)] += beta_star_psi;
                        s_x[static_cast<std::size_t>(m)] += qx * beta_star_psi;
                        s_y[static_cast<std::size_t>(m)] += qy * beta_star_psi;
                        s_z[static_cast<std::size_t>(m)] += qz * beta_star_psi;
                    }
                }

                // F_α = −2 D pref Re(c* · i s_α) = 2 D pref Im(c* s_α)
                Vec3 dF = Vec3::zeros();
                for (int m = 0; m < nm; ++m) {
                    auto c = c_lm[static_cast<std::size_t>(m)];
                    auto cstar = std::conj(c);
                    AccReal scale = AccReal(2) * D * pref;
                    dF[0] +=
                        scale
                        * std::imag(cstar * s_x[static_cast<std::size_t>(m)]);
                    dF[1] +=
                        scale
                        * std::imag(cstar * s_y[static_cast<std::size_t>(m)]);
                    dF[2] +=
                        scale
                        * std::imag(cstar * s_z[static_cast<std::size_t>(m)]);
                }
                F_ha_bohr[ia] = F_ha_bohr[ia] + dF;
            }
        }
    }
}

/// Scalar-occupation convenience (fixed closed shell f=2 / spin-orbital f=1).
void accumulate_nl_forces_k(const PWWorkspace& ws, AccReal weight, int n_bands,
                            AccReal occ, std::vector<Vec3>& F_ha_bohr) {
    const int nb = std::max(n_bands, 0);
    std::vector<AccReal> occ_v(static_cast<std::size_t>(nb), occ);
    accumulate_nl_forces_k(ws, weight, nb, occ_v.data(), F_ha_bohr);
}

} // namespace

std::vector<std::array<double, 3>>
pw_forces_analytic_impl(const cpu::Frame& frame, const PW& method,
                        PWResult* scf_out, PWProblem* solved) {
    const int n = frame.nlocal();
    if (n <= 0)
        throw std::invalid_argument("pw_forces_analytic: frame has no atoms");
    const bool use_smear = (method.smearing != ATV::electronic::Smearing::none);
    if (use_smear && !(method.degauss > 0.0))
        throw std::invalid_argument(
            "pw_forces_analytic: degauss must be > 0 [Ry] when smearing is "
            "enabled");

    internal::validate_potential_frame(frame, *method.potential);
    int n_elec = method.potential->valence() * frame.nlocal();

    if (solved)
        n_elec = solved->n_elec;

    const bool spin = method.spin_polarized;
    int n_alpha = 0;
    int n_beta = 0;
    int n_bands = std::max(n_elec / 2, 1);
    if (spin) {
        if (method.multiplicity < 1)
            throw std::invalid_argument(
                "pw_forces_analytic: multiplicity must be >= 1 when "
                "spin_polarized");
        const int mag = method.multiplicity - 1;
        if ((n_elec + mag) % 2 != 0 || mag > n_elec)
            throw std::invalid_argument(
                "pw_forces_analytic: n_elec and multiplicity have "
                "incompatible parity");
        n_alpha = (n_elec + mag) / 2;
        n_beta = (n_elec - mag) / 2;
        n_bands = std::max(std::max(n_alpha, n_beta), 1);
        if (use_smear) {
            const int n_occ_ceil = std::max((n_elec + 1) / 2, 1);
            n_bands = std::max(
                {n_bands, n_occ_ceil + std::max(method.extra_bands, 0), 1});
        }
    } else if (use_smear) {
        // Empty states near E_F: ceil(N_e/2) + extra_bands (energy twin).
        const int n_occ_ceil = std::max((n_elec + 1) / 2, 1);
        n_bands = n_occ_ceil + std::max(method.extra_bands, 0);
    } else {
        require_closed_shell(n_elec, "pw_forces_analytic");
    }

    PoissonSolver poisson;
    XCBuilder xc;
    VeffBuilder veff;
    DensityBuilder density;

    PWProblem local_problem;
    PWProblem& scf_prob = solved ? *solved : local_problem;
    PWResult res{};
    if (solved) {
        if (!scf_out)
            throw std::logic_error("PW solved state requires its result");
        res = *scf_out;
        xc.prepare(scf_prob.ws);
        veff.prepare(scf_prob.ws);
    } else if (spin && use_smear) {
        res = run_collinear_smear_scf(frame, method, scf_prob, n_bands);
        xc.prepare(scf_prob.ws);
        veff.prepare(scf_prob.ws);
    } else if (spin) {
        // Collinear fixed-occupation UKS (shared helper with bands/Berry).
        res = run_collinear_scf(frame, method, scf_prob, n_alpha, n_beta);
        // Local builders inside run_collinear_scf are gone; prepare ours for
        // the NL V_eff rebuild below.
        xc.prepare(scf_prob.ws);
        veff.prepare(scf_prob.ws);
    } else if (use_smear) {
        DavidsonSolver davidson{n_bands, AccReal(method.tol) * AccReal(0.1),
                                40};
        res =
            run_scf_smear_restricted(frame, method, scf_prob, poisson, xc, veff,
                                     density, davidson, n_bands, "forces");
    } else {
        DavidsonSolver davidson{n_bands, AccReal(method.tol) * AccReal(0.1),
                                40};
        res = run_scf_fixed_occ(frame, method, scf_prob, poisson, xc, veff,
                                density, davidson, "forces");
    }
    if (!res.converged) {
        ATV_WARN("[pw] pw_forces_analytic: SCF not converged after {} iters; "
                 "force Hamiltonian uses last density",
                 res.iterations);
    }
    // Keep the energy of the accepted SCF. The force assembly below is one
    // frozen-Hamiltonian band solve, not another self-consistency loop; its
    // local and nonlocal HF terms use the same orbital density. At finite
    // SCF tolerance this does not imply an exact total-energy gradient.
    if (scf_out != nullptr)
        *scf_out = res;

    PWWorkspace& ws = scf_prob.ws;
    ws.gvec = &scf_prob.gv0;
    ws.n_g = scf_prob.gv0.count();

    // Fresh Davidson for the force-path band re-solve (same size as SCF).
    DavidsonSolver davidson{n_bands, AccReal(method.tol) * AccReal(0.1), 40};
    davidson.prepare(ws);

    std::vector<Vec3> F_ha(static_cast<std::size_t>(n), Vec3::zeros());

    // The same immutable radial snapshot supplies SCF, local forces and ions.
    const NcPotentialData& nc = *scf_prob.potential;
    const std::vector<AccReal> ion_charges(static_cast<std::size_t>(n),
                                           static_cast<AccReal>(nc.z_val));
    const auto& ewald_charges = ion_charges;

    // ── Ewald ──
    std::vector<Vec3> pos_ang(static_cast<std::size_t>(n));
    const StorageReal* sx = frame.span(kAtom_x);
    const StorageReal* sy = frame.span(kAtom_y);
    const StorageReal* sz = frame.span(kAtom_z);
    for (int i = 0; i < n; ++i)
        pos_ang[static_cast<std::size_t>(i)] = {static_cast<AccReal>(sx[i]),
                                                static_cast<AccReal>(sy[i]),
                                                static_cast<AccReal>(sz[i])};

    const cpu::Box& box_ang = frame.box;
    EwaldSummation ewald(pos_ang, ewald_charges, box_ang);
    const auto& few = ewald.forces_bohr();
    for (int i = 0; i < n; ++i)
        F_ha[static_cast<std::size_t>(i)] = F_ha[static_cast<std::size_t>(i)]
                                            + few[static_cast<std::size_t>(i)];

    // Transform the same radial snapshot used by the Hamiltonian directly;
    // structure-factor extinction shells retain their per-atom derivatives.
    // Local HF is accumulated after the common orbital density is available.
    const std::vector<AccReal> nc_vg =
        nc_vloc_to_gspace(nc, scf_prob.gv0, ws.omega);

    // ── Common fresh band state for local and nonlocal HF ──
    // Purely local potentials need the same consistent orbital density even
    // though their KB contribution is zero. Keep the mixed SCF rho untouched.
    {
        std::vector<Vec3> F_nl(static_cast<std::size_t>(n), Vec3::zeros());

        const auto layout = mpi_layout();
        const int n_kpts = static_cast<int>(scf_prob.kg.kpoints.size());
        auto* saved_gvec = ws.gvec;
        density.prepare(ws);
        density.begin(ws);

        if (spin && use_smear) {
            poisson.step(ws);
            xc.step_spin(ws, method.xc);

            ATV::electronic::Smearing smear_kind =
                ATV::electronic::Smearing::fermi_dirac;
            if (method.smearing == ATV::electronic::Smearing::methfessel_paxton)
                smear_kind = ATV::electronic::Smearing::methfessel_paxton;
            else if (method.smearing == ATV::electronic::Smearing::gaussian)
                smear_kind = ATV::electronic::Smearing::gaussian;
            const AccReal degauss_ha =
                static_cast<AccReal>(method.degauss) * AccReal(0.5);
            const int mp_order = method.mp_order;

            std::vector<std::vector<AccReal>> evals_sk(
                static_cast<std::size_t>(2 * n_kpts));
            std::vector<xt::xarray<std::complex<AccReal>>> wfc_sk(
                static_cast<std::size_t>(2 * n_kpts));
            for (int s = 0; s < 2; ++s) {
                for (int ig = 0; ig < scf_prob.gv0.count(); ++ig)
                    ws.V_xc_G[ig] = ws.V_xc_spin_G[s][ig];
                ws.gvec = saved_gvec;
                ws.n_g = scf_prob.gv0.count();
                veff.step(ws);
                for (int ik = layout.rank; ik < n_kpts; ik += layout.nprocs) {
                    auto& gvk = scf_prob.gv_k[static_cast<std::size_t>(ik)];
                    ws.gvec = &gvk;
                    ws.n_g = gvk.count();
                    ws.T_G.resize({static_cast<std::size_t>(ws.n_g)});
                    for (int ig = 0; ig < ws.n_g; ++ig)
                        ws.T_G[ig] = gvk.ekin[ig];
                    if (scf_prob.kb_proj_k.empty())
                        ws.kb_proj.clear();
                    else
                        ws.kb_proj =
                            scf_prob.kb_proj_k[static_cast<std::size_t>(ik)];
                    if (!scf_prob.atom_wfc_k.empty())
                        ws.atom_wfc =
                            scf_prob.atom_wfc_k[static_cast<std::size_t>(ik)];
                    davidson.step(ws);
                    evals_sk[static_cast<std::size_t>(s * n_kpts + ik)].assign(
                        ws.eigenvalues.begin(), ws.eigenvalues.end());
                    wfc_sk[static_cast<std::size_t>(s * n_kpts + ik)] = ws.wfc;
                    ws.gvec = saved_gvec;
                    ws.n_g = scf_prob.gv0.count();
                }
            }

#ifdef ATV_HAS_MPI
            if (layout.active) {
                std::vector<AccReal> pack(
                    static_cast<std::size_t>(2 * n_kpts * n_bands), AccReal(0));
                for (int spin_index = 0; spin_index < 2; ++spin_index)
                    for (int ik = layout.rank; ik < n_kpts;
                         ik += layout.nprocs) {
                        const auto& ev = evals_sk[static_cast<std::size_t>(
                            spin_index * n_kpts + ik)];
                        for (int band = 0;
                             band < n_bands
                             && band < static_cast<int>(ev.size());
                             ++band)
                            pack[static_cast<std::size_t>(
                                (spin_index * n_kpts + ik) * n_bands + band)] =
                                ev[static_cast<std::size_t>(band)];
                    }
                MPI_Allreduce(MPI_IN_PLACE, pack.data(), 2 * n_kpts * n_bands,
                              mpi_real_type(), MPI_SUM, MPI_COMM_WORLD);
                for (int spin_index = 0; spin_index < 2; ++spin_index)
                    for (int ik = 0; ik < n_kpts; ++ik) {
                        auto& ev = evals_sk[static_cast<std::size_t>(
                            spin_index * n_kpts + ik)];
                        ev.resize(static_cast<std::size_t>(n_bands));
                        for (int band = 0; band < n_bands; ++band)
                            ev[static_cast<std::size_t>(band)] =
                                pack[static_cast<std::size_t>(
                                    (spin_index * n_kpts + ik) * n_bands
                                    + band)];
                    }
            }
#endif

            std::vector<AccReal> all_evals;
            std::vector<AccReal> all_weights;
            for (int s = 0; s < 2; ++s) {
                for (int ik = 0; ik < n_kpts; ++ik) {
                    const AccReal w =
                        scf_prob.kg.weights[static_cast<std::size_t>(ik)];
                    for (AccReal e :
                         evals_sk[static_cast<std::size_t>(s * n_kpts + ik)]) {
                        all_evals.push_back(e);
                        all_weights.push_back(w);
                    }
                }
            }
            const AccReal mu_ha =
                find_fermi_level(all_evals, all_weights, AccReal(n_elec),
                                 degauss_ha, smear_kind, mp_order, AccReal(1));

            for (int s = 0; s < 2; ++s) {
                for (int ik = layout.rank; ik < n_kpts; ik += layout.nprocs) {
                    auto& gvk = scf_prob.gv_k[static_cast<std::size_t>(ik)];
                    ws.gvec = &gvk;
                    ws.n_g = gvk.count();
                    ws.wfc = wfc_sk[static_cast<std::size_t>(s * n_kpts + ik)];
                    if (scf_prob.kb_proj_k.empty())
                        ws.kb_proj.clear();
                    else
                        ws.kb_proj =
                            scf_prob.kb_proj_k[static_cast<std::size_t>(ik)];
                    const auto& ev =
                        evals_sk[static_cast<std::size_t>(s * n_kpts + ik)];
                    const int nb = static_cast<int>(ev.size());
                    ws.eigenvalues.resize({ev.size()});
                    std::copy(ev.begin(), ev.end(), ws.eigenvalues.begin());
                    std::vector<AccReal> occ(static_cast<std::size_t>(nb));
                    for (int b = 0; b < nb; ++b)
                        occ[static_cast<std::size_t>(b)] = smear_f_spinor(
                            ev[static_cast<std::size_t>(b)], mu_ha, degauss_ha,
                            smear_kind, mp_order);
                    const AccReal weight =
                        scf_prob.kg.weights[static_cast<std::size_t>(ik)];
                    density.accumulate_kpoint(ws, weight, nb, occ.data());
                    accumulate_nl_forces_k(ws, weight, nb, occ.data(), F_nl);
                    ws.gvec = saved_gvec;
                    ws.n_g = scf_prob.gv0.count();
                }
            }
        } else if (spin) {
            // Rebuild Hartree from total ρ + spin XC; per-channel V_eff.
            // Rebuild each channel's potential from the final mixed density,
            // independently of the last beta-channel V_eff scratch.
            poisson.step(ws);
            xc.step_spin(ws, method.xc);
            const int n_occ[2] = {n_alpha, n_beta};
            for (int s = 0; s < 2; ++s) {
                if (n_occ[s] <= 0)
                    continue;
                for (int ig = 0; ig < ws.n_g; ++ig)
                    ws.V_xc_G[ig] = ws.V_xc_spin_G[s][ig];
                veff.step(ws);

                for (int ik = layout.rank; ik < n_kpts; ik += layout.nprocs) {
                    auto& gvk = scf_prob.gv_k[static_cast<std::size_t>(ik)];
                    ws.gvec = &gvk;
                    ws.n_g = gvk.count();
                    ws.T_G.resize({static_cast<std::size_t>(ws.n_g)});
                    for (int ig = 0; ig < ws.n_g; ++ig)
                        ws.T_G[ig] = gvk.ekin[ig];

                    if (scf_prob.kb_proj_k.empty())
                        ws.kb_proj.clear();
                    else
                        ws.kb_proj =
                            scf_prob.kb_proj_k[static_cast<std::size_t>(ik)];
                    if (!scf_prob.atom_wfc_k.empty())
                        ws.atom_wfc =
                            scf_prob.atom_wfc_k[static_cast<std::size_t>(ik)];

                    davidson.step(ws);

                    const AccReal weight =
                        scf_prob.kg.weights[static_cast<std::size_t>(ik)];
                    density.accumulate_kpoint(ws, weight, n_occ[s], AccReal(1));
                    accumulate_nl_forces_k(ws, weight, n_occ[s], AccReal(1),
                                           F_nl);

                    ws.gvec = saved_gvec;
                    ws.n_g = scf_prob.gv0.count();
                }
            }
        } else if (use_smear) {
            // Restricted metallic smear: rebuild V_eff, re-solve bands, fill
            // fractional occupations from a shared μ, then NL with f_b∈[0,2].
            poisson.step(ws);
            xc.step(ws, method.xc);
            veff.step(ws);

            ATV::electronic::Smearing smear_kind =
                ATV::electronic::Smearing::fermi_dirac;
            if (method.smearing == ATV::electronic::Smearing::methfessel_paxton)
                smear_kind = ATV::electronic::Smearing::methfessel_paxton;
            else if (method.smearing == ATV::electronic::Smearing::gaussian)
                smear_kind = ATV::electronic::Smearing::gaussian;
            const AccReal degauss_ha =
                static_cast<AccReal>(method.degauss) * AccReal(0.5);
            const int mp_order = method.mp_order;

            std::vector<std::vector<AccReal>> evals_k(
                static_cast<std::size_t>(n_kpts));
            std::vector<xt::xarray<std::complex<AccReal>>> wfc_k(
                static_cast<std::size_t>(n_kpts));

            for (int ik = layout.rank; ik < n_kpts; ik += layout.nprocs) {
                auto& gvk = scf_prob.gv_k[static_cast<std::size_t>(ik)];
                ws.gvec = &gvk;
                ws.n_g = gvk.count();
                ws.T_G.resize({static_cast<std::size_t>(ws.n_g)});
                for (int ig = 0; ig < ws.n_g; ++ig)
                    ws.T_G[ig] = gvk.ekin[ig];
                if (scf_prob.kb_proj_k.empty())
                    ws.kb_proj.clear();
                else
                    ws.kb_proj =
                        scf_prob.kb_proj_k[static_cast<std::size_t>(ik)];
                if (!scf_prob.atom_wfc_k.empty())
                    ws.atom_wfc =
                        scf_prob.atom_wfc_k[static_cast<std::size_t>(ik)];

                davidson.step(ws);

                evals_k[static_cast<std::size_t>(ik)].assign(
                    ws.eigenvalues.begin(),
                    ws.eigenvalues.begin()
                        + std::min(n_bands,
                                   static_cast<int>(ws.eigenvalues.size())));
                if (static_cast<int>(
                        evals_k[static_cast<std::size_t>(ik)].size())
                    > n_bands)
                    evals_k[static_cast<std::size_t>(ik)].resize(
                        static_cast<std::size_t>(n_bands));
                wfc_k[static_cast<std::size_t>(ik)] = ws.wfc;

                ws.gvec = saved_gvec;
                ws.n_g = scf_prob.gv0.count();
            }

#ifdef ATV_HAS_MPI
            if (layout.active) {
                std::vector<AccReal> pack(
                    static_cast<std::size_t>(n_kpts * n_bands), AccReal(0));
                for (int ik = layout.rank; ik < n_kpts; ik += layout.nprocs) {
                    const auto& ev = evals_k[static_cast<std::size_t>(ik)];
                    for (int b = 0;
                         b < n_bands && b < static_cast<int>(ev.size()); ++b)
                        pack[static_cast<std::size_t>(ik * n_bands + b)] =
                            ev[static_cast<std::size_t>(b)];
                }
                MPI_Allreduce(MPI_IN_PLACE, pack.data(), n_kpts * n_bands,
                              mpi_real_type(), MPI_SUM, MPI_COMM_WORLD);
                for (int ik = 0; ik < n_kpts; ++ik) {
                    evals_k[static_cast<std::size_t>(ik)].resize(
                        static_cast<std::size_t>(n_bands));
                    for (int b = 0; b < n_bands; ++b)
                        evals_k[static_cast<std::size_t>(ik)][b] =
                            pack[static_cast<std::size_t>(ik * n_bands + b)];
                }
            }
#endif

            std::vector<AccReal> all_evals;
            std::vector<AccReal> all_weights;
            all_evals.reserve(static_cast<std::size_t>(n_kpts * n_bands));
            all_weights.reserve(static_cast<std::size_t>(n_kpts * n_bands));
            for (int ik = 0; ik < n_kpts; ++ik) {
                const AccReal w =
                    scf_prob.kg.weights[static_cast<std::size_t>(ik)];
                const auto& ev = evals_k[static_cast<std::size_t>(ik)];
                for (int b = 0; b < n_bands && b < static_cast<int>(ev.size());
                     ++b) {
                    all_evals.push_back(ev[static_cast<std::size_t>(b)]);
                    all_weights.push_back(w);
                }
            }
            const AccReal mu_ha =
                find_fermi_level(all_evals, all_weights, AccReal(n_elec),
                                 degauss_ha, smear_kind, mp_order);

            for (int ik = layout.rank; ik < n_kpts; ik += layout.nprocs) {
                auto& gvk = scf_prob.gv_k[static_cast<std::size_t>(ik)];
                ws.gvec = &gvk;
                ws.n_g = gvk.count();
                if (scf_prob.kb_proj_k.empty())
                    ws.kb_proj.clear();
                else
                    ws.kb_proj =
                        scf_prob.kb_proj_k[static_cast<std::size_t>(ik)];
                ws.wfc = wfc_k[static_cast<std::size_t>(ik)];

                const auto& ev = evals_k[static_cast<std::size_t>(ik)];
                const int nb = static_cast<int>(ev.size());
                std::vector<AccReal> occ(static_cast<std::size_t>(nb));
                for (int b = 0; b < nb; ++b)
                    occ[static_cast<std::size_t>(b)] =
                        smear_f_band(ev[static_cast<std::size_t>(b)], mu_ha,
                                     degauss_ha, smear_kind, mp_order);

                const AccReal weight =
                    scf_prob.kg.weights[static_cast<std::size_t>(ik)];
                ws.eigenvalues.resize({ev.size()});
                std::copy(ev.begin(), ev.end(), ws.eigenvalues.begin());
                density.accumulate_kpoint(ws, weight, nb, occ.data());
                accumulate_nl_forces_k(ws, weight, nb, occ.data(), F_nl);

                ws.gvec = saved_gvec;
                ws.n_g = scf_prob.gv0.count();
            }
        } else {
            // Rebuild V_eff from the final mixed density with the SAME XC the
            // SCF used, then obtain the common orbital state for both HF terms.
            poisson.step(ws);
            xc.step(ws, method.xc);
            veff.step(ws);

            for (int ik = layout.rank; ik < n_kpts; ik += layout.nprocs) {
                auto& gvk = scf_prob.gv_k[static_cast<std::size_t>(ik)];
                ws.gvec = &gvk;
                ws.n_g = gvk.count();
                ws.T_G.resize({static_cast<std::size_t>(ws.n_g)});
                for (int ig = 0; ig < ws.n_g; ++ig)
                    ws.T_G[ig] = gvk.ekin[ig];

                if (scf_prob.kb_proj_k.empty())
                    ws.kb_proj.clear();
                else
                    ws.kb_proj =
                        scf_prob.kb_proj_k[static_cast<std::size_t>(ik)];
                if (!scf_prob.atom_wfc_k.empty())
                    ws.atom_wfc =
                        scf_prob.atom_wfc_k[static_cast<std::size_t>(ik)];

                davidson.step(ws);

                const AccReal weight =
                    scf_prob.kg.weights[static_cast<std::size_t>(ik)];
                density.accumulate_kpoint(ws, weight, n_bands, AccReal(2));
                accumulate_nl_forces_k(ws, weight, n_bands, AccReal(2), F_nl);

                ws.gvec = saved_gvec;
                ws.n_g = scf_prob.gv0.count();
            }
        }

        ws.gvec = saved_gvec;
        ws.n_g = scf_prob.gv0.count();
#ifdef ATV_HAS_MPI
        if (layout.active)
            MPI_Allreduce(MPI_IN_PLACE, density.accumulation(), ws.nr,
                          mpi_real_type(), MPI_SUM, MPI_COMM_WORLD);
#endif
        xt::xarray<std::complex<AccReal>> orbital_rho;
        density.extract(ws, orbital_rho);
        accumulate_local_forces(ws, scf_prob.gv0,
                                {orbital_rho.data(), orbital_rho.size()}, n,
                                nc_vg, F_ha);

#ifdef ATV_HAS_MPI
        if (layout.active) {
            std::vector<AccReal> buf(static_cast<std::size_t>(n * 3));
            for (int i = 0; i < n; ++i) {
                buf[static_cast<std::size_t>(3 * i + 0)] =
                    F_nl[static_cast<std::size_t>(i)][0];
                buf[static_cast<std::size_t>(3 * i + 1)] =
                    F_nl[static_cast<std::size_t>(i)][1];
                buf[static_cast<std::size_t>(3 * i + 2)] =
                    F_nl[static_cast<std::size_t>(i)][2];
            }
            MPI_Allreduce(MPI_IN_PLACE, buf.data(), n * 3, mpi_real_type(),
                          MPI_SUM, MPI_COMM_WORLD);
            for (int i = 0; i < n; ++i) {
                F_nl[static_cast<std::size_t>(i)] = {
                    buf[static_cast<std::size_t>(3 * i + 0)],
                    buf[static_cast<std::size_t>(3 * i + 1)],
                    buf[static_cast<std::size_t>(3 * i + 2)]};
            }
        }
#endif

        for (int i = 0; i < n; ++i)
            F_ha[static_cast<std::size_t>(i)] =
                F_ha[static_cast<std::size_t>(i)]
                + F_nl[static_cast<std::size_t>(i)];
    }

    const double ha_per_bohr_to_ev_per_ang = units::convert(
        1.0, units::Force::HartreePerBohr, units::Force::eVPerAngstrom);

    std::vector<std::array<double, 3>> out(static_cast<std::size_t>(n));
    for (int i = 0; i < n; ++i) {
        out[static_cast<std::size_t>(i)] = {
            static_cast<double>(F_ha[static_cast<std::size_t>(i)][0])
                * ha_per_bohr_to_ev_per_ang,
            static_cast<double>(F_ha[static_cast<std::size_t>(i)][1])
                * ha_per_bohr_to_ev_per_ang,
            static_cast<double>(F_ha[static_cast<std::size_t>(i)][2])
                * ha_per_bohr_to_ev_per_ang};
    }
    return out;
}

} // namespace internal
} // namespace pw
} // namespace cpu
} // namespace ATV
