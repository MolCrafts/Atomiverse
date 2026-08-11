/// @file cpu/pw/internal/pw_forces.cpp
/// @brief Analytic plane-wave ionic forces (local HF + Ewald + KB nonlocal).

#include "atomiverse/cpu/planewave/internal/pw_forces.h"

#include "atomiverse/core/frame_schema.h"
#include "atomiverse/core/log.h"
#include "atomiverse/core/mat3.h"
#include "atomiverse/core/units.h"
#include "atomiverse/cpu/planewave/internal/davidson.h"
#include "atomiverse/cpu/planewave/internal/density_builder.h"
#include "atomiverse/cpu/planewave/internal/ewald.h"
#include "atomiverse/cpu/planewave/internal/poisson.h"
#include "atomiverse/cpu/planewave/internal/pw_setup.h"
#include "atomiverse/cpu/planewave/internal/pw_subspace.h"
#include "atomiverse/cpu/planewave/internal/pw_workspace.h"
#include "atomiverse/cpu/planewave/internal/restricted_scf.h"
#include "atomiverse/cpu/planewave/internal/scf_loop.h"
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
#include <stdexcept>
#include <vector>

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

/// Local Hellmann–Feynman forces [Hartree/Bohr] from ρ(G) and the atomic
/// form factor: the shared radial v(G) (`upf_vloc_to_gspace`) × phase.
///
/// `v(G)` is a required input. It used to be optional, with a bare-Coulomb
/// −4π Z_v/(Ω G²) fallback when no UPF file was given — the local half of
/// the deleted toy potential. Every run now carries a real pseudopotential
/// (UPF file or analytic GTH), so there is nothing left to fall back to.
///
/// @param n_atoms Atom count (form factor is per-species, not per-atom).
/// @param upf_vg  Radial local form factor v(G) [Hartree], one per G.
void accumulate_local_forces(const PWWorkspace& ws, const GVectorList& gv,
                             int n_atoms,
                             const std::vector<AccReal>& upf_vg,
                             std::vector<Vec3>& F_ha_bohr) {
    const AccReal omega = ws.omega;

    for (int ig = 0; ig < ws.n_g; ++ig) {
        AccReal g2 = gv.g[ig].norm_squared();
        if (g2 < AccReal(1e-16))
            continue;
        AccReal Gx = gv.g[ig][0], Gy = gv.g[ig][1], Gz = gv.g[ig][2];
        AccReal rho_r = ws.rho_G[ig].real();
        AccReal rho_i = ws.rho_G[ig].imag();

        for (int ia = 0; ia < n_atoms; ++ia) {
            AccReal ax = ws.atom_pos_bohr[static_cast<std::size_t>(ia)][0];
            AccReal ay = ws.atom_pos_bohr[static_cast<std::size_t>(ia)][1];
            AccReal az = ws.atom_pos_bohr[static_cast<std::size_t>(ia)][2];
            AccReal phase = Gx * ax + Gy * ay + Gz * az;
            AccReal cosph = std::cos(phase);
            AccReal sinph = std::sin(phase);

            const AccReal vg_atom = upf_vg[static_cast<std::size_t>(ig)];
            AccReal scale = omega * vg_atom * (rho_r * sinph + rho_i * cosph);
            F_ha_bohr[static_cast<std::size_t>(ia)] =
                F_ha_bohr[static_cast<std::size_t>(ia)]
                + Vec3{scale * Gx, scale * Gy, scale * Gz};
        }
    }
}

/// Nonlocal KB forces for one k-point (ws.wfc, ws.kb_proj, ws.gvec bound).
///
/// E_nl = Σ_{b,a,lm} w_k f_b D_l |c_lm|^{2} with f_b = `occ` (2 closed shell,
/// 1 per collinear spin-orbital), c = Σ_G β* ψ, β ∝ exp(−i q·R).
/// F_Iα = −∂E/∂R_Iα = −Σ 2 D Re(c* ∂c/∂R_α),  ∂c/∂R_α = i Σ q_α β* ψ.
void accumulate_nl_forces_k(const PWWorkspace& ws, AccReal weight, int n_bands,
                            AccReal occ, std::vector<Vec3>& F_ha_bohr) {
    if (ws.kb_proj.empty() || !ws.gvec)
        return;

    const int ng = ws.n_g;
    const int nb = std::min(n_bands, static_cast<int>(ws.wfc.shape(0)));
    const AccReal four_pi_over_sqrt_omega =
        AccReal(4.0 * M_PI) / std::sqrt(ws.omega);
    auto& gv = *ws.gvec;

    for (int b = 0; b < nb; ++b) {
        const AccReal pref = weight * occ;
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

} // namespace

std::vector<std::array<double, 3>>
pw_forces_analytic_impl(const cpu::Frame& frame, const PW& method) {
    const int n = frame.nlocal();
    if (n <= 0)
        throw std::invalid_argument("pw_forces_analytic: frame has no atoms");
    if (method.spin_polarized && method.smearing != Smearing::none)
        throw std::invalid_argument(
            "pw_forces_analytic: spin_polarized with metallic smearing has "
            "no force path yet");

    int n_elec = 0;
    {
        const int* types = frame.span(kAtom_type);
        for (int i = 0; i < n; ++i)
            n_elec += valence_from_z(types[i]);
    }

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
    }

    PoissonSolver poisson;
    XCBuilder xc;
    VeffBuilder veff;
    DensityBuilder density;

    PWProblem scf_prob;
    PWResult res{};
    if (spin) {
        // Collinear fixed-occupation UKS (shared helper with bands/Berry).
        res = run_collinear_scf(frame, method, scf_prob, n_alpha, n_beta);
        // Local builders inside run_collinear_scf are gone; prepare ours for
        // the NL V_eff rebuild below.
        xc.prepare(scf_prob.ws);
        veff.prepare(scf_prob.ws);
    } else {
        DavidsonSolver davidson{n_bands, AccReal(method.tol) * AccReal(0.1),
                                40};
        res = run_scf_fixed_occ(frame, method, scf_prob, poisson, xc, veff,
                                density, davidson, "forces");
    }
    if (!res.converged) {
        ATV_WARN("[pw] pw_forces_analytic: SCF not converged after {} iters; "
                 "forces use last density",
                 res.iterations);
    }

    PWWorkspace& ws = scf_prob.ws;
    ws.gvec = &scf_prob.gv0;
    ws.n_g = scf_prob.gv0.count();

    // Fresh Davidson for the force-path band re-solve (same size as SCF).
    DavidsonSolver davidson{n_bands, AccReal(method.tol) * AccReal(0.1), 40};
    davidson.prepare(ws);

    std::vector<Vec3> F_ha(static_cast<std::size_t>(n), Vec3::zeros());

    // The pseudopotential — UPF file or analytic GTH — is the single source
    // of both the ion charge the Ewald sum sees and the local form factor,
    // exactly as in setup_pw_problem(). It used to be read only when a UPF
    // *file* was named: a GTH run took the `else` branches and silently got
    // bare nuclear Z in the Ewald sum, a bare-Coulomb local force and no KB
    // term at all, i.e. the deleted toy potential's forces.
    const UPFData upf =
        load_pseudopotential(frame, method.upf_path, method.gth_name);
    const std::vector<AccReal> ion_charges(static_cast<std::size_t>(n),
                                           static_cast<AccReal>(upf.z_val));
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
    AccReal ecut_ht = static_cast<AccReal>(method.ecut_ry) * AccReal(0.5);
    EwaldSummation ewald(pos_ang, ewald_charges, box_ang, ecut_ht);
    const auto& few = ewald.forces_bohr();
    for (int i = 0; i < n; ++i)
        F_ha[static_cast<std::size_t>(i)] = F_ha[static_cast<std::size_t>(i)]
                                            + few[static_cast<std::size_t>(i)];

    // ── Local HF ── (total ρ = ρ↑+ρ↓ on the collinear path)
    // Per-species radial form factor v(G) from the UPF directly. The
    // historical route divided the assembled V_loc_G by the structure
    // factor S(G) — ill-defined on extinction shells (diamond S(G) = 0
    // with nonzero per-atom force contributions, since ρ(G) does not
    // vanish there) and silently zeroed those shells. Exact per-species
    // v(G) is shared with setup via internal::upf_vloc_to_gspace.
    const std::vector<AccReal> upf_vg =
        upf_vloc_to_gspace(upf, scf_prob.gv0, ws.omega);
    accumulate_local_forces(ws, scf_prob.gv0, n, upf_vg, F_ha);

    // ── Nonlocal KB ── (absent only for a potential with no projectors,
    // e.g. GTH-PBE-q1 for hydrogen, which is purely local.)
    if (!scf_prob.kb_proj_k.empty()) {
        std::vector<Vec3> F_nl(static_cast<std::size_t>(n), Vec3::zeros());

        const auto layout = mpi_layout();
        const int n_kpts = static_cast<int>(scf_prob.kg.kpoints.size());
        auto* saved_gvec = ws.gvec;

        if (spin) {
            // Rebuild Hartree from total ρ + spin XC; per-channel V_eff.
            // run_collinear_scf already left poisson/XC frozen; re-run so
            // NL re-solve matches CUDA (and is independent of last β V_eff).
            for (int ig = 0; ig < ws.n_g; ++ig)
                ws.rho_G[ig] = ws.rho_spin_G[0][ig] + ws.rho_spin_G[1][ig];
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

                    ws.kb_proj =
                        scf_prob.kb_proj_k[static_cast<std::size_t>(ik)];
                    if (!scf_prob.atom_wfc_k.empty())
                        ws.atom_wfc =
                            scf_prob.atom_wfc_k[static_cast<std::size_t>(ik)];

                    davidson.step(ws);

                    const AccReal weight =
                        scf_prob.kg.weights[static_cast<std::size_t>(ik)];
                    accumulate_nl_forces_k(ws, weight, n_occ[s], AccReal(1),
                                           F_nl);

                    ws.gvec = saved_gvec;
                    ws.n_g = scf_prob.gv0.count();
                }
            }
        } else {
            // Rebuild V_eff from the final mixed density with the SAME XC the
            // SCF used. `XCBuilder::step` defaults to PBE — a silent LSDA→PBE
            // swap here made analytic forces disagree with FD (and with the
            // CUDA twin, which rebuilds V_eff from the method's xc).
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

                ws.kb_proj = scf_prob.kb_proj_k[static_cast<std::size_t>(ik)];
                if (!scf_prob.atom_wfc_k.empty())
                    ws.atom_wfc =
                        scf_prob.atom_wfc_k[static_cast<std::size_t>(ik)];

                davidson.step(ws);

                const AccReal weight =
                    scf_prob.kg.weights[static_cast<std::size_t>(ik)];
                accumulate_nl_forces_k(ws, weight, n_bands, AccReal(2), F_nl);

                ws.gvec = saved_gvec;
                ws.n_g = scf_prob.gv0.count();
            }
        }

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
