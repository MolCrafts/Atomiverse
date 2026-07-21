/// @file cpu/pw/internal/pw_forces.cpp
/// @brief Analytic plane-wave ionic forces (local HF + Ewald + KB nonlocal).

#include "atomiverse/cpu/pw/internal/pw_forces.h"

#include "atomiverse/core/box.h"
#include "atomiverse/core/frame_schema.h"
#include "atomiverse/core/log.h"
#include "atomiverse/core/mat3.h"
#include "atomiverse/core/units.h"
#include "atomiverse/cpu/pw/internal/convergence.h"
#include "atomiverse/cpu/pw/internal/davidson.h"
#include "atomiverse/cpu/pw/internal/density_builder.h"
#include "atomiverse/cpu/pw/internal/ewald.h"
#include "atomiverse/cpu/pw/internal/poisson.h"
#include "atomiverse/cpu/pw/internal/pw_setup.h"
#include "atomiverse/cpu/pw/internal/pw_subspace.h"
#include "atomiverse/cpu/pw/internal/pw_workspace.h"
#include "atomiverse/cpu/pw/internal/veff_builder.h"
#include "atomiverse/cpu/pw/internal/xc_builder.h"
#include "atomiverse/cpu/pw/qe_units.h"

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

/// Local Hellmann–Feynman forces [Hartree/Bohr] from ρ(G) and atomic form
/// factors. Toy: v_I(G) = −4π Z_I /(Ω G²). UPF: shared radial v(G) × phase.
void accumulate_local_forces(const PWWorkspace& ws, const GVectorList& gv,
                             const std::vector<AccReal>& charges,
                             const std::vector<AccReal>* upf_vg,
                             std::vector<Vec3>& F_ha_bohr) {
    const int n_atoms = static_cast<int>(charges.size());
    const AccReal omega = ws.omega;
    const AccReal four_pi = AccReal(4.0 * M_PI);

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

            AccReal vg_atom;
            if (upf_vg) {
                vg_atom = (*upf_vg)[static_cast<std::size_t>(ig)];
            } else {
                vg_atom = -four_pi * charges[static_cast<std::size_t>(ia)]
                          / (omega * g2);
            }
            AccReal scale = omega * vg_atom * (rho_r * sinph + rho_i * cosph);
            F_ha_bohr[static_cast<std::size_t>(ia)] =
                F_ha_bohr[static_cast<std::size_t>(ia)]
                + Vec3{scale * Gx, scale * Gy, scale * Gz};
        }
    }
}

/// Nonlocal KB forces for one k-point (ws.wfc, ws.kb_proj, ws.gvec bound).
///
/// E_nl = Σ_{b,a,lm} w_k f_b D_l |c_lm|^{2} with f_b = 2 (closed shell),
/// c = Σ_G β* ψ, β ∝ exp(−i q·R).
/// F_Iα = −∂E/∂R_Iα = −Σ 2 D Re(c* ∂c/∂R_α),  ∂c/∂R_α = i Σ q_α β* ψ.
void accumulate_nl_forces_k(const PWWorkspace& ws, AccReal weight, int n_bands,
                            std::vector<Vec3>& F_ha_bohr) {
    if (ws.kb_proj.empty() || !ws.gvec)
        return;

    const int ng = ws.n_g;
    const int nb = std::min(n_bands, static_cast<int>(ws.wfc.shape(0)));
    const AccReal four_pi_over_sqrt_omega =
        AccReal(4.0 * M_PI) / std::sqrt(ws.omega);
    const AccReal occ = AccReal(2); // closed-shell band occupation
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

                    std::vector<AccReal> ylm(static_cast<std::size_t>(nm));
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

/// Run SCF; on exit `prob` holds final density on the Γ sphere.
PWResult run_scf(const cpu::Frame& frame, const PWConfig& cfg, PWProblem& prob,
                 PoissonSolver& poisson, XCBuilder& xc, VeffBuilder& veff,
                 DensityBuilder& density, DavidsonSolver& davidson) {
    PWResult result{};
    const auto layout = mpi_layout();
    const bool log_rank0 = (layout.rank == 0);

    setup_pw_problem(frame, cfg.upf_path, cfg.ecut_ry, cfg.k_mesh, prob);
    PWWorkspace& ws = prob.ws;
    auto& gv0 = prob.gv0;
    const int n_kpts = static_cast<int>(prob.kg.kpoints.size());
    const int n_bands = prob.n_bands;
    const AccReal vol_bohr = prob.vol_bohr;
    ws.mix_beta = AccReal(cfg.mix_beta);
    ws.pulay_depth = cfg.pulay_depth;

    xc.prepare(ws);
    veff.prepare(ws);
    density.prepare(ws);
    density.reset(ws);
    davidson.prepare(ws);
    ConvergenceCheck convergence{AccReal(cfg.tol), 2};

    bool converged = false;
    for (int iter = 0; iter < cfg.max_iter; ++iter) {
        poisson.step(ws);
        xc.step(ws);
        veff.step(ws);

        density.begin(ws);
        AccReal e_band = 0;
        auto* saved_gvec = ws.gvec;
        for (int ik = layout.rank; ik < n_kpts; ik += layout.nprocs) {
            auto& gvk = prob.gv_k[static_cast<std::size_t>(ik)];
            ws.gvec = &gvk;
            ws.n_g = gvk.count();
            ws.T_G.resize({static_cast<std::size_t>(ws.n_g)});
            for (int ig = 0; ig < ws.n_g; ++ig)
                ws.T_G[ig] = gvk.ekin[ig];

            if (!prob.kb_proj_k.empty())
                ws.kb_proj = prob.kb_proj_k[static_cast<std::size_t>(ik)];
            if (!prob.atom_wfc_k.empty())
                ws.atom_wfc = prob.atom_wfc_k[static_cast<std::size_t>(ik)];

            davidson.step(ws);

            const AccReal weight =
                prob.kg.weights[static_cast<std::size_t>(ik)];
            for (int b = 0; b < n_bands; ++b)
                e_band += weight * ws.eigenvalues[b] * AccReal(2);

            density.accumulate_kpoint(ws, weight, n_bands);

            ws.gvec = saved_gvec;
            ws.n_g = gv0.count();
        }

#ifdef ATV_HAS_MPI
        if (layout.active) {
            MPI_Allreduce(MPI_IN_PLACE, density.accumulation(), ws.nr,
                          mpi_real_type(), MPI_SUM, MPI_COMM_WORLD);
            MPI_Allreduce(MPI_IN_PLACE, &e_band, 1, mpi_real_type(), MPI_SUM,
                          MPI_COMM_WORLD);
        }
#endif

        density.finish(ws);
        converged = convergence.step(ws);
        result.iterations = iter + 1;

        const auto terms =
            iteration_energy(ws, e_band, prob.ew_total, vol_bohr);
        result.energy_ha = static_cast<double>(terms.total);
        result.energy = result.energy_ha / qe_units::rydberg_to_hartree;
        log_iteration_energy(terms, ws, prob.ew_total, iter);

        if (converged) {
            result.converged = true;
            if (log_rank0)
                ATV_INFO("[pw] forces-scf converged iter={} energy_ry={:.8f}",
                         iter + 1, result.energy);
            break;
        }
    }

    if (!result.converged && log_rank0)
        ATV_WARN("[pw] forces-scf not converged after {} iters", cfg.max_iter);
    return result;
}

} // namespace

std::vector<std::array<double, 3>>
pw_forces_analytic_impl(const cpu::Frame& frame, const PWConfig& cfg) {
    const int n = frame.nlocal();
    if (n <= 0)
        throw std::invalid_argument("pw_forces_analytic: frame has no atoms");

    int n_elec = 0;
    {
        const int* types = frame.span(kAtom_type);
        for (int i = 0; i < n; ++i)
            n_elec += valence_from_z(types[i]);
    }
    const int n_bands = std::max(n_elec / 2, 1);

    PoissonSolver poisson;
    XCBuilder xc;
    VeffBuilder veff;
    DensityBuilder density{AccReal(cfg.mix_beta), cfg.pulay_depth};
    DavidsonSolver davidson{n_bands, AccReal(cfg.tol) * AccReal(0.1), 40};

    PWProblem scf_prob;
    const PWResult res =
        run_scf(frame, cfg, scf_prob, poisson, xc, veff, density, davidson);
    if (!res.converged) {
        ATV_WARN("[pw] pw_forces_analytic: SCF not converged after {} iters; "
                 "forces use last density",
                 res.iterations);
    }

    PWWorkspace& ws = scf_prob.ws;
    ws.gvec = &scf_prob.gv0;
    ws.n_g = scf_prob.gv0.count();

    std::vector<Vec3> F_ha(static_cast<std::size_t>(n), Vec3::zeros());

    std::vector<AccReal> valence_charges, nuclear_Z;
    valence_charges.reserve(static_cast<std::size_t>(n));
    nuclear_Z.reserve(static_cast<std::size_t>(n));
    const int* types = frame.span(kAtom_type);
    for (int i = 0; i < n; ++i) {
        int z = types[i];
        nuclear_Z.push_back(static_cast<AccReal>(z));
        valence_charges.push_back(static_cast<AccReal>(valence_from_z(z)));
    }
    const bool has_upf = !cfg.upf_path.empty();
    const auto& ewald_charges = has_upf ? valence_charges : nuclear_Z;
    const auto& loc_charges = valence_charges;

    // ── Ewald ──
    std::vector<Vec3> pos_ang(static_cast<std::size_t>(n));
    const StorageReal* sx = frame.span(kAtom_x);
    const StorageReal* sy = frame.span(kAtom_y);
    const StorageReal* sz = frame.span(kAtom_z);
    for (int i = 0; i < n; ++i)
        pos_ang[static_cast<std::size_t>(i)] = {static_cast<AccReal>(sx[i]),
                                                static_cast<AccReal>(sy[i]),
                                                static_cast<AccReal>(sz[i])};

    ::ATV::Box box_ang;
    {
        const auto& c = frame.simbox.cell();
        Mat3 cell;
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j)
                cell(i, j) = c(i, j);
        box_ang.cell = cell;
    }
    AccReal ecut_ht = static_cast<AccReal>(cfg.ecut_ry) * AccReal(0.5);
    EwaldSummation ewald(pos_ang, ewald_charges, box_ang, ecut_ht);
    const auto& few = ewald.forces_bohr();
    for (int i = 0; i < n; ++i)
        F_ha[static_cast<std::size_t>(i)] = F_ha[static_cast<std::size_t>(i)]
                                            + few[static_cast<std::size_t>(i)];

    // ── Local HF ──
    std::vector<AccReal> upf_vg;
    const std::vector<AccReal>* upf_ptr = nullptr;
    if (has_upf) {
        const auto& gv = scf_prob.gv0;
        const auto& pos = ws.atom_pos_bohr;
        upf_vg.resize(static_cast<std::size_t>(ws.n_g));
        for (int ig = 0; ig < ws.n_g; ++ig) {
            AccReal s_re = 0, s_im = 0;
            for (size_t ia = 0; ia < pos.size(); ++ia) {
                AccReal phase = gv.g[ig][0] * pos[ia][0]
                                + gv.g[ig][1] * pos[ia][1]
                                + gv.g[ig][2] * pos[ia][2];
                s_re += std::cos(phase);
                s_im -= std::sin(phase);
            }
            AccReal s2 = s_re * s_re + s_im * s_im;
            if (s2 < AccReal(1e-30)
                || gv.g[ig].norm_squared() < AccReal(1e-16)) {
                upf_vg[static_cast<std::size_t>(ig)] = AccReal(0);
            } else {
                std::complex<AccReal> S(s_re, s_im);
                upf_vg[static_cast<std::size_t>(ig)] =
                    (ws.V_loc_G[ig] / S).real();
            }
        }
        upf_ptr = &upf_vg;
    }
    accumulate_local_forces(ws, scf_prob.gv0, loc_charges, upf_ptr, F_ha);

    // ── Nonlocal KB (UPF only) ──
    if (has_upf && !scf_prob.kb_proj_k.empty()) {
        std::vector<Vec3> F_nl(static_cast<std::size_t>(n), Vec3::zeros());

        poisson.step(ws);
        xc.step(ws);
        veff.step(ws);

        const auto layout = mpi_layout();
        const int n_kpts = static_cast<int>(scf_prob.kg.kpoints.size());
        auto* saved_gvec = ws.gvec;

        for (int ik = layout.rank; ik < n_kpts; ik += layout.nprocs) {
            auto& gvk = scf_prob.gv_k[static_cast<std::size_t>(ik)];
            ws.gvec = &gvk;
            ws.n_g = gvk.count();
            ws.T_G.resize({static_cast<std::size_t>(ws.n_g)});
            for (int ig = 0; ig < ws.n_g; ++ig)
                ws.T_G[ig] = gvk.ekin[ig];

            ws.kb_proj = scf_prob.kb_proj_k[static_cast<std::size_t>(ik)];
            if (!scf_prob.atom_wfc_k.empty())
                ws.atom_wfc = scf_prob.atom_wfc_k[static_cast<std::size_t>(ik)];

            davidson.step(ws);

            const AccReal weight =
                scf_prob.kg.weights[static_cast<std::size_t>(ik)];
            accumulate_nl_forces_k(ws, weight, n_bands, F_nl);

            ws.gvec = saved_gvec;
            ws.n_g = scf_prob.gv0.count();
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
