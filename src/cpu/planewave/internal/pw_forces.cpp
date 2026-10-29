/// @file cpu/planewave/internal/pw_forces.cpp
/// @brief Analytic plane-wave ionic forces (local HF + Ewald + KB nonlocal).

#include "atomiverse/cpu/planewave/internal/pw_forces.h"
#include "atomiverse/cpu/planewave/internal/pw_final_state.h"

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
        // The final density and virial retain every nonzero occupation.
        // Apply the same ensemble to its nonlocal force, including tiny tails.
        if (f_b == AccReal(0))
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
                if (D == AccReal(0))
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
                    if (fl_val == AccReal(0))
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
pw_forces_from_bands(const cpu::Frame& frame, const PWProblem& problem,
                     const FinalBandState& bands, const EwaldSummation* ionic) {
    const int n = frame.nlocal();
    const auto& geometry = problem.ws;
    if (n <= 0 || !problem.potential
        || geometry.atom_pos_bohr.size() != static_cast<std::size_t>(n)
        || bands.orbital_density_G.size() != problem.gv0.g.size())
        throw std::invalid_argument(
            "PW band forces: inconsistent geometry/density");

    std::unique_ptr<EwaldSummation> local_ionic;
    if (!ionic) {
        std::vector<Vec3> positions(static_cast<std::size_t>(n));
        for (int i = 0; i < n; ++i)
            positions[i] = {frame.span(kAtom_x)[i], frame.span(kAtom_y)[i],
                            frame.span(kAtom_z)[i]};
        const std::vector<AccReal> charges(static_cast<std::size_t>(n),
                                           problem.potential->z_val);
        local_ionic =
            std::make_unique<EwaldSummation>(positions, charges, frame.box);
        ionic = local_ionic.get();
    }
    if (ionic->forces_bohr().size() != static_cast<std::size_t>(n))
        throw std::invalid_argument(
            "PW band forces: inconsistent ionic geometry");

    // Only the distributed band contribution is reduced. Orbital density and
    // ionic geometry are already replicated on every participating rank.
    std::vector<Vec3> nonlocal(static_cast<std::size_t>(n), Vec3::zeros());
    PWWorkspace ws;
    ws.omega = geometry.omega;
    ws.atom_pos_bohr = geometry.atom_pos_bohr;
    for (const auto& block : bands.local_blocks) {
        if (block.occupations.empty())
            continue;
        if (block.ik < 0 || block.ik >= static_cast<int>(problem.gv_k.size()))
            throw std::invalid_argument("PW band forces: invalid k block");
        const auto& gv = problem.gv_k[block.ik];
        if (block.coefficients.dimension() != 2
            || block.coefficients.shape(0) != block.occupations.size()
            || block.coefficients.shape(1) != gv.g.size())
            throw std::invalid_argument("PW band forces: coefficient shape");
        ws.gvec = &gv;
        ws.n_g = gv.count();
        ws.wfc = block.coefficients;
        ws.kb_proj = problem.kb_proj_k.empty()
                         ? std::vector<std::vector<PWWorkspace::KBProjG>>{}
                         : problem.kb_proj_k.at(block.ik);
        accumulate_nl_forces_k(ws, block.k_weight,
                               static_cast<int>(block.occupations.size()),
                               block.occupations.data(), nonlocal);
    }
#ifdef ATV_HAS_MPI
    const auto mpi = mpi_layout();
    if (mpi.active) {
        std::vector<AccReal> packed(static_cast<std::size_t>(n * 3));
        for (int i = 0; i < n; ++i)
            for (int d = 0; d < 3; ++d)
                packed[3 * i + d] = nonlocal[i][d];
        MPI_Allreduce(MPI_IN_PLACE, packed.data(), n * 3, mpi_real_type(),
                      MPI_SUM, MPI_COMM_WORLD);
        for (int i = 0; i < n; ++i)
            for (int d = 0; d < 3; ++d)
                nonlocal[i][d] = packed[3 * i + d];
    }
#endif
    auto forces = ionic->forces_bohr();
    ws.gvec = &problem.gv0;
    ws.n_g = problem.gv0.count();
    const auto vloc =
        nc_vloc_to_gspace(*problem.potential, problem.gv0, ws.omega);
    accumulate_local_forces(ws, problem.gv0, bands.orbital_density_G, n, vloc,
                            forces);
    const double conversion = units::convert(1.0, units::Force::HartreePerBohr,
                                             units::Force::eVPerAngstrom);
    std::vector<std::array<double, 3>> result(static_cast<std::size_t>(n));
    for (int i = 0; i < n; ++i)
        for (int d = 0; d < 3; ++d)
            result[i][d] = (forces[i][d] + nonlocal[i][d]) * conversion;
    return result;
}

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
    } else if (spin && use_smear) {
        res = run_collinear_smear_scf(frame, method, scf_prob, n_bands);
    } else if (spin) {
        // Collinear fixed-occupation UKS (shared helper with bands/Berry).
        res = run_collinear_scf(frame, method, scf_prob, n_alpha, n_beta);
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

    const auto bands = extract_final_bands(scf_prob, method, res);
    return pw_forces_from_bands(frame, scf_prob, bands);
}

} // namespace internal
} // namespace pw
} // namespace cpu
} // namespace ATV
