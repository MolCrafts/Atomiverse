/// @file cpu/planewave/internal/pw_forces.cpp
/// @brief Analytic plane-wave ionic forces (local HF + Ewald + KB nonlocal).

#include "atomiverse/cpu/planewave/internal/pw_forces.h"
#include "atomiverse/cpu/planewave/internal/pw_final_state.h"

#include "atomiverse/core/frame_schema.h"
#include "atomiverse/core/log.h"
#include "atomiverse/core/mat3.h"
#include "atomiverse/core/units.h"
#include "atomiverse/cpu/planewave/internal/ewald.h"
#include "atomiverse/cpu/planewave/internal/nc_potential.h"
#include "atomiverse/cpu/planewave/internal/poisson.h"
#include "atomiverse/cpu/planewave/internal/pw_setup.h"
#include "atomiverse/cpu/planewave/internal/pw_subspace.h"
#include "atomiverse/cpu/planewave/internal/pw_workspace.h"
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
void accumulate_local_forces(std::span<const Vec3> positions,
                             const GVectorList& gv, AccReal omega,
                             std::span<const std::complex<AccReal>> rho,
                             std::span<const std::size_t> atom_species,
                             const std::vector<std::vector<AccReal>>& nc_vg,
                             std::vector<Vec3>& F_ha_bohr) {
    for (int ig = 0; ig < gv.count(); ++ig) {
        AccReal g2 = gv.g[ig].norm_squared();
        if (g2 < AccReal(1e-16))
            continue;
        AccReal Gx = gv.g[ig][0], Gy = gv.g[ig][1], Gz = gv.g[ig][2];
        AccReal rho_r = rho[ig].real();
        AccReal rho_i = rho[ig].imag();

        for (std::size_t ia = 0; ia < atom_species.size(); ++ia) {
            AccReal ax = positions[static_cast<std::size_t>(ia)][0];
            AccReal ay = positions[static_cast<std::size_t>(ia)][1];
            AccReal az = positions[static_cast<std::size_t>(ia)][2];
            AccReal phase = Gx * ax + Gy * ay + Gz * az;
            AccReal cosph = std::cos(phase);
            AccReal sinph = std::sin(phase);

            const AccReal vg_atom = nc_vg[atom_species[ia]][ig];
            AccReal scale = omega * vg_atom * (rho_r * sinph + rho_i * cosph);
            F_ha_bohr[static_cast<std::size_t>(ia)] =
                F_ha_bohr[static_cast<std::size_t>(ia)]
                + Vec3{scale * Gx, scale * Gy, scale * Gz};
        }
    }
}

/// Nonlocal KB forces for one k-point from the accepted coefficients and projector view.
///
/// E_nl = Σ_{b,a,lm} w_k f_b D_l |c_lm|^{2} with f_b = `occ_b` (scalar `occ`
/// for fixed closed shell / spin, or per-band metallic occupations in
/// \([0,2]\)), c = Σ_G β* ψ, β ∝ exp(−i q·R).
/// F_Iα = −∂E/∂R_Iα = −Σ 2 D Re(c* ∂c/∂R_α),  ∂c/∂R_α = i Σ q_α β* ψ.
void accumulate_nl_forces_k(PWProjectorView op, const FinalBandBlock& bands,
                            std::vector<Vec3>& F_ha_bohr) {
    if (op.projectors.empty())
        return;

    const int ng = op.modes.count();
    const int nb = static_cast<int>(bands.occupations.size());
    const AccReal four_pi_over_sqrt_omega =
        AccReal(4.0 * M_PI) / std::sqrt(op.volume);
    auto& gv = op.modes;

    for (int b = 0; b < nb; ++b) {
        const AccReal f_b = bands.occupations[static_cast<std::size_t>(b)];
        // The final density and virial retain every nonzero occupation.
        // Apply the same ensemble to its nonlocal force, including tiny tails.
        if (f_b == AccReal(0))
            continue;
        const AccReal pref = bands.k_weight * f_b;
        for (size_t ia = 0; ia < op.projectors.size(); ++ia) {
            const auto& atom_proj = op.projectors[ia];
            AccReal ax = op.positions[ia][0];
            AccReal ay = op.positions[ia][1];
            AccReal az = op.positions[ia][2];

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

                    const auto psi = bands.coefficients(b, ig);
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
pw_forces_from_bands(const cpu::Frame& frame, const PWIons& ions,
                     const GVectorList& density_modes,
                     std::span<const GVectorList> modes,
                     std::span<const KBProjectors> projectors,
                     const FinalBandState& bands, const EwaldSummation* ionic) {
    const int n = frame.nlocal();
    if (n <= 0 || ions.atom_species.size() != static_cast<std::size_t>(n)
        || ions.positions.size() != static_cast<std::size_t>(n)
        || (!projectors.empty() && projectors.size() != modes.size())
        || bands.orbital_density_G.size() != density_modes.g.size())
        throw std::invalid_argument(
            "PW band forces: inconsistent geometry/density");

    std::unique_ptr<EwaldSummation> local_ionic;
    if (!ionic) {
        std::vector<Vec3> positions(static_cast<std::size_t>(n));
        for (int i = 0; i < n; ++i)
            positions[i] = {frame.span(kAtom_x)[i], frame.span(kAtom_y)[i],
                            frame.span(kAtom_z)[i]};
        local_ionic = std::make_unique<EwaldSummation>(
            positions, ions.ionic_charges, frame.box);
        ionic = local_ionic.get();
    }
    if (ionic->forces_bohr().size() != static_cast<std::size_t>(n))
        throw std::invalid_argument(
            "PW band forces: inconsistent ionic geometry");

    // Only the distributed band contribution is reduced. Orbital density and
    // ionic geometry are already replicated on every participating rank.
    std::vector<Vec3> nonlocal(static_cast<std::size_t>(n), Vec3::zeros());
    for (const auto& block : bands.local_blocks) {
        if (block.occupations.empty())
            continue;
        if (block.ik < 0 || block.ik >= static_cast<int>(modes.size()))
            throw std::invalid_argument("PW band forces: invalid k block");
        const auto& gv = modes[block.ik];
        if (block.coefficients.dimension() != 2
            || block.coefficients.shape(0) != block.occupations.size()
            || block.coefficients.shape(1) != gv.g.size())
            throw std::invalid_argument("PW band forces: coefficient shape");
        const KBProjectorView kb =
            projectors.empty() ? KBProjectorView{} : projectors[block.ik];
        accumulate_nl_forces_k({gv, kb, ions.positions, ions.vol_bohr}, block,
                               nonlocal);
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
    const auto vloc =
        nc_vloc_to_gspace(ions.species, density_modes, ions.vol_bohr);
    accumulate_local_forces(ions.positions, density_modes, ions.vol_bohr,
                            bands.orbital_density_G, ions.atom_species, vloc,
                            forces);
    const double conversion = units::convert(1.0, units::Force::HartreePerBohr,
                                             units::Force::eVPerAngstrom);
    std::vector<std::array<double, 3>> result(static_cast<std::size_t>(n));
    for (int i = 0; i < n; ++i)
        for (int d = 0; d < 3; ++d)
            result[i][d] = (forces[i][d] + nonlocal[i][d]) * conversion;
    return result;
}

} // namespace internal
} // namespace pw
} // namespace cpu
} // namespace ATV
