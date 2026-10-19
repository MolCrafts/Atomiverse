/// @file src/cpu/semiempirical/dftb0.cpp
/// Non-SCC DFTB0 (H/C/N/O): transcribed mio-1-1 SK tables for every pair.
/// Nuclear forces are the closed-form non-SCC gradient (Hellmann–Feynman +
/// Pulay + repulsive). There is no SCC shift and no γ term.
#include "atomiverse/cpu/semiempirical/dftb0.h"

#include <array>
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

#include "atomiverse/core/frame_schema.h"
#include "atomiverse/cpu/linalg/eigensolve.h"
#include "atomiverse/cpu/linalg/matrix.h"
#include "atomiverse/cpu/semiempirical/basis_data/dftb_mio_tables.h"
#include "atomiverse/cpu/semiempirical/internal/mio_sk_block.h"
#include "atomiverse/cpu/semiempirical/internal/mio_sk_table.h"

// Column-major GEMM, same binding as the DFTB2 density build: a rank-n_occ
// update on raw buffers, not an expression-template temporary.
extern "C" {
void dgemm_(const char* transa, const char* transb, const int* m, const int* n,
            const int* k, const double* alpha, const double* a, const int* lda,
            const double* b, const int* ldb, const double* beta, double* c,
            const int* ldc);
}

namespace ATV {
namespace cpu {
namespace semiempirical {
namespace {

using basis_data::DftbElementRow;
using basis_data::DftbPairSkRow;
using basis_data::kDftbElements;
using basis_data::kDftbNElements;
using basis_data::kDftbNPairSk;
using basis_data::kDftbPairSk;

const DftbElementRow& element_row(int z) {
    for (std::size_t i = 0; i < kDftbNElements; ++i)
        if (kDftbElements[i].z == z)
            return kDftbElements[i];
    throw std::runtime_error("DFTB0: unsupported element Z=" + std::to_string(z)
                             + " (static mio-1-1 tables cover H/C/N/O only)");
}

const DftbPairSkRow& pair_sk(int za, int zb) {
    for (std::size_t i = 0; i < kDftbNPairSk; ++i)
        if (kDftbPairSk[i].z_a == za && kDftbPairSk[i].z_b == zb)
            return kDftbPairSk[i];
    throw std::runtime_error("DFTB0: no SK table for pair Z="
                             + std::to_string(za) + "-" + std::to_string(zb));
}

/// Occupied density \f$P = 2 C C^{\mathsf T}\f$ and energy-weighted
/// \f$W = 2 C\,\mathrm{diag}(\varepsilon)\,C^{\mathsf T}\f$ [e], [eV].
void build_density_weighted(const linalg::RealMatrix& C,
                            const linalg::RealVector& eps, int n_ao, int n_occ,
                            linalg::RealMatrix& P, linalg::RealMatrix& W) {
    constexpr double fill = 2.0;
    constexpr double beta = 0.0;
    dgemm_("N", "T", &n_ao, &n_ao, &n_occ, &fill, C.data(), &n_ao, C.data(),
           &n_ao, &beta, P.data(), &n_ao);
    std::vector<double> scaled(static_cast<std::size_t>(n_ao)
                               * static_cast<std::size_t>(n_occ));
    for (int o = 0; o < n_occ; ++o) {
        const double e = eps(o);
        const double* col =
            C.data()
            + static_cast<std::size_t>(o) * static_cast<std::size_t>(n_ao);
        double* dst =
            scaled.data()
            + static_cast<std::size_t>(o) * static_cast<std::size_t>(n_ao);
        for (int mu = 0; mu < n_ao; ++mu)
            dst[mu] = e * col[mu];
    }
    dgemm_("N", "T", &n_ao, &n_ao, &n_occ, &fill, scaled.data(), &n_ao,
           C.data(), &n_ao, &beta, W.data(), &n_ao);
}

struct NonScc {
    double energy = 0;
    linalg::RealMatrix C;
    linalg::RealVector eps;
};

/// Non-SCC total energy and occupied eigenvectors at the current geometry.
/// \f$E = \sum_i 2\varepsilon_i + E_\mathrm{rep}\f$ [eV]. One min-image per
/// pair — the same geometry the force walk differentiates.
NonScc solve_non_scc(const Frame& f, const StorageReal* x, const StorageReal* y,
                     const StorageReal* z, const int* type, int n_at,
                     const std::vector<const DftbElementRow*>& el,
                     const std::vector<int>& ao0, int n_ao, int n_occ) {
    linalg::RealMatrix H0 = xt::zeros<double>({n_ao, n_ao});
    linalg::RealMatrix S = xt::zeros<double>({n_ao, n_ao});

    auto dist_vec = [&](int i, int j, double& dx, double& dy, double& dz) {
        dx = static_cast<double>(x[j] - x[i]);
        dy = static_cast<double>(y[j] - y[i]);
        dz = static_cast<double>(z[j] - z[i]);
        f.box.min_image(dx, dy, dz);
        return std::sqrt(dx * dx + dy * dy + dz * dz);
    };

    for (int a = 0; a < n_at; ++a) {
        const auto& ea = *el[static_cast<std::size_t>(a)];
        const int ia = ao0[static_cast<std::size_t>(a)];
        H0(ia, ia) = ea.e_s;
        S(ia, ia) = 1.0;
        if (ea.n_ao == 4) {
            for (int p = 1; p < 4; ++p) {
                H0(ia + p, ia + p) = ea.e_p;
                S(ia + p, ia + p) = 1.0;
            }
        }
        for (int b = a + 1; b < n_at; ++b) {
            const auto& eb = *el[static_cast<std::size_t>(b)];
            double dx, dy, dz;
            const double r = dist_vec(a, b, dx, dy, dz);
            const double rcut = internal::MioSkTable{ea.z, eb.z}.cutoff();
            if (r > rcut)
                continue;
            const int ib = ao0[static_cast<std::size_t>(b)];
            const auto& sk_ab = pair_sk(type[a], type[b]);
            const auto& sk_ba = pair_sk(type[b], type[a]);
            internal::MioSketchPair{ea, eb, sk_ab}.add_block(dx, dy, dz, r, H0,
                                                             S, ia, ib);
            internal::MioSketchPair{eb, ea, sk_ba}.add_block(-dx, -dy, -dz, r,
                                                             H0, S, ib, ia);
        }
    }

    auto sol = linalg::diagonalize_fock_generalized(H0, S);
    const auto& eps = sol.first;

    double e_band = 0;
    for (int o = 0; o < n_occ; ++o)
        e_band += 2.0 * eps(o);

    double e_repulsive = 0;
    for (int a = 0; a < n_at; ++a)
        for (int b = a + 1; b < n_at; ++b) {
            double dx, dy, dz;
            const double r = dist_vec(a, b, dx, dy, dz);
            e_repulsive += internal::MioSkTable{type[a], type[b]}.repulsive(r);
        }

    NonScc out;
    out.energy = e_band + e_repulsive;
    out.eps = std::move(sol.first);
    out.C = std::move(sol.second);
    return out;
}

/// Closed-form \f$F = -\nabla E\f$ [eV/Å] at the non-SCC density.
///
/// Elstner et al., Phys. Rev. B 58, 7260 (1998), the band + repulsive piece
/// of Eq. (15) with the second-order charge term absent. The Hamiltonian is
/// pure \f$H^0\f$ (no SCC shift on \f$S\f$), so
/// \f$\partial E/\partial R = \mathrm{Tr}(P\,\partial H^0) -
/// \mathrm{Tr}(W\,\partial S) + \partial E_\mathrm{rep}\f$.
/// Only the A→B block is built; the B→A block is its transpose at \f$-R\f$,
/// and the two contributions are equal, hence the factor 2. \f$P\f$ and
/// \f$W\f$ already carry the closed-shell occupation 2.
void add_analytic_forces(const Frame& f, const StorageReal* x,
                         const StorageReal* y, const StorageReal* z,
                         const int* type, int n_at,
                         const std::vector<const DftbElementRow*>& el,
                         const std::vector<int>& ao0, int n_ao, int n_occ,
                         const NonScc& sol, AccReal* fx, AccReal* fy,
                         AccReal* fz) {
    linalg::RealMatrix P = xt::zeros<double>({n_ao, n_ao});
    linalg::RealMatrix W = xt::zeros<double>({n_ao, n_ao});
    build_density_weighted(sol.C, sol.eps, n_ao, n_occ, P, W);

    // One SK block is at most 4×4. `set_block_derivative` assigns every
    // element the contraction reads, so the scratch is not re-zeroed.
    constexpr int kSkBlock = 4;
    linalg::RealMatrix dH0[3] = {xt::zeros<double>({kSkBlock, kSkBlock}),
                                 xt::zeros<double>({kSkBlock, kSkBlock}),
                                 xt::zeros<double>({kSkBlock, kSkBlock})};
    linalg::RealMatrix dS[3] = {xt::zeros<double>({kSkBlock, kSkBlock}),
                                xt::zeros<double>({kSkBlock, kSkBlock}),
                                xt::zeros<double>({kSkBlock, kSkBlock})};

    for (int a = 0; a < n_at; ++a) {
        const auto& ea = *el[static_cast<std::size_t>(a)];
        const int ia = ao0[static_cast<std::size_t>(a)];
        const int na = ea.n_ao;
        for (int b = a + 1; b < n_at; ++b) {
            const auto& eb = *el[static_cast<std::size_t>(b)];
            const int ib = ao0[static_cast<std::size_t>(b)];
            const int nb = eb.n_ao;
            double dx = static_cast<double>(x[b] - x[a]);
            double dy = static_cast<double>(y[b] - y[a]);
            double dz = static_cast<double>(z[b] - z[a]);
            f.box.min_image(dx, dy, dz);
            const double r = std::sqrt(dx * dx + dy * dy + dz * dz);
            if (r < 1e-12)
                continue;
            const double dir[3] = {dx / r, dy / r, dz / r};
            // ∂E/∂R_a for this pair (F_b = −F_a). No γ, no SCC shift.
            std::array<double, 3> Fa = {0, 0, 0};

            const double rcut = internal::MioSkTable{ea.z, eb.z}.cutoff();
            if (r <= rcut) {
                const auto& sk = pair_sk(type[a], type[b]);
                internal::MioSketchPair{ea, eb, sk}.set_block_derivative(
                    dx, dy, dz, r, dH0, dS, 0, 0);
                for (int k = 0; k < 3; ++k) {
                    double acc = 0;
                    for (int i = 0; i < na; ++i)
                        for (int j = 0; j < nb; ++j) {
                            const double dh = dH0[k](i, j);
                            const double ds = dS[k](i, j);
                            acc +=
                                P(ia + i, ib + j) * dh - W(ia + i, ib + j) * ds;
                        }
                    Fa[static_cast<std::size_t>(k)] += 2.0 * acc;
                }
            }

            const double dErep_dr =
                internal::MioSkTable{type[a], type[b]}.d_repulsive_dr(r);
            for (int k = 0; k < 3; ++k)
                Fa[static_cast<std::size_t>(k)] += dErep_dr * dir[k];

            fx[a] += static_cast<AccReal>(Fa[0]);
            fy[a] += static_cast<AccReal>(Fa[1]);
            fz[a] += static_cast<AccReal>(Fa[2]);
            fx[b] -= static_cast<AccReal>(Fa[0]);
            fy[b] -= static_cast<AccReal>(Fa[1]);
            fz[b] -= static_cast<AccReal>(Fa[2]);
        }
    }
}

} // namespace

void DFTB0::build(const BuildContext& /*ctx*/) {}

void DFTB0::launch(const LaunchContext& ctx) {
    auto& f = ctx.frame;
    const int n_at = f.nlocal();
    if (n_at < 1)
        throw std::invalid_argument("DFTB0: empty frame");

    const auto type = f.span(kAtom_type);
    StorageReal* x = f.span(kAtom_x);
    StorageReal* y = f.span(kAtom_y);
    StorageReal* z = f.span(kAtom_z);
    AccReal* fx = f.span(kAtom_fx);
    AccReal* fy = f.span(kAtom_fy);
    AccReal* fz = f.span(kAtom_fz);
    double& energy_meta = f.meta.get(kEnergy);

    std::vector<const DftbElementRow*> el(static_cast<std::size_t>(n_at));
    std::vector<int> ao0(static_cast<std::size_t>(n_at) + 1, 0);
    double n_elec = 0;
    for (int a = 0; a < n_at; ++a) {
        el[static_cast<std::size_t>(a)] = &element_row(type[a]);
        n_elec += el[static_cast<std::size_t>(a)]->n_val;
        ao0[static_cast<std::size_t>(a) + 1] =
            ao0[static_cast<std::size_t>(a)]
            + el[static_cast<std::size_t>(a)]->n_ao;
    }
    const int n_ao = ao0[static_cast<std::size_t>(n_at)];
    if (n_ao < 1)
        throw std::runtime_error("DFTB0: no basis functions");
    if (std::fabs(n_elec - std::round(n_elec)) > 1e-9
        || static_cast<int>(std::lround(n_elec)) % 2 != 0)
        throw std::runtime_error(
            "DFTB0: odd/non-integer valence electron count not supported");
    const int n_occ = static_cast<int>(std::lround(n_elec)) / 2;

    const NonScc sol =
        solve_non_scc(f, x, y, z, type, n_at, el, ao0, n_ao, n_occ);
    energy_ = sol.energy;
    energy_meta += energy_;

    if (with_forces && ctx.forces_requested)
        add_analytic_forces(f, x, y, z, type, n_at, el, ao0, n_ao, n_occ, sol,
                            fx, fy, fz);
}

} // namespace semiempirical
} // namespace cpu
} // namespace ATV
