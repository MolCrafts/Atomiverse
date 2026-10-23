/// @file src/cpu/semiempirical/dftb0.cpp
/// Non-SCC DFTB0 (H/C/N/O): transcribed mio-1-1 SK tables for every pair.
/// Nuclear forces are the closed-form non-SCC gradient (Hellmann–Feynman +
/// Pulay + repulsive). There is no SCC shift and no γ term.
#include "atomiverse/cpu/semiempirical/dftb0.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
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
                            linalg::RealMatrix& P, linalg::RealMatrix& W,
                            std::vector<double>& scaled) {
    constexpr double fill = 2.0;
    constexpr double beta = 0.0;
    dgemm_("N", "T", &n_ao, &n_ao, &n_occ, &fill, C.data(), &n_ao, C.data(),
           &n_ao, &beta, P.data(), &n_ao);
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

} // namespace

struct DFTB0::Workspace {
    explicit Workspace(const Frame& f);
    bool matches(const Frame& f) const;
    void solve(const Frame& f, bool want_forces);
    void assemble(const Frame& f);
    void analytic_forces(const Frame& f);

    int n_at = 0, n_ao = 0, n_occ = 0;
    std::vector<int> species, ao0;
    std::vector<const DftbElementRow*> el;
    linalg::RealMatrix H0, S, P, W;
    linalg::RealMatrix dH0[3], dS[3];
    std::unique_ptr<linalg::GeneralizedEigenSolver> eigen;
    std::vector<double> scaled, charges, fx, fy, fz;
    double energy = 0;
};

DFTB0::Workspace::Workspace(const Frame& f) {
    if (f.nlocal() < 1 || f.nlocal() > std::numeric_limits<int>::max() / 8)
        throw std::invalid_argument("DFTB0: invalid atom count");
    n_at = static_cast<int>(f.nlocal());
    const auto type = f.span(kAtom_type);
    species.assign(type.data(), type.data() + n_at);
    el.resize(n_at);
    ao0.resize(n_at + 1, 0);
    int n_elec = 0;
    for (int a = 0; a < n_at; ++a) {
        el[a] = &element_row(type[a]);
        n_elec += static_cast<int>(el[a]->n_val);
        ao0[a + 1] = ao0[a] + el[a]->n_ao;
    }
    if (n_elec % 2)
        throw std::invalid_argument(
            "DFTB0: odd valence electron count not supported");
    n_ao = ao0.back();
    n_occ = n_elec / 2;
    eigen = std::make_unique<linalg::GeneralizedEigenSolver>(n_ao);
    const auto n = static_cast<std::size_t>(n_ao);
    H0.resize({n, n});
    S.resize({n, n});
    P.resize({n, n});
    W.resize({n, n});
    for (int k = 0; k < 3; ++k) {
        dH0[k].resize({4, 4});
        dS[k].resize({4, 4});
    }
    scaled.resize(n * static_cast<std::size_t>(n_occ));
    charges.resize(n_at);
    fx.resize(n_at);
    fy.resize(n_at);
    fz.resize(n_at);
}

bool DFTB0::Workspace::matches(const Frame& f) const {
    if (f.nlocal() != n_at)
        return false;
    const auto type = f.span(kAtom_type);
    return std::equal(species.begin(), species.end(), type.data());
}

/// Non-SCC total energy and occupied eigenvectors at the current geometry.
/// \f$E = \sum_i 2\varepsilon_i + E_\mathrm{rep}\f$ [eV]. One min-image per
/// pair — the same geometry the force walk differentiates.
void DFTB0::Workspace::assemble(const Frame& f) {
    const auto x = f.span(kAtom_x), y = f.span(kAtom_y), z = f.span(kAtom_z);
    const auto type = f.span(kAtom_type);
    H0.fill(0.0);
    S.fill(0.0);
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

    eigen->solve(H0, S);
    const auto& eps = eigen->eigenvalues();

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

    energy = e_band + e_repulsive;
}

void DFTB0::Workspace::solve(const Frame& f, bool want_forces) {
    const auto x = f.span(kAtom_x), y = f.span(kAtom_y), z = f.span(kAtom_z);
    for (int a = 0; a < n_at; ++a) {
        if (!std::isfinite(x[a]) || !std::isfinite(y[a])
            || !std::isfinite(z[a]))
            throw std::invalid_argument("DFTB0: non-finite position");
        for (int b = a + 1; b < n_at; ++b) {
            double dx = x[b] - x[a], dy = y[b] - y[a], dz = z[b] - z[a];
            f.box.min_image(dx, dy, dz);
            if (!(dx * dx + dy * dy + dz * dz > 1e-24))
                throw std::invalid_argument("DFTB0: coincident atoms");
        }
    }
    assemble(f);
    build_density_weighted(eigen->eigenvectors(), eigen->eigenvalues(), n_ao,
                           n_occ, P, W, scaled);
    for (int a = 0; a < n_at; ++a) {
        double population = 0;
        for (int mu = ao0[a]; mu < ao0[a + 1]; ++mu)
            for (int nu = 0; nu < n_ao; ++nu)
                population += P(mu, nu) * S(mu, nu);
        charges[a] = el[a]->n_val - population;
        if (!std::isfinite(charges[a]))
            throw std::runtime_error("DFTB0: non-finite Mulliken charge");
    }
    if (!std::isfinite(energy))
        throw std::runtime_error("DFTB0: non-finite energy");
    if (want_forces) {
        std::fill(fx.begin(), fx.end(), 0.0);
        std::fill(fy.begin(), fy.end(), 0.0);
        std::fill(fz.begin(), fz.end(), 0.0);
        analytic_forces(f);
        for (int a = 0; a < n_at; ++a)
            if (!std::isfinite(fx[a]) || !std::isfinite(fy[a])
                || !std::isfinite(fz[a]))
                throw std::runtime_error("DFTB0: non-finite force");
    }
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
void DFTB0::Workspace::analytic_forces(const Frame& f) {
    const auto x = f.span(kAtom_x), y = f.span(kAtom_y), z = f.span(kAtom_z);
    const auto type = f.span(kAtom_type);
    // At most 4 x 4; set_block_derivative assigns every contracted entry.
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

DFTB0::DFTB0() = default;
DFTB0::~DFTB0() = default;
DFTB0::DFTB0(DFTB0&&) noexcept = default;
DFTB0& DFTB0::operator=(DFTB0&&) noexcept = default;

void DFTB0::prepare(const Frame& frame) {
    if (!workspace_ || !workspace_->matches(frame))
        workspace_ = std::make_unique<Workspace>(frame);
}

void DFTB0::build(const BuildContext& ctx) { prepare(ctx.frame); }

void DFTB0::evaluate(const Frame& frame, bool want_forces) {
    energy_ = std::numeric_limits<double>::quiet_NaN();
    prepare(frame);
    workspace_->solve(frame, want_forces);
    energy_ = workspace_->energy;
}

void DFTB0::publish(Frame& frame, bool want_forces, bool accumulate) const {
    auto q = frame.span(kAtom_charge);
    auto fx = frame.span(kAtom_fx), fy = frame.span(kAtom_fy),
         fz = frame.span(kAtom_fz);
    for (int a = 0; a < workspace_->n_at; ++a) {
        q[a] = workspace_->charges[a];
        if (want_forces) {
            fx[a] = (accumulate ? fx[a] : 0.0) + workspace_->fx[a];
            fy[a] = (accumulate ? fy[a] : 0.0) + workspace_->fy[a];
            fz[a] = (accumulate ? fz[a] : 0.0) + workspace_->fz[a];
        }
    }
    frame.meta.get(kEnergy) =
        (accumulate ? frame.meta.get(kEnergy) : 0.0) + energy_;
    frame.meta.get(kEnergyElectronic) =
        (accumulate ? frame.meta.get(kEnergyElectronic) : 0.0) + energy_;
}

void DFTB0::run(const Frame& frame) { evaluate(frame, false); }

void DFTB0::run(Frame& frame) {
    evaluate(frame, false);
    publish(frame, false, false);
}

void DFTB0::forces(Frame& frame) {
    evaluate(frame, true);
    publish(frame, true, false);
}

void DFTB0::launch(const LaunchContext& ctx) {
    evaluate(ctx.frame, ctx.forces_requested);
    publish(ctx.frame, ctx.forces_requested, true);
}

std::span<const double> DFTB0::last_charges() const {
    if (!std::isfinite(energy_) || !workspace_)
        return {};
    return workspace_->charges;
}

} // namespace semiempirical
} // namespace cpu
} // namespace ATV
