/// @file src/cpu/gpw/internal/collocate.cpp
/// @brief Implementation of uniform-grid Gaussian collocation / integration.

#include "atomiverse/cpu/gpw/internal/collocate.h"

#include "atomiverse/cpu/device/chunked_range.h"

#include <algorithm>
#include <cmath>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace ATV {
namespace cpu {
namespace gpw {

void UniformGrid::point(long idx, double r[3]) const {
    // k fastest, then j, then i (row-major i,j,k).
    const long nk = n[2];
    const long nj = n[1];
    const long k = idx % nk;
    const long j = (idx / nk) % nj;
    const long i = idx / (nk * nj);
    r[0] = origin[0] + (L[0] * static_cast<double>(i)) / n[0];
    r[1] = origin[1] + (L[1] * static_cast<double>(j)) / n[1];
    r[2] = origin[2] + (L[2] * static_cast<double>(k)) / n[2];
}

UniformGrid make_uniform_grid(const double cell_abc[3], double cutoff_ry) {
    UniformGrid g;
    const double gmax =
        std::sqrt(cutoff_ry); // |G_max| [1/Bohr], E_cut[Ry] = G^2
    for (int d = 0; d < 3; ++d) {
        g.L[d] = cell_abc[d];
        g.origin[d] = 0.0;
        // Nyquist: G_max = pi * n / L  ->  n = L * gmax / pi. Round up to even.
        int n = static_cast<int>(std::ceil(cell_abc[d] * gmax / M_PI));
        if (n < 2)
            n = 2;
        if (n % 2 != 0)
            ++n;
        g.n[d] = n;
    }
    return g;
}

namespace {

/// @brief Evaluate one primitive Gaussian at a grid point.
///
/// For a periodic cell the displacement r - center is taken as the
/// **minimum image** per axis (disp -= L * round(disp/L)), so a Gaussian
/// centered outside [0,L) (an unwrapped slab atom) contributes its density
/// to the grid through its nearest periodic image — matching CP2K's periodic
/// collocation. Atom positions are left untouched (wrapping them would overlap
/// the oxide bilayer). For an orthorhombic cell per-axis min-image is exact when
/// the Gaussian extent < L/2. Non-periodic (Martyna-Tuckerman / isolated) keeps
/// the raw displacement.
inline double eval_primitive(const GridPrimitive& p, const double r[3],
                             const double L[3], bool periodic) {
    double dx = r[0] - p.center[0];
    double dy = r[1] - p.center[1];
    double dz = r[2] - p.center[2];
    if (periodic) {
        dx -= L[0] * std::nearbyint(dx / L[0]);
        dy -= L[1] * std::nearbyint(dy / L[1]);
        dz -= L[2] * std::nearbyint(dz / L[2]);
    }
    const double r2 = dx * dx + dy * dy + dz * dz;
    double v = p.coeff * std::exp(-p.alpha * r2);
    for (int i = 0; i < p.l[0]; ++i)
        v *= dx;
    for (int i = 0; i < p.l[1]; ++i)
        v *= dy;
    for (int i = 0; i < p.l[2]; ++i)
        v *= dz;
    return v;
}

inline double eval_basis_function(const GridBasisFunction& bf,
                                  const double r[3], const double L[3],
                                  bool periodic) {
    double v = 0.0;
    for (const auto& p : bf)
        v += eval_primitive(p, r, L, periodic);
    return v;
}

} // namespace

// Radius beyond which a contracted Gaussian basis function is below `eps`.
// Per primitive solve |coeff| r^l exp(-alpha r^2) = eps; the r^l growth is folded
// in via one fixed-point correction on the pure-Gaussian root, then a generous
// margin so no real contribution is dropped (correctness over tightness).
double basis_function_radius(const GridBasisFunction& bf, double eps) {
    double rc = 0.0;
    const double leps = std::log(eps);
    for (const auto& p : bf) {
        if (p.coeff == 0.0 || p.alpha <= 0.0)
            continue;
        const int l = p.l[0] + p.l[1] + p.l[2];
        const double lc = std::log(std::abs(p.coeff));
        double r2 =
            std::max(0.0, (lc - leps) / p.alpha); // pure-Gaussian root^2
        // One correction for the r^l factor: ln|c| + l ln r - alpha r^2 = ln eps.
        for (int it = 0; it < 3; ++it) {
            const double r = std::sqrt(std::max(r2, 1e-12));
            r2 = std::max(0.0, (lc + l * std::log(r) - leps) / p.alpha);
        }
        rc = std::max(rc, std::sqrt(r2));
    }
    return rc;
}

CompactBasisGrid
build_compact_basis_grid(const UniformGrid& grid,
                         const std::vector<GridBasisFunction>& basis,
                         bool periodic, double eps) {
    const int nb = static_cast<int>(basis.size());
    const long n0 = grid.n[0], n1 = grid.n[1], n2 = grid.n[2];
    const double dx0 = grid.L[0] / n0, dx1 = grid.L[1] / n1,
                 dx2 = grid.L[2] / n2;

    struct Triple {
        long g;
        int u;
        double v;
    };
    // One bucket PER BASIS FUNCTION, not per thread. A per-thread bucket makes
    // the concatenated triple order — and therefore the summation order of every
    // downstream density/KS contraction over this CSR — a function of which
    // thread happened to take which `u` chunk, which a dynamic schedule varies
    // between two runs of the same binary. Bucketing by the loop index and
    // concatenating in ascending `u` gives one canonical layout at any thread
    // count. (Each iteration owns its bucket, so there is still no sharing.)
    std::vector<std::vector<Triple>> per_u(static_cast<std::size_t>(nb));

#pragma omp parallel for schedule(dynamic, 8)
    for (int u = 0; u < nb; ++u) {
        std::vector<Triple>& out = per_u[static_cast<std::size_t>(u)];
        const double rc = basis_function_radius(basis[u], eps);
        const double c[3] = {basis[u][0].center[0], basis[u][0].center[1],
                             basis[u][0].center[2]};
        // Γ-point periodic image sum χ_u(r) = Σ_T φ_u(r − T): enumerate the
        // UNWRAPPED window [ic−ih, ic+ih] of grid cells within rc of the
        // center, evaluate each at its true (unwrapped) coordinate — so the
        // image offset is carried by the raw displacement — then wrap the
        // storage index into [0,n). When the window is wider than the box
        // (2·rc ≥ L, e.g. a thin cell) multiple unwrapped cells map to the
        // same grid point; the duplicate (g,u) triples sum correctly in the
        // density/integration double loop, giving the multi-image χ_u.
        const long n[3] = {n0, n1, n2};
        const double dx[3] = {dx0, dx1, dx2};
        long lo[3], hi[3];
        for (int d = 0; d < 3; ++d) {
            const long ih = static_cast<long>(std::ceil(rc / dx[d]));
            const long ic = static_cast<long>(std::floor(c[d] / dx[d]));
            lo[d] = ic - ih;
            hi[d] = ic + ih;
            if (!periodic) { // isolated: clamp to the box, no wrap
                lo[d] = std::max(0L, lo[d]);
                hi[d] = std::min(n[d] - 1, hi[d]);
            }
        }
        for (long i = lo[0]; i <= hi[0]; ++i) {
            const long ii = ((i % n0) + n0) % n0;
            const double r0 =
                grid.origin[0] + grid.L[0] * static_cast<double>(i) / n0;
            for (long j = lo[1]; j <= hi[1]; ++j) {
                const long jj = ((j % n1) + n1) % n1;
                const double r1 = grid.origin[1]
                                  + grid.L[1] * static_cast<double>(j) / n1;
                const long base = (ii * n1 + jj) * n2;
                for (long k = lo[2]; k <= hi[2]; ++k) {
                    const long kk = ((k % n2) + n2) % n2;
                    const double r2 =
                        grid.origin[2]
                        + grid.L[2] * static_cast<double>(k) / n2;
                    const long g = base + kk;
                    // Raw displacement to the UNWRAPPED window position (the
                    // image offset is in r, so no min-image here).
                    const double r[3] = {r0, r1, r2};
                    const double val =
                        eval_basis_function(basis[u], r, grid.L, false);
                    if (std::fabs(val) > eps)
                        out.push_back({g, u, val});
                }
            }
        }
    }

    // Concatenate in ascending `u`, then bucket by grid point via counting sort
    // (O(nnz+ngrid), no comparison sort) into the CSR-by-point layout. The
    // counting sort is stable, so each point's entry list comes out ordered by
    // `u` — the canonical order, identical at every thread count.
    std::size_t total = 0;
    for (const auto& v : per_u)
        total += v.size();
    std::vector<Triple> all;
    all.reserve(total);
    for (auto& v : per_u) {
        all.insert(all.end(), v.begin(), v.end());
        std::vector<Triple>().swap(v);
    }

    const long ng = grid.npts();
    std::vector<long> cnt(static_cast<std::size_t>(ng), 0);
    for (const auto& t : all)
        ++cnt[static_cast<std::size_t>(t.g)];

    CompactBasisGrid cbg;
    cbg.npts = ng;
    cbg.weight = grid.weight();
    // Active points = grid points with >=1 nonzero; assign each an index and its
    // CSR offset (prefix sum of counts over active points only).
    std::vector<long> g2a(static_cast<std::size_t>(ng), -1);
    cbg.off.push_back(0);
    std::size_t run = 0;
    for (long g = 0; g < ng; ++g) {
        if (cnt[static_cast<std::size_t>(g)] == 0)
            continue;
        g2a[static_cast<std::size_t>(g)] = static_cast<long>(cbg.gidx.size());
        cbg.gidx.push_back(g);
        run += static_cast<std::size_t>(cnt[static_cast<std::size_t>(g)]);
        cbg.off.push_back(run);
    }
    cbg.fu.resize(run);
    cbg.fv.resize(run);
    std::vector<std::size_t> cur(cbg.gidx.size(), 0);
    for (const auto& t : all) {
        const long a = g2a[static_cast<std::size_t>(t.g)];
        const std::size_t pos = cbg.off[static_cast<std::size_t>(a)]
                                + cur[static_cast<std::size_t>(a)]++;
        cbg.fu[pos] = t.u;
        cbg.fv[pos] = t.v;
    }
    return cbg;
}

std::vector<double> collocate_density(const CompactBasisGrid& cbg, int nbasis,
                                      const std::vector<double>& P) {
    std::vector<double> rho(static_cast<std::size_t>(cbg.npts), 0.0);
    const long na = static_cast<long>(cbg.gidx.size());
#pragma omp parallel for schedule(dynamic, 64)
    for (long a = 0; a < na; ++a) {
        const std::size_t s = cbg.off[static_cast<std::size_t>(a)];
        const std::size_t e = cbg.off[static_cast<std::size_t>(a) + 1];
        // ρ(g) = Σ_uv P_uv χ_u χ_v with P symmetric → diagonal + 2×upper triangle,
        // halving the inner work (exact; P = 2 C_occ C_occᵀ is symmetric).
        double acc = 0.0;
        for (std::size_t ia = s; ia < e; ++ia) {
            const double* Pu =
                &P[static_cast<std::size_t>(cbg.fu[ia]) * nbasis];
            const double va = cbg.fv[ia];
            acc += va * va * Pu[cbg.fu[ia]]; // diagonal
            double s2 = 0.0;
            for (std::size_t ib = ia + 1; ib < e; ++ib)
                s2 += Pu[cbg.fu[ib]] * cbg.fv[ib];
            acc += 2.0 * va * s2; // upper triangle counted twice
        }
        rho[cbg.gidx[static_cast<std::size_t>(a)]] = acc;
    }
    return rho;
}

namespace {
// Grow-only flat chunk pool for integrate_potential (CSR + dense-phi paths).
// Levels / SCF iters reuse capacity; only the active nc·nn prefix is zeroed.
// Entry points are not nested concurrent (one integrate at a time per process).
struct IntegratePartPool {
    std::vector<double> data;
    double* ensure(long nc, std::size_t nn) {
        const std::size_t need = static_cast<std::size_t>(nc) * nn;
        if (data.size() < need)
            data.resize(need);
        std::fill(data.begin(), data.begin() + static_cast<std::ptrdiff_t>(need),
                  0.0);
        return data.data();
    }
};
IntegratePartPool g_cbg_integrate_part;
IntegratePartPool g_dense_integrate_part;
} // namespace

std::vector<double> integrate_potential(const CompactBasisGrid& cbg, int nbasis,
                                        const std::vector<double>& V_r) {
    const std::size_t nn = static_cast<std::size_t>(nbasis) * nbasis;
    std::vector<double> V(nn, 0.0);
    const long na = static_cast<long>(cbg.gidx.size());
    const double w = cbg.weight;
    // Reproducible reduction: a FIXED partition of the active points into
    // chunks, one nb² accumulator each, folded in ascending chunk order. The
    // per-thread accumulator this replaces was mixed by `#pragma omp critical`
    // in thread-completion order over a dynamically assigned point set, so the
    // KS matrix differed run to run at one thread count and again between
    // counts — noise the SCF then amplified through its line search.
    const device::ChunkedRange range(
        na, device::block_reduce_chunks(nn * sizeof(double)));
    const long nc = range.chunks();
    double* part = g_cbg_integrate_part.ensure(nc, nn);
#pragma omp parallel for schedule(dynamic, 1)
    for (long c = 0; c < nc; ++c) {
        double* Vloc = &part[static_cast<std::size_t>(c) * nn];
        const long alo = range.begin(c), ahi = range.end(c);
        for (long a = alo; a < ahi; ++a) {
            const std::size_t s = cbg.off[static_cast<std::size_t>(a)];
            const std::size_t e = cbg.off[static_cast<std::size_t>(a) + 1];
            const double vw = V_r[cbg.gidx[static_cast<std::size_t>(a)]] * w;
            for (std::size_t ia = s; ia < e; ++ia) {
                const double fu = cbg.fv[ia] * vw;
                double* Vu =
                    &Vloc[static_cast<std::size_t>(cbg.fu[ia]) * nbasis];
                for (std::size_t ib = s; ib < e; ++ib)
                    Vu[cbg.fu[ib]] += fu * cbg.fv[ib];
            }
        }
    }
    for (long c = 0; c < nc; ++c) {
        const double* Vloc = &part[static_cast<std::size_t>(c) * nn];
        for (std::size_t kk = 0; kk < nn; ++kk)
            V[kk] += Vloc[kk];
    }
    return V;
}

std::vector<double>
eval_basis_on_grid(const UniformGrid& grid,
                   const std::vector<GridBasisFunction>& basis, bool periodic) {
    const long ng = grid.npts();
    const int nb = static_cast<int>(basis.size());
    std::vector<double> phi(static_cast<std::size_t>(ng) * nb);
#pragma omp parallel for schedule(static)
    for (long g = 0; g < ng; ++g) {
        double r[3];
        grid.point(g, r);
        double* row = &phi[static_cast<std::size_t>(g) * nb];
        for (int u = 0; u < nb; ++u)
            row[u] = eval_basis_function(basis[u], r, grid.L, periodic);
    }
    return phi;
}

std::vector<long> active_grid_points(const std::vector<double>& phi, long ng,
                                     int nb, double eps) {
    std::vector<long> active;
    for (long g = 0; g < ng; ++g) {
        const double* row = &phi[static_cast<std::size_t>(g) * nb];
        for (int u = 0; u < nb; ++u) {
            if (std::fabs(row[u]) > eps) {
                active.push_back(g);
                break;
            }
        }
    }
    return active;
}

std::vector<double> collocate_density(const UniformGrid& grid,
                                      const std::vector<double>& phi, int nb,
                                      const std::vector<double>& P,
                                      const std::vector<long>& active) {
    std::vector<double> rho(static_cast<std::size_t>(grid.npts()), 0.0);
    const long na = static_cast<long>(active.size());
#pragma omp parallel for schedule(static)
    for (long i = 0; i < na; ++i) {
        const long g = active[static_cast<std::size_t>(i)];
        const double* row = &phi[static_cast<std::size_t>(g) * nb];
        double acc = 0.0;
        for (int u = 0; u < nb; ++u) {
            double s = 0.0;
            const double* Pu = &P[static_cast<std::size_t>(u) * nb];
            for (int v = 0; v < nb; ++v)
                s += Pu[v] * row[v];
            acc += row[u] * s;
        }
        rho[g] = acc;
    }
    return rho;
}

std::vector<double> collocate_density(const UniformGrid& grid,
                                      const std::vector<double>& phi, int nb,
                                      const std::vector<double>& P) {
    return collocate_density(grid, phi, nb, P,
                             active_grid_points(phi, grid.npts(), nb));
}

std::vector<double>
collocate_density(const UniformGrid& grid,
                  const std::vector<GridBasisFunction>& basis,
                  const std::vector<double>& P) {
    return collocate_density(grid, eval_basis_on_grid(grid, basis),
                             static_cast<int>(basis.size()), P);
}

std::vector<double> integrate_potential(const UniformGrid& grid,
                                        const std::vector<double>& phi, int nb,
                                        const std::vector<double>& V_r,
                                        const std::vector<long>& active) {
    const double w = grid.weight();
    const long na = static_cast<long>(active.size());
    const std::size_t nn = static_cast<std::size_t>(nb) * nb;
    std::vector<double> V(nn, 0.0);
    // Fixed-partition partial KS blocks, folded in ascending chunk order — the
    // reproducible form of the shared reduction (see cpu/device/chunked_range.h).
    // Grow-only pool: kill per-call nc·nb² heap across SCF iterations.
    const device::ChunkedRange range(
        na, device::block_reduce_chunks(nn * sizeof(double)));
    const long nc = range.chunks();
    double* part = g_dense_integrate_part.ensure(nc, nn);
#pragma omp parallel for schedule(dynamic, 1)
    for (long c = 0; c < nc; ++c) {
        double* Vloc = &part[static_cast<std::size_t>(c) * nn];
        const long ilo = range.begin(c), ihi = range.end(c);
        for (long i = ilo; i < ihi; ++i) {
            const long g = active[static_cast<std::size_t>(i)];
            const double* row = &phi[static_cast<std::size_t>(g) * nb];
            const double vw = V_r[g] * w;
            for (int u = 0; u < nb; ++u) {
                const double fu = row[u] * vw;
                double* Vu = &Vloc[static_cast<std::size_t>(u) * nb];
                for (int v = 0; v < nb; ++v)
                    Vu[v] += fu * row[v];
            }
        }
    }
    for (long c = 0; c < nc; ++c) {
        const double* Vloc = &part[static_cast<std::size_t>(c) * nn];
        for (std::size_t k = 0; k < nn; ++k)
            V[k] += Vloc[k];
    }
    return V;
}

std::vector<double> integrate_potential(const UniformGrid& grid,
                                        const std::vector<double>& phi, int nb,
                                        const std::vector<double>& V_r) {
    return integrate_potential(grid, phi, nb, V_r,
                               active_grid_points(phi, grid.npts(), nb));
}

std::vector<double>
integrate_potential(const UniformGrid& grid,
                    const std::vector<GridBasisFunction>& basis,
                    const std::vector<double>& V_r) {
    return integrate_potential(grid, eval_basis_on_grid(grid, basis),
                               static_cast<int>(basis.size()), V_r);
}

} // namespace gpw
} // namespace cpu
} // namespace ATV
