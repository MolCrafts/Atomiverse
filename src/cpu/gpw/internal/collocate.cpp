/// @file src/cpu/gpw/internal/collocate.cpp
/// @brief Implementation of uniform-grid Gaussian collocation / integration.

#include "atomiverse/cpu/gpw/internal/collocate.h"

#include "atomiverse/cpu/device/chunked_range.h"

#include <algorithm>
#include <cmath>
#include <vector>

#include <xsimd/xsimd.hpp>

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
IntegratePartPool g_cbg_integrate_part_b; // dual-spin β accumulators
IntegratePartPool g_dense_integrate_part;

// Symmetrize a packed nb×nb upper-triangle matrix (incl. diagonal) into a full
// symmetric matrix in place. Used after triangle-only integrate accumulate.
inline void symmetrize_packed(double* V, int nbasis) {
    for (int u = 0; u < nbasis; ++u) {
        double* row_u = &V[static_cast<std::size_t>(u) * nbasis];
        for (int v = u + 1; v < nbasis; ++v)
            V[static_cast<std::size_t>(v) * nbasis + u] = row_u[v];
    }
}

// Triangle accumulate into upper of nb×nb over one contiguous χ run:
// V += vw · φφᵀ. Callers pass the CSR span directly (`cbg.fu.data() + s`).
inline void accumulate_outer_upper_packed(double* Vloc, int nbasis,
                                          const int* fu, const double* fv,
                                          std::size_t k, double vw) {
    for (std::size_t ia = 0; ia < k; ++ia) {
        const int u = fu[ia];
        const double pi = fv[ia];
        double* Vu = &Vloc[static_cast<std::size_t>(u) * nbasis];
        Vu[u] += vw * pi * pi;
        const double pwa = pi * vw;
        for (std::size_t ib = ia + 1; ib < k; ++ib)
            Vu[fu[ib]] += pwa * fv[ib];
    }
}

// Dual-spin triangle: share χ loads; one φφᵀ product scaled by both vwa/vwb.
// Packed (fu,fv) spans stream better than CSR-indirect inner loads.
inline void accumulate_outer_upper_dual_packed(double* Va, double* Vb,
                                               int nbasis, const int* fu,
                                               const double* fv, std::size_t k,
                                               double vwa, double vwb) {
    for (std::size_t ia = 0; ia < k; ++ia) {
        const int u = fu[ia];
        const double pi = fv[ia];
        double* Vau = &Va[static_cast<std::size_t>(u) * nbasis];
        double* Vbu = &Vb[static_cast<std::size_t>(u) * nbasis];
        const double tii = pi * pi;
        Vau[u] += vwa * tii;
        Vbu[u] += vwb * tii;
        const double pwa = pi * vwa;
        const double pwb = pi * vwb;
        for (std::size_t ib = ia + 1; ib < k; ++ib) {
            const int v = fu[ib];
            const double fvb = fv[ib];
            Vau[v] += pwa * fvb;
            Vbu[v] += pwb * fvb;
        }
    }
}
} // namespace

std::vector<double> collocate_density(const CompactBasisGrid& cbg, int nbasis,
                                      const std::vector<double>& P) {
    std::vector<double> rho(static_cast<std::size_t>(cbg.npts), 0.0);
    const long na = static_cast<long>(cbg.gidx.size());
    // The CSR span [s,e) is already the contiguous χ run the triangle wants, so
    // it is read in place — copying it into scratch first only adds O(k) stores
    // per active point and removes no indirection.
#pragma omp parallel for schedule(dynamic, 64)
    for (long a = 0; a < na; ++a) {
        const std::size_t s = cbg.off[static_cast<std::size_t>(a)];
        const std::size_t k = cbg.off[static_cast<std::size_t>(a) + 1] - s;
        const int* fu = cbg.fu.data() + s;
        const double* fv = cbg.fv.data() + s;
        // ρ(g) = Σ_uv P_uv χ_u χ_v with P symmetric → diagonal + 2×upper.
        double acc = 0.0;
        for (std::size_t ia = 0; ia < k; ++ia) {
            const int u = fu[ia];
            const double va = fv[ia];
            const double* Pu = &P[static_cast<std::size_t>(u) * nbasis];
            acc += va * va * Pu[u];
            double s2 = 0.0;
            for (std::size_t ib = ia + 1; ib < k; ++ib)
                s2 += Pu[fu[ib]] * fv[ib];
            acc += 2.0 * va * s2;
        }
        rho[cbg.gidx[static_cast<std::size_t>(a)]] = acc;
    }
    return rho;
}

void collocate_density_spin(const CompactBasisGrid& cbg, int nbasis,
                            const std::vector<double>& P_a,
                            const std::vector<double>& P_b,
                            std::vector<double>& rho_a,
                            std::vector<double>& rho_b) {
    // One CSR walk for both spins. Per-spin FP order matches two separate
    // collocate_density calls (α then β at each point, same ia/ib order).
    // Pack χ into contiguous TLS so the inner ib contract streams; P rows stay
    // indirect (fu-ordered). Bit-identical order of summands preserved.
    const std::size_t ng = static_cast<std::size_t>(cbg.npts);
    rho_a.assign(ng, 0.0);
    rho_b.assign(ng, 0.0);
    const long na = static_cast<long>(cbg.gidx.size());
#pragma omp parallel
    {
        // Grow-only per-thread pack buffers (k ≤ nbasis at each point).
        thread_local std::vector<int> t_fu;
        thread_local std::vector<double> t_fv;
#pragma omp for schedule(dynamic, 64)
        for (long a = 0; a < na; ++a) {
            const std::size_t s = cbg.off[static_cast<std::size_t>(a)];
            const std::size_t e = cbg.off[static_cast<std::size_t>(a) + 1];
            const std::size_t k = e - s;
            if (t_fu.size() < k) {
                t_fu.resize(k);
                t_fv.resize(k);
            }
            for (std::size_t i = 0; i < k; ++i) {
                t_fu[i] = cbg.fu[s + i];
                t_fv[i] = cbg.fv[s + i];
            }
            double acc_a = 0.0;
            double acc_b = 0.0;
            for (std::size_t ia = 0; ia < k; ++ia) {
                const int u = t_fu[ia];
                const double va = t_fv[ia];
                const double* Pua =
                    &P_a[static_cast<std::size_t>(u) * nbasis];
                const double* Pub =
                    &P_b[static_cast<std::size_t>(u) * nbasis];
                acc_a += va * va * Pua[u];
                acc_b += va * va * Pub[u];
                double s2a = 0.0;
                double s2b = 0.0;
                for (std::size_t ib = ia + 1; ib < k; ++ib) {
                    const double vb = t_fv[ib];
                    const int v = t_fu[ib];
                    s2a += Pua[v] * vb;
                    s2b += Pub[v] * vb;
                }
                acc_a += 2.0 * va * s2a;
                acc_b += 2.0 * va * s2b;
            }
            const std::size_t g =
                static_cast<std::size_t>(cbg.gidx[static_cast<std::size_t>(a)]);
            rho_a[g] = acc_a;
            rho_b[g] = acc_b;
        }
    }
}

std::vector<double> integrate_potential(const CompactBasisGrid& cbg, int nbasis,
                                        const std::vector<double>& V_r) {
    const std::size_t nn = static_cast<std::size_t>(nbasis) * nbasis;
    std::vector<double> V(nn, 0.0);
    const long na = static_cast<long>(cbg.gidx.size());
    const double w = cbg.weight;
    // Reproducible reduction: fixed partition of active points into chunks,
    // one nb² accumulator each, folded in ascending chunk order.
    // Triangle accumulate (upper + diagonal) then symmetrize: halves the
    // φφᵀ work vs the full double loop. Deterministic at any thread count;
    // not bit-identical to the pre-triangle full-loop form.
    const device::ChunkedRange range(
        na, device::block_reduce_chunks(nn * sizeof(double)));
    const long nc = range.chunks();
    double* part = g_cbg_integrate_part.ensure(nc, nn);
    // The CSR span is already contiguous, so the triangle reads it in place.
#pragma omp parallel for schedule(dynamic, 1)
    for (long c = 0; c < nc; ++c) {
        double* Vloc = &part[static_cast<std::size_t>(c) * nn];
        const long alo = range.begin(c), ahi = range.end(c);
        for (long a = alo; a < ahi; ++a) {
            const std::size_t s = cbg.off[static_cast<std::size_t>(a)];
            const std::size_t k = cbg.off[static_cast<std::size_t>(a) + 1] - s;
            const double vw = V_r[cbg.gidx[static_cast<std::size_t>(a)]] * w;
            accumulate_outer_upper_packed(Vloc, nbasis, cbg.fu.data() + s,
                                          cbg.fv.data() + s, k, vw);
        }
    }
    for (long c = 0; c < nc; ++c) {
        const double* Vloc = &part[static_cast<std::size_t>(c) * nn];
        for (std::size_t kk = 0; kk < nn; ++kk)
            V[kk] += Vloc[kk];
    }
    symmetrize_packed(V.data(), nbasis);
    return V;
}

void integrate_potential_spin(const CompactBasisGrid& cbg, int nbasis,
                              const std::vector<double>& V_a,
                              const std::vector<double>& V_b,
                              std::vector<double>& out_a,
                              std::vector<double>& out_b) {
    const std::size_t nn = static_cast<std::size_t>(nbasis) * nbasis;
    out_a.assign(nn, 0.0);
    out_b.assign(nn, 0.0);
    const long na = static_cast<long>(cbg.gidx.size());
    const double w = cbg.weight;
    // Dual: one CSR walk, shared φφᵀ, scaled by V_eff^α/β. Pack χ per point
    // into TLS; fold chunk partials with portable xsimd (deterministic order).
    const device::ChunkedRange range(
        na, device::block_reduce_chunks(nn * sizeof(double)));
    const long nc = range.chunks();
    double* part_a = g_cbg_integrate_part.ensure(nc, nn);
    double* part_b = g_cbg_integrate_part_b.ensure(nc, nn);
#pragma omp parallel
    {
        thread_local std::vector<int> t_fu;
        thread_local std::vector<double> t_fv;
#pragma omp for schedule(dynamic, 1)
        for (long c = 0; c < nc; ++c) {
            double* Va = &part_a[static_cast<std::size_t>(c) * nn];
            double* Vb = &part_b[static_cast<std::size_t>(c) * nn];
            const long alo = range.begin(c), ahi = range.end(c);
            for (long a = alo; a < ahi; ++a) {
                const std::size_t s = cbg.off[static_cast<std::size_t>(a)];
                const std::size_t e = cbg.off[static_cast<std::size_t>(a) + 1];
                const std::size_t k = e - s;
                if (t_fu.size() < k) {
                    t_fu.resize(k);
                    t_fv.resize(k);
                }
                for (std::size_t i = 0; i < k; ++i) {
                    t_fu[i] = cbg.fu[s + i];
                    t_fv[i] = cbg.fv[s + i];
                }
                const long g = cbg.gidx[static_cast<std::size_t>(a)];
                const double vwa = V_a[static_cast<std::size_t>(g)] * w;
                const double vwb = V_b[static_cast<std::size_t>(g)] * w;
                accumulate_outer_upper_dual_packed(Va, Vb, nbasis, t_fu.data(),
                                                   t_fv.data(), k, vwa, vwb);
            }
        }
    }
    using batch_t = xsimd::batch<double>;
    constexpr std::size_t B = batch_t::size;
    for (long c = 0; c < nc; ++c) {
        const double* Va = &part_a[static_cast<std::size_t>(c) * nn];
        const double* Vb = &part_b[static_cast<std::size_t>(c) * nn];
        std::size_t kk = 0;
        for (; kk + B <= nn; kk += B) {
            const batch_t sa =
                batch_t::load_unaligned(out_a.data() + kk)
                + batch_t::load_unaligned(Va + kk);
            const batch_t sb =
                batch_t::load_unaligned(out_b.data() + kk)
                + batch_t::load_unaligned(Vb + kk);
            sa.store_unaligned(out_a.data() + kk);
            sb.store_unaligned(out_b.data() + kk);
        }
        for (; kk < nn; ++kk) {
            out_a[kk] += Va[kk];
            out_b[kk] += Vb[kk];
        }
    }
    symmetrize_packed(out_a.data(), nbasis);
    symmetrize_packed(out_b.data(), nbasis);
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
