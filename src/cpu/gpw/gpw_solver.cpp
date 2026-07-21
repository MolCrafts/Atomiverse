#include <chrono>
#include <cstdio>
#include <cstdlib>
/// @file src/cpu/gpw/gpw_solver.cpp
/// @brief Implementation of the closed-shell GPW SCF (see header).
///
/// Electrostatics use the GTH Gaussian core charge: the long-range local PP
/// -Z erf(r/(sqrt2 rloc))/r is the potential of a negative Gaussian -Z g(rloc).
/// Solving one periodic Poisson of the NEUTRAL total charge n = rho_e - n_ion
/// yields V_es = v_H + v_ext^lr in a single FFT, and its energy
/// 1/2 integral n V_es = E_H + E_ext^lr + E_ii^smeared. The spurious Gaussian
/// self-energy of each core is removed analytically; the erf-vs-point ion-ion
/// correction is a documented TODO for the CP2K bit-parity milestone.

#include "atomiverse/cpu/gpw/gpw_solver.h"

#include <spdlog/spdlog.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "atomiverse/cpu/gpw/internal/dftd3.h"
#include "atomiverse/cpu/gpw/internal/diis.h"
#include "atomiverse/cpu/gpw/internal/gth_integrals.h"
#include "atomiverse/cpu/gpw/internal/hartree_grid.h"
#include "atomiverse/cpu/gpw/internal/local_pp.h"
#include "atomiverse/cpu/gpw/internal/mt_poisson.h"
#include "atomiverse/cpu/gpw/internal/multigrid.h"
#include "atomiverse/cpu/gpw/internal/one_electron.h"
#include "atomiverse/cpu/gpw/internal/ot.h"
#include "atomiverse/cpu/gpw/internal/periodic.h"
#include "atomiverse/cpu/gpw/internal/poisson_fft.h"
#include "atomiverse/core/real_harmonics.h"
#include "atomiverse/cpu/gpw/internal/xc_grid.h"
#include "atomiverse/cpu/gpw/internal/xc_grid_gga.h"
#include "atomiverse/cpu/linalg/density.h"    // build_density_matrix
#include "atomiverse/cpu/linalg/eigensolve.h" // diagonalize_fock_generalized
#include "atomiverse/cpu/linalg/matrix.h"     // RealMatrix

#include <xtensor-blas/xlinalg.hpp>
#include <xtensor/views/xview.hpp>

#ifdef ATV_GPW_FFTW_THREADS
#include <fftw3.h>
#include <omp.h>
#endif

namespace ATV {
namespace cpu {
namespace gpw {

namespace {

using ATV::cpu::linalg::RealMatrix;
using ATV::cpu::linalg::RealVector;

// Flatten all atoms' basis functions into one global list.
std::vector<GridBasisFunction> global_basis(const std::vector<GpwAtom>& atoms) {
    std::vector<GridBasisFunction> b;
    for (const auto& a : atoms)
        for (const auto& bf : a.basis)
            b.push_back(bf);
    return b;
}

Prim to_prim(const GridPrimitive& p) {
    Prim q;
    q.alpha = p.alpha;
    q.center[0] = p.center[0];
    q.center[1] = p.center[1];
    q.center[2] = p.center[2];
    q.l[0] = p.l[0];
    q.l[1] = p.l[1];
    q.l[2] = p.l[2];
    return q;
}

// Screening tolerance for the geometry-fixed GTH/one-electron build. A term is
// skipped only when a rigorous Gaussian-magnitude bound falls below this — the
// exponential decay dominates, so dropped terms are far below the SCF energy
// tolerance (validated bit-unchanged against the small-system CP2K-parity tests).
// This is a compute-time saving in the O(nb^2*natoms) build; the stored matrices
// remain dense (the overlap matrices are not sparse for this geometry).
constexpr double kBuildScreen = 1e-12;

// Spatial bin list of atom positions for O(1)-per-query neighbour lookup, so the
// GTH atom loop (PP is short-ranged) visits only atoms near a basis-pair product
// centre instead of all N — turning the O(nb²·natoms) build into O(nb²). Periodic
// orthorhombic; query returns each atom once (gth applies its own min-image).
struct AtomCells {
    double L[3] = {0, 0, 0};
    int nc[3] = {1, 1, 1};
    double cs[3] = {1, 1, 1};
    bool periodic = true;
    std::vector<std::vector<int>> bins;

    void build(const std::vector<GpwAtom>& atoms, const double cell[3],
               double target_cs, bool per) {
        periodic = per;
        long ntot = 1;
        for (int d = 0; d < 3; ++d) {
            L[d] = cell[d];
            nc[d] = std::max(1, static_cast<int>(std::floor(L[d] / target_cs)));
            cs[d] = L[d] / nc[d];
            ntot *= nc[d];
        }
        bins.assign(static_cast<std::size_t>(ntot), {});
        for (int a = 0; a < static_cast<int>(atoms.size()); ++a) {
            int c[3];
            for (int d = 0; d < 3; ++d) {
                double x = atoms[a].pos[d];
                if (periodic && L[d] > 0.0)
                    x -= L[d] * std::floor(x / L[d]);
                c[d] = std::min(nc[d] - 1,
                                std::max(0, static_cast<int>(x / cs[d])));
            }
            bins[static_cast<std::size_t>((c[0] * nc[1] + c[1]) * nc[2] + c[2])]
                .push_back(a);
        }
    }

    // Visit every atom whose bin lies within radius R of P (each atom once).
    template <typename Visit>
    void query(const double P[3], double R, Visit visit) const {
        int cells[3][256];
        int len[3];
        for (int d = 0; d < 3; ++d) {
            double x = P[d];
            if (periodic && L[d] > 0.0)
                x -= L[d] * std::floor(x / L[d]);
            const int pc =
                std::min(nc[d] - 1, std::max(0, static_cast<int>(x / cs[d])));
            // +1 guards the floor slop: an atom within R can land one cell beyond
            // ceil(R/cs) from pc, so visit that extra layer (no neighbour missed).
            const int span = static_cast<int>(std::ceil(R / cs[d])) + 1;
            int n = 0;
            if (!periodic) {
                for (int c = std::max(0, pc - span);
                     c <= std::min(nc[d] - 1, pc + span); ++c)
                    if (n < 256)
                        cells[d][n++] = c;
            } else if (2 * span + 1 >= nc[d]) {
                for (int c = 0; c < nc[d] && n < 256; ++c)
                    cells[d][n++] = c; // whole axis
            } else {
                for (int t = -span; t <= span && n < 256; ++t)
                    cells[d][n++] = ((pc + t) % nc[d] + nc[d]) % nc[d];
            }
            len[d] = n;
        }
        for (int i = 0; i < len[0]; ++i)
            for (int j = 0; j < len[1]; ++j) {
                const int base = (cells[0][i] * nc[1] + cells[1][j]) * nc[2];
                for (int k = 0; k < len[2]; ++k)
                    for (int a :
                         bins[static_cast<std::size_t>(base + cells[2][k])])
                        visit(a);
            }
    }
};

// Spatial bin list of basis-function CENTERS, so the geometry-fixed S/T/Vpp
// build visits only the O(neighbours) pairs (i,j) whose supports can overlap
// instead of all nb² — turning the outer build loop from O(nb²) (every pair
// screened) into O(nb·neighbours). Periodic orthorhombic; query returns each
// basis index once. Conservative: a surviving pair has centres within
// radii[i]+radii[j] ≤ radii[i]+rad_max (min-image), so querying that radius
// misses none; the exact pair_overlap_bound screen then filters → bit-identical.
struct BasisCells {
    double L[3] = {0, 0, 0};
    int nc[3] = {1, 1, 1};
    double cs[3] = {1, 1, 1};
    bool periodic = true;
    std::vector<std::vector<int>> bins;

    void build(const std::vector<GridBasisFunction>& basis,
               const double cell[3], double target_cs, bool per) {
        periodic = per;
        long ntot = 1;
        for (int d = 0; d < 3; ++d) {
            L[d] = cell[d];
            nc[d] = std::max(1, static_cast<int>(std::floor(L[d] / target_cs)));
            cs[d] = L[d] / nc[d];
            ntot *= nc[d];
        }
        bins.assign(static_cast<std::size_t>(ntot), {});
        for (int b = 0; b < static_cast<int>(basis.size()); ++b) {
            if (basis[static_cast<std::size_t>(b)].empty())
                continue;
            const double* ctr = basis[static_cast<std::size_t>(b)][0].center;
            int c[3];
            for (int d = 0; d < 3; ++d) {
                double x = ctr[d];
                if (periodic && L[d] > 0.0)
                    x -= L[d] * std::floor(x / L[d]);
                c[d] = std::min(nc[d] - 1,
                                std::max(0, static_cast<int>(x / cs[d])));
            }
            bins[static_cast<std::size_t>((c[0] * nc[1] + c[1]) * nc[2] + c[2])]
                .push_back(b);
        }
    }

    // Visit every basis index whose bin lies within radius R of P (each once).
    template <typename Visit>
    void query(const double P[3], double R, Visit visit) const {
        int cells[3][256];
        int len[3];
        for (int d = 0; d < 3; ++d) {
            double x = P[d];
            if (periodic && L[d] > 0.0)
                x -= L[d] * std::floor(x / L[d]);
            const int pc =
                std::min(nc[d] - 1, std::max(0, static_cast<int>(x / cs[d])));
            const int span = static_cast<int>(std::ceil(R / cs[d])) + 1;
            int n = 0;
            if (!periodic) {
                for (int c = std::max(0, pc - span);
                     c <= std::min(nc[d] - 1, pc + span); ++c)
                    if (n < 256)
                        cells[d][n++] = c;
            } else if (2 * span + 1 >= nc[d]) {
                for (int c = 0; c < nc[d] && n < 256; ++c)
                    cells[d][n++] = c;
            } else {
                for (int t = -span; t <= span && n < 256; ++t)
                    cells[d][n++] = ((pc + t) % nc[d] + nc[d]) % nc[d];
            }
            len[d] = n;
        }
        for (int i = 0; i < len[0]; ++i)
            for (int j = 0; j < len[1]; ++j) {
                const int base = (cells[0][i] * nc[1] + cells[1][j]) * nc[2];
                for (int k = 0; k < len[2]; ++k)
                    for (int b :
                         bins[static_cast<std::size_t>(base + cells[2][k])])
                        visit(b);
            }
    }
};

// <mu| V_loc^sr + V_nl |nu> summed over all atoms acting as PP centers.
//
// Screened: a primitive pair (p,q) is skipped when its Gaussian product
// prefactor exp(-a*b/(a+b)|Rp-Rq|^2) is negligible; for a surviving pair, an
// atom A is skipped when the combined product-PP decay exp(-(g*k/(g+k))|P-A|^2)
// (g=a+b, k=1/(2 rmax^2), P the product center, rmax the atom's largest PP
// width) puts its contribution below the screen. The |P-A| local-style bound
// over-estimates the separable nonlocal magnitude when p,q overlap, so it is
// conservative for both the local and nonlocal terms. Turns the per-pair atom
// loop from O(natoms) into O(local neighbours) -> overall O(N) build.
// Γ-point periodic GTH pseudopotential matrix element. The basis is diffuse
// (reach > L/2 in a thin cell), so the bra φ_ν is summed over its lattice images
// T_ν (outer loop, screened by the Gaussian-product prefactor K_pq). The GTH
// potential itself is short-ranged (rloc / projector radii ~0.3-0.5 Bohr ≪ L/2),
// so for each atom only its single nearest periodic image to the product center P
// contributes — taken by min-image, no atom-image loop. `m` (from image_counts)
// must match the basis-collocation / overlap image set for consistency.
double gth_matrix_element(const GridBasisFunction& mu,
                          const GridBasisFunction& nu,
                          const std::vector<GpwAtom>& atoms,
                          const AtomCells& cells, double rmax_global,
                          const double L[3], const int m[3], double reach) {
    if (mu.empty() || nu.empty())
        return 0.0;
    const double* muc = mu[0].center;
    const double* nuc = nu[0].center;
    const double reach2 = reach * reach;
    const bool per = m[0] || m[1] || m[2];
    // Widest PP across all atoms → smallest κ → largest possible screen reach;
    // used to size the spatial query conservatively (the exact per-atom screen
    // below is unchanged, so results are bit-identical to the all-atoms loop).
    const double kappa_min = 1.0 / (2.0 * rmax_global * rmax_global);
    double v = 0.0;
    for (const auto& p : mu) {
        const Prim pp = to_prim(p);
        for (const auto& q0 : nu) {
            const double cc = p.coeff * q0.coeff;
            if (cc == 0.0)
                continue;
            const double acc = std::abs(cc);
            const double a = p.alpha, b = q0.alpha, g = a + b;
            for (int tx = -m[0]; tx <= m[0]; ++tx)
                for (int ty = -m[1]; ty <= m[1]; ++ty)
                    for (int tz = -m[2]; tz <= m[2]; ++tz) {
                        // Skip ν-images whose shifted center cannot overlap μ:
                        // no μ-ν overlap ⇒ no 3-centre PP contribution.
                        const double sx = muc[0] - nuc[0] - tx * L[0];
                        const double sy = muc[1] - nuc[1] - ty * L[1];
                        const double sz = muc[2] - nuc[2] - tz * L[2];
                        if (sx * sx + sy * sy + sz * sz > reach2)
                            continue;
                        GridPrimitive q = q0; // image T_ν of the bra primitive
                        q.center[0] += tx * L[0];
                        q.center[1] += ty * L[1];
                        q.center[2] += tz * L[2];
                        double dpq2 = 0.0, P[3];
                        for (int d = 0; d < 3; ++d) {
                            const double diff = p.center[d] - q.center[d];
                            dpq2 += diff * diff;
                            P[d] = (a * p.center[d] + b * q.center[d]) / g;
                        }
                        const double Kpq = std::exp(-a * b / g * dpq2);
                        if (acc * Kpq < kBuildScreen)
                            continue; // image too far
                        const Prim qq = to_prim(q);
                        // Query radius: dPA² where the (conservative, widest-PP)
                        // screen reaches kBuildScreen — atoms beyond it can't pass.
                        const double Rq = std::sqrt(std::max(
                            0.0, std::log(acc * Kpq / kBuildScreen)
                                     * (g + kappa_min) / (g * kappa_min)));
                        cells.query(P, Rq, [&](int ai) {
                            const GpwAtom& at =
                                atoms[static_cast<std::size_t>(ai)];
                            double rmax = at.rloc;
                            for (const auto& ch : at.nonlocal)
                                rmax = std::max(rmax, ch.r);
                            const double kappa = 1.0 / (2.0 * rmax * rmax);
                            double apos[3], dPA2 = 0.0;
                            for (int d = 0; d < 3; ++d) {
                                apos[d] = at.pos[d];
                                if (L[d] > 0.0 && per)
                                    apos[d] -= L[d]
                                               * std::nearbyint(
                                                   (at.pos[d] - P[d]) / L[d]);
                                const double e = P[d] - apos[d];
                                dPA2 += e * e;
                            }
                            if (acc * Kpq
                                    * std::exp(-(g * kappa / (g + kappa))
                                               * dPA2)
                                < kBuildScreen)
                                return; // atom too far for any PP contribution
                            if (!at.cloc.empty())
                                v += cc
                                     * gth_local_sr(pp, qq, apos, at.rloc,
                                                    at.cloc);
                            // V_nl (nonlocal) is built separately by add_nonlocal_kb
                            // in the separable Kleinman-Bylander form: recomputing
                            // the projector overlaps per basis PAIR here is O(nb²·proj)
                            // and was the GPW setup bottleneck (hundreds-× vs CP2K).
                        });
                    }
        }
    }
    return v;
}

// Separable periodic Kleinman-Bylander assembly of the GTH nonlocal matrix V_nl.
//   V_nl(μ,ν) = Σ_{A in cell} Σ_{ch,m} Σ_{ij} Oμ[A,ch,m,i] h^ch_ij Oν[A,ch,m,j],
//   Oμ[A,ch,m,i] = <φ_μ^Γ | proj_{A,ch,m,i}>  (Bloch sum over μ-images reaching A).
// Each Oμ is computed ONCE per (μ, atom) — vs the old gth_nonlocal recomputing
// the projector overlaps for every basis PAIR. That per-pair recomputation was
// O(nb²·prim²·proj·moment3d); this is O(nb·prim·proj·moment3d) projector
// precompute + cheap O(pairs·proj²) scalar assembly. This is CP2K's GPW form and
// the fix for the hundreds-× setup gap. The atom loop is parallel; pair writes go
// through omp atomic (two atoms can both contribute to one (μ,ν)).
void add_nonlocal_kb(RealMatrix& Vpp,
                     const std::vector<GridBasisFunction>& basis,
                     const std::vector<GpwAtom>& atoms,
                     const std::vector<double>& radii, const double L[3],
                     const int mimg[3], const BasisCells& bcells) {
    const int natoms = static_cast<int>(atoms.size());
    double rad_max = 0.0;
    for (double r : radii)
        rad_max = std::max(rad_max, r);
    const double proj_eps = std::sqrt(2.0 * std::log(1.0 / kBuildScreen));
#pragma omp parallel for schedule(dynamic, 1)
    for (int ai = 0; ai < natoms; ++ai) {
        const GpwAtom& at = atoms[static_cast<std::size_t>(ai)];
        if (at.nonlocal.empty())
            continue;
        double rmax_ch = 0.0;
        for (const auto& ch : at.nonlocal)
            rmax_ch = std::max(rmax_ch, ch.r);
        const double proj_reach = rmax_ch * proj_eps;
        // Flat slot layout over (channel, m, iproj); per-channel base offsets.
        int nslot = 0;
        std::vector<int> choff(at.nonlocal.size());
        for (std::size_t c = 0; c < at.nonlocal.size(); ++c) {
            choff[c] = nslot;
            nslot += (2 * at.nonlocal[c].l + 1) * at.nonlocal[c].nproj;
        }
        const double* A = at.pos;
        // Gather basis funcs reaching A. The periodic cell-list query around A
        // already accounts for A's images relative to the (home) basis centres.
        std::vector<int> mus;
        std::vector<std::vector<double>> Ov, hOv;
        bcells.query(A, rad_max + proj_reach, [&](int mu) {
            const GridBasisFunction& bf = basis[static_cast<std::size_t>(mu)];
            if (bf.empty())
                return;
            const double reach =
                radii[static_cast<std::size_t>(mu)] + proj_reach;
            const double reach2 = reach * reach;
            std::vector<double> o(static_cast<std::size_t>(nslot), 0.0);
            bool any = false;
            for (const auto& p : bf) {
                if (p.coeff == 0.0)
                    continue;
                // Bloch sum: every μ-image whose centre reaches A contributes.
                for (int tx = -mimg[0]; tx <= mimg[0]; ++tx)
                    for (int ty = -mimg[1]; ty <= mimg[1]; ++ty)
                        for (int tz = -mimg[2]; tz <= mimg[2]; ++tz) {
                            Prim g = to_prim(p);
                            g.center[0] += tx * L[0];
                            g.center[1] += ty * L[1];
                            g.center[2] += tz * L[2];
                            const double dx = g.center[0] - A[0],
                                         dy = g.center[1] - A[1],
                                         dz = g.center[2] - A[2];
                            if (dx * dx + dy * dy + dz * dz > reach2)
                                continue;
                            for (std::size_t c = 0; c < at.nonlocal.size();
                                 ++c) {
                                const auto& ch = at.nonlocal[c];
                                const int nm = 2 * ch.l + 1;
                                for (int mm = 0; mm < nm; ++mm)
                                    for (int ip = 0; ip < ch.nproj; ++ip)
                                        o[static_cast<std::size_t>(
                                            choff[c] + mm * ch.nproj + ip)] +=
                                            p.coeff
                                            * gth_projector_overlap(
                                                g, A, ch.l, ip + 1, ch.r, mm);
                            }
                            any = true;
                        }
            }
            if (!any)
                return;
            // Apply the channel coupling once: hO[i] = Σ_j h_ij O[j] within (ch,m).
            std::vector<double> ho(static_cast<std::size_t>(nslot), 0.0);
            for (std::size_t c = 0; c < at.nonlocal.size(); ++c) {
                const auto& ch = at.nonlocal[c];
                const int nm = 2 * ch.l + 1;
                for (int mm = 0; mm < nm; ++mm) {
                    const int base = choff[c] + mm * ch.nproj;
                    for (int i = 0; i < ch.nproj; ++i) {
                        double s = 0.0;
                        for (int j = 0; j < ch.nproj; ++j)
                            s += ch.hij(i, j)
                                 * o[static_cast<std::size_t>(base + j)];
                        ho[static_cast<std::size_t>(base + i)] = s;
                    }
                }
            }
            mus.push_back(mu);
            Ov.push_back(std::move(o));
            hOv.push_back(std::move(ho));
        });
        // V_nl(μ,ν) += Oμ · (h Oν), full double loop fills both off-diagonals.
        const int ng = static_cast<int>(mus.size());
        for (int k1 = 0; k1 < ng; ++k1)
            for (int k2 = 0; k2 < ng; ++k2) {
                double s = 0.0;
                const std::vector<double>& o1 =
                    Ov[static_cast<std::size_t>(k1)];
                const std::vector<double>& h2 =
                    hOv[static_cast<std::size_t>(k2)];
                for (int t = 0; t < nslot; ++t)
                    s += o1[static_cast<std::size_t>(t)]
                         * h2[static_cast<std::size_t>(t)];
                if (s == 0.0)
                    continue;
                double& dst = Vpp(mus[static_cast<std::size_t>(k1)],
                                  mus[static_cast<std::size_t>(k2)]);
#pragma omp atomic
                dst += s;
            }
    }
}

// Max Gaussian-product prefactor over all primitive pairs of (mu,nu): bounds the
// magnitude of S, T, and Vpp for this pair. Below kBuildScreen the whole (i,j)
// element is negligible and can be skipped (overlap pair screening).
double pair_overlap_bound(const GridBasisFunction& mu,
                          const GridBasisFunction& nu, const double L[3],
                          bool periodic) {
    double best = 0.0;
    for (const auto& p : mu)
        for (const auto& q : nu) {
            const double a = p.alpha, b = q.alpha;
            double d2 = 0.0;
            for (int k = 0; k < 3; ++k) {
                double e = p.center[k] - q.center[k];
                // Min-image: a pair straddling a periodic boundary is a true near
                // neighbour through its nearest image — keep it past the screen.
                if (periodic && L[k] > 0.0)
                    e -= L[k] * std::nearbyint(e / L[k]);
                d2 += e * e;
            }
            best = std::max(best, std::abs(p.coeff * q.coeff)
                                      * std::exp(-a * b / (a + b) * d2));
        }
    return best;
}

// Ionic core density n_ion(r) = sum_A z_A (2 pi rloc^2)^{-3/2} exp(-|r-R|^2/2 rloc^2).
// Periodic: minimum-image displacement so unwrapped slab cores collocate through
// their nearest image (matching the electron-density collocation in collocate.cpp).
std::vector<double> ionic_density(const UniformGrid& grid,
                                  const std::vector<GpwAtom>& atoms,
                                  bool periodic) {
    const long ng = grid.npts();
    std::vector<double> n(static_cast<std::size_t>(ng), 0.0);
#pragma omp parallel for schedule(static)
    for (long g = 0; g < ng; ++g) {
        double r[3];
        grid.point(g, r);
        double s = 0.0;
        for (const auto& a : atoms) {
            const double s2 = a.rloc * a.rloc;
            const double norm = a.z_val * std::pow(2.0 * M_PI * s2, -1.5);
            double dx = r[0] - a.pos[0], dy = r[1] - a.pos[1],
                   dz = r[2] - a.pos[2];
            if (periodic) {
                dx -= grid.L[0] * std::nearbyint(dx / grid.L[0]);
                dy -= grid.L[1] * std::nearbyint(dy / grid.L[1]);
                dz -= grid.L[2] * std::nearbyint(dz / grid.L[2]);
            }
            s += norm * std::exp(-(dx * dx + dy * dy + dz * dz) / (2.0 * s2));
        }
        n[g] = s;
    }
    return n;
}

// GTH local short-range pseudopotential on the real-space grid:
//   V_loc^SR(r) = Σ_A exp(−t/2) Σ_i C_i^A t^i,  t = |r−R_A|²/rloc_A².
// This is the SAME short-range form gth_local_sr integrates analytically (per
// basis pair, O(nb²·natoms) — 57% of the STVpp build). CP2K instead collocates it
// onto the grid (here) and folds it into V_eff, so the integral against the
// density is shared with the Hartree/XC potential — O(grid), not O(nb²). Geometry-
// fixed: built once and added to V_eff every SCF step.

// Analytic Hartree self-energy of one normalized Gaussian core of charge z and
// width rloc: 1/2 integral rho V[rho] = z^2 / (2 rloc sqrt(pi)).
double core_self_energy(const GpwAtom& a) {
    return a.z_val * a.z_val / (2.0 * a.rloc * std::sqrt(M_PI));
}

RealMatrix vec_to_mat(const std::vector<double>& v, int n) {
    RealMatrix m = xt::zeros<double>({n, n});
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            m(i, j) = v[static_cast<std::size_t>(i) * n + j];
    return m;
}

std::vector<double> mat_to_vec(const RealMatrix& m, int n) {
    std::vector<double> v(static_cast<std::size_t>(n) * n);
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            v[static_cast<std::size_t>(i) * n + j] = m(i, j);
    return v;
}

double trace_product(const RealMatrix& P, const RealMatrix& M, int n) {
    double t = 0.0;
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            t += P(i, j) * M(i, j);
    return t;
}

// Level-shifted diagonalization in the canonical-orthogonal basis.
//
// In the orthogonal basis F' = XᵀFX, add λ·(I − P'_occ) so the VIRTUAL block is
// pushed up by λ while the occupied block is untouched. This widens the effective
// HOMO-LUMO gap and kills the occupied/virtual state-swapping that makes a
// small-gap system (ZnO: 400 iterations, parked millihartrees above the minimum
// OT reaches) oscillate forever. The shift is removed from the reported orbital
// energies of the virtuals, and it vanishes at convergence because P'_occ then
// stops changing — the converged density is the unshifted one.
//
// P'_occ = C'_occ C'_occᵀ with C'_occ = Xᵀ S C_occ (XᵀSX = I ⇒ C_occ = X C'_occ).
std::pair<RealVector, RealMatrix>
diagonalize_fock_shifted(const RealMatrix& F, const RealMatrix& X,
                         const RealMatrix& S, const RealMatrix& C_prev,
                         int n_occ, double shift) {
    if (shift <= 0.0 || C_prev.size() == 0)
        return linalg::diagonalize_fock(F, X);

    const int m = static_cast<int>(X.shape(1));
    RealMatrix Fp = xt::linalg::dot(xt::transpose(X), xt::linalg::dot(F, X));

    // C'_occ = Xᵀ S C_occ  (m × n_occ)
    const RealMatrix C_occ = xt::view(C_prev, xt::all(), xt::range(0, n_occ));
    const RealMatrix Cp_occ =
        xt::linalg::dot(xt::transpose(X), xt::linalg::dot(S, C_occ));
    const RealMatrix Pocc =
        xt::linalg::dot(Cp_occ, xt::transpose(Cp_occ)); // m × m projector

    for (int i = 0; i < m; ++i) {
        Fp(i, i) += shift;
        for (int j = 0; j < m; ++j)
            Fp(i, j) -= shift * Pocc(i, j);
    }

    auto [eig, Cp] = xt::linalg::eigh(Fp);
    RealMatrix C = xt::linalg::dot(X, Cp);
    for (int i = n_occ; i < m; ++i)
        eig(i) -= shift; // report unshifted virtuals
    return {eig, C};
}

// On convergence, diagonalize the converged KS matrix once more to record a
// self-consistent (C, eps, P) triple in the result (P == density(C)). This is
// what the analytic-force assembly needs to build the energy-weighted density
// matrix W = Σ_i 2 ε_i c_i c_iᵀ.
void store_converged_state(GpwResult& res, const RealMatrix& H,
                           const RealMatrix& X, int n_occ, int nb) {
    auto [eig, C] = linalg::diagonalize_fock(H, X);
    res.C = C;
    res.eps.assign(eig.begin(), eig.end());
    res.P = linalg::build_density_matrix(C, n_occ, nb, 2.0);
}

} // namespace

namespace {
// One Cartesian term of a real solid harmonic: powers + weight.
struct CartTerm {
    int l[3];
    double w;
};
// Real solid harmonics as Cartesian combinations, per angular momentum, for
// arbitrary l. Generated by real_solid_harmonics() (Use A) and cached: the
// (2l+1) pure-spherical functions span the same space as CP2K's, and the overall
// per-function scale is irrelevant to the energy (basis-norm invariance). l<=2
// reproduces the canonical s; p:{x,y,z}; d:{xy,yz,xz,x^2-y^2,2z^2-x^2-y^2}.
constexpr int kHarmLMax = 6; // s..i; covers every GTH/MOLOPT basis shell.
const std::vector<std::vector<std::vector<CartTerm>>>& harmonics() {
    static const std::vector<std::vector<std::vector<CartTerm>>> H = [] {
        std::vector<std::vector<std::vector<CartTerm>>> table;
        table.reserve(kHarmLMax + 1);
        for (int l = 0; l <= kHarmLMax; ++l) {
            std::vector<std::vector<CartTerm>> shell;
            for (const auto& harm : real_solid_harmonics(l)) {
                std::vector<CartTerm> terms;
                for (const auto& t : harm)
                    terms.push_back({{t.lx, t.ly, t.lz}, t.coeff});
                shell.push_back(std::move(terms));
            }
            table.push_back(std::move(shell));
        }
        return table;
    }();
    return H;
}
} // namespace

GpwAtom::GpwAtom(const double pos[3], const GaussianBasis& basis,
                 const GthPotential& pot, int z) {
    this->z_val = static_cast<double>(pot.zion);
    this->z = z;
    this->pos[0] = pos[0];
    this->pos[1] = pos[1];
    this->pos[2] = pos[2];
    this->rloc = pot.rloc;
    this->cloc = pot.cloc;
    this->nonlocal = pot.channels;

    for (const auto& set : basis.sets) {
        int col = 0;
        for (int l = set.lmin; l <= set.lmax; ++l) {
            const int L = l;
            for (int sh = 0; sh < set.nshell_l(l); ++sh, ++col) {
                for (const auto& harm :
                     harmonics()[static_cast<std::size_t>(L)]) {
                    GridBasisFunction bf;
                    for (int e = 0; e < set.nexp(); ++e) {
                        const double a = set.exps[static_cast<std::size_t>(e)];
                        const double c = set.coef_at(e, col);
                        if (c == 0.0)
                            continue;
                        // N(a,L) = (2a/pi)^{3/4} (4a)^{L/2}: the L alpha-power fixes
                        // the spanned space to CP2K's; constant factors cancel.
                        const double norm = std::pow(2.0 * a / M_PI, 0.75)
                                            * std::pow(4.0 * a, 0.5 * L);
                        for (const auto& term : harm) {
                            GridPrimitive p;
                            p.alpha = a;
                            p.center[0] = pos[0];
                            p.center[1] = pos[1];
                            p.center[2] = pos[2];
                            p.l[0] = term.l[0];
                            p.l[1] = term.l[1];
                            p.l[2] = term.l[2];
                            p.coeff = c * norm * term.w;
                            bf.push_back(p);
                        }
                    }
                    if (!bf.empty())
                        this->basis.push_back(std::move(bf));
                }
            }
        }
    }
}

/// The full GPW SCF implementation, owned and reached through
/// `GPWSolver::run`.
static GpwResult run_gpw_scf_impl(const std::vector<GpwAtom>& atoms,
                                  const GpwConfig& cfg,
                                  const RealMatrix* guess_C) {
#ifdef ATV_GPW_FFTW_THREADS
    // One-time global FFTW threading init; fftw_plan_with_nthreads is a global
    // setting, so every FFT plan created downstream (Poisson, GGA grad/div, MT)
    // runs multithreaded. The density-grid FFTs dominate GPW's wall time.
    static const int kFftwThreadsOk = fftw_init_threads();
    if (kFftwThreadsOk)
        fftw_plan_with_nthreads(omp_get_max_threads());
#endif
    const std::vector<GridBasisFunction> basis = global_basis(atoms);
    const int nb = static_cast<int>(basis.size());
    UniformGrid grid = make_uniform_grid(cfg.cell, cfg.cutoff_ry);

    // Γ-point lattice-image extent, shared by the analytic S/T/Vpp lattice sums
    // and the grid collocation. A Gaussian *pair* reaches up to twice the largest
    // single-function radius; images within that distance contribute. EPS_DEFAULT
    // sets the radius (CP2K's Gaussian-tail tolerance).
    const bool periodic = !cfg.nonperiodic;
    const double eps_default = cfg.eps_default;
    std::vector<double> radii(static_cast<std::size_t>(nb));
    double rmax = 0.0;
    for (int i = 0; i < nb; ++i) {
        radii[static_cast<std::size_t>(i)] =
            basis_function_radius(basis[i], eps_default);
        rmax = std::max(rmax, radii[static_cast<std::size_t>(i)]);
    }
    int mimg[3];
    pair_image_counts(cfg.cell, periodic, 2.0 * rmax, mimg);

    // Atom cell list + widest PP width, for the O(nb²) (was O(nb²·natoms)) GTH
    // build: each basis-pair queries only atoms near its product centre.
    double rmax_pp = 0.0;
    for (const auto& at : atoms) {
        double r = at.rloc;
        for (const auto& ch : at.nonlocal)
            r = std::max(r, ch.r);
        rmax_pp = std::max(rmax_pp, r);
    }
    AtomCells acells;
    // ATV_GPW_NOCELLS forces one giant cell (query returns all atoms) — the
    // pre-cell-list path, for bit-identity A/B verification.
    acells.build(atoms, cfg.cell, std::getenv("ATV_GPW_NOCELLS") ? 1e30 : 3.0,
                 periodic);

    // Grimme DFT-D3 (zero damping, PBE) dispersion — a geometry-only additive
    // correction (independent of the SCF), folded into the final total energy.
    double e_d3 = 0.0;
    if (cfg.d3) {
        std::vector<int> Z(atoms.size());
        std::vector<std::array<double, 3>> pos(Z.size());
        for (std::size_t i = 0; i < Z.size(); ++i) {
            Z[i] = atoms[i].z;
            pos[i] = {atoms[i].pos[0], atoms[i].pos[1], atoms[i].pos[2]};
        }
        // In-tree D3 tables only (basis_data/dftd3_tables.h) — coverage is the
        // supported element set; anything outside it throws (no fallback).
        for (int z : Z)
            if (!D3Params::supports_element(z))
                throw std::runtime_error(
                    "GPW D3: element Z=" + std::to_string(z)
                    + " is not in the in-tree D3 table (no fallback)");
        e_d3 = dftd3_pbe(Z, pos, cfg.cell, periodic, D3Params{}).energy;
    }

    // Geometry-fixed matrices.
    RealMatrix S = xt::zeros<double>({nb, nb});
    RealMatrix T = xt::zeros<double>({nb, nb});
    RealMatrix Vpp = xt::zeros<double>({nb, nb}); // V_loc^sr + V_nl
    // The geometry-fixed one-electron / GTH matrices dominate the GPW SCF wall
    // time at scale (profiled: ~75-80% for hundreds of atoms; the eigensolve is
    // <2%). Each (i,j) is an independent primitive contraction → parallelize the
    // outer index. The atom loop inside gth is cell-list screened (acells), and
    // the outer pair loop is cell-list screened (bcells, below) → O(nb·neigh).
    const bool sprof = std::getenv("ATV_GPW_PROF") != nullptr;
    auto sclk = std::chrono::steady_clock::now();
    auto smark = [&](const char* what) {
        if (!sprof)
            return;
        auto now = std::chrono::steady_clock::now();
        std::fprintf(
            stderr, "[gpw setup] %-14s %.0f ms\n", what,
            std::chrono::duration<double, std::milli>(now - sclk).count());
        sclk = now;
    };
    // Basis-center cell list: the outer S/T/Vpp build is O(nb²) only because every
    // pair is screened; the overlaps are short-ranged, so visit only neighbours.
    // ATV_GPW_NOCELLS forces one giant cell (query returns all j) for A/B bit-id.
    BasisCells bcells;
    bcells.build(basis, cfg.cell, std::getenv("ATV_GPW_NOCELLS") ? 1e30 : rmax,
                 periodic);
    // Local short-range PP: by default collocated on the grid (local_pp_grid +
    // V_eff, O(grid)); local_pp_on_grid=false selects the per-pair analytic
    // gth_local_sr (O(nb²·natoms)), kept for grid-vs-analytic energy validation.
    // gpw_forces_analytic reads the SAME flag — the gradient must be taken of the
    // functional the SCF actually converged, or the two disagree by the grid
    // discretization error of V_loc^SR (see internal/local_pp.h).
    const bool grid_loc = cfg.local_pp_on_grid;
#pragma omp parallel for schedule(dynamic, 8)
    for (int i = 0; i < nb; ++i) {
        if (basis[i].empty())
            continue;
        const double* ci = basis[i][0].center;
        const double Rq =
            radii[static_cast<std::size_t>(i)] + rmax; // conservative
        bcells.query(ci, Rq, [&](int j) {
            // Overlap pair screening: S, T, Vpp are all bounded by the Gaussian
            // product magnitude of (i,j). If even the best primitive pair is
            // below the screen, the whole element is negligible -> leave 0.
            if (pair_overlap_bound(basis[i], basis[j], cfg.cell, periodic)
                < kBuildScreen)
                return;
            const double reach = radii[static_cast<std::size_t>(i)]
                                 + radii[static_cast<std::size_t>(j)];
            S(i, j) = overlap(basis[i], basis[j], cfg.cell, mimg, reach);
            T(i, j) = kinetic(basis[i], basis[j], cfg.cell, mimg, reach);
            if (!grid_loc) // analytic local PP only when grid path is disabled
                Vpp(i, j) =
                    gth_matrix_element(basis[i], basis[j], atoms, acells,
                                       rmax_pp, cfg.cell, mimg, reach);
        });
    }
    // GTH nonlocal V_nl via the separable Kleinman-Bylander form (projectors
    // computed once per (basis,atom), not per basis pair) — adds onto Vpp.
    add_nonlocal_kb(Vpp, basis, atoms, radii, cfg.cell, mimg, bcells);
    const RealMatrix H0 = T + Vpp;
    smark("STVpp_build");

    // Geometry-fixed ionic density and the constant self-energy to remove.
    const std::vector<double> n_ion =
        ionic_density(grid, atoms, !cfg.nonperiodic);
    double e_self = 0.0;
    for (const auto& a : atoms)
        e_self += core_self_energy(a);
    smark("ionic_density");

    // Geometry-fixed local short-range PP on the grid (added to V_eff each step);
    // empty when the analytic per-pair path is selected.
    const std::vector<double> v_locsr =
        grid_loc ? local_pp_grid(grid, atoms, !cfg.nonperiodic)
                 : std::vector<double>{};
    smark("local_pp_grid");

    // Core (H0) initial guess.
    // Warm start: use the supplied guess (skip the O(nb³) core diagonalization);
    // CANONICAL orthogonalization, not the full-rank Löwdin S^{-1/2}. In a dense
    // periodic cell the diffuse MOLOPT functions drive S toward linear dependence:
    // its smallest eigenvalues go to ~0, Löwdin AMPLIFIES those directions by
    // 1/sqrt(s), and the raw generalized solver (dsygvd) simply fails — a 13726-AO
    // slab single point aborted with `dsygvd failed, info=20590` (S not positive
    // definite), and ZnO's diagonalization SCF oscillated for 400 iterations without
    // converging. CanonicalOrth drops the near-null directions (s_i < lindep_threshold)
    // instead of amplifying them, exactly as cpu/scf does (hf-robust-02).
    //
    // Built lazily: the OT path only needs it for the final store_converged_state, so
    // a cold OT step still skips the O(nb³) build.
    std::optional<linalg::CanonicalOrth> orth_cache;
    auto orth = [&]() -> const linalg::CanonicalOrth& {
        if (!orth_cache) {
            orth_cache.emplace(S, cfg.lindep_threshold);
            const int m = static_cast<int>(orth_cache->X.shape(1));

            // An overlap matrix is positive definite BY CONSTRUCTION, so a negative
            // eigenvalue means S is WRONG — not merely ill-conditioned. This guard
            // caught a real one: the pair lattice sum was missing its outermost image
            // shell (image_counts covers |T| <= rcut, but a pair screens on
            // |R_A - R_B - T| <= reach and two centers can be a full cell edge apart).
            // A 13726-AO slab overlap came out at s_min = -1.92 (unit-diagonal);
            // with pair_image_counts it is -6.8e-14, and its spectrum now matches the
            // PySCF reference (s_max 19.65012 vs 19.650). See internal/periodic.h.
            //
            // NOTE this is NOT the same thing as the basis being near-linearly-
            // dependent, which for this system it genuinely is: PySCF puts 6761 of
            // 13726 overlap eigenvalues below 1e-8. Dropping those directions
            // (CanonicalOrth) is the correct physics, not a fallback.
            constexpr double kPsdTol = -1e-8;
            if (orth_cache->s_min < kPsdTol)
                throw std::runtime_error(
                    "GPW: the overlap matrix is NOT positive definite (s_min="
                    + std::to_string(orth_cache->s_min)
                    + "). An overlap matrix cannot have a negative eigenvalue, "
                      "so S is "
                      "wrong — the lattice sum is incomplete. Refusing to "
                      "orthogonalize "
                      "a non-Gram matrix.");

            if (orth_cache->n_dropped > 0)
                spdlog::warn(
                    "GPW: basis is near-linearly-dependent — dropped {} of {} "
                    "overlap directions (s_min={:.3e} < "
                    "lindep_threshold={:.1e})",
                    orth_cache->n_dropped, nb, orth_cache->s_min,
                    cfg.lindep_threshold);
            if (cfg.n_occ > m)
                throw std::runtime_error(
                    "GPW: n_occ=" + std::to_string(cfg.n_occ) + " exceeds the "
                    + std::to_string(m)
                    + " linearly independent basis directions that survive "
                      "lindep_threshold — the basis cannot represent the "
                      "occupied "
                      "manifold (no-fallback rule)");
        }
        return *orth_cache;
    };

    // cold start falls back to diagonalizing the core Hamiltonian H0.
    RealMatrix C0;
    if (guess_C != nullptr && static_cast<int>(guess_C->shape(0)) == nb
        && static_cast<int>(guess_C->shape(1)) == nb) {
        C0 = *guess_C;
    } else {
        C0 = std::get<1>(linalg::diagonalize_fock(H0, orth().X));
    }
    RealMatrix P = linalg::build_density_matrix(C0, cfg.n_occ, nb, 2.0);
    RealMatrix C_prev = C0; // level-shift needs the previous occupied space
    smark("init_eigensolve");

    // Basis-on-grid is geometry-fixed: evaluate the Gaussians once and reuse
    // across all SCF iterations. The molecule fills a small part of the box, so
    // also precompute the "active" grid points (where any basis function is
    // non-negligible) and collocate/integrate only over those.
    // Compact-support basis-on-grid: each Gaussian touches only the grid points
    // within its radius (CSR-by-point), built once. Replaces the dense
    // npts x nb matrix — prohibitive in memory at scale (ngrid*nb*8 bytes) and
    // O(na*nb^2) per iter; the compact path is O(sum_g k_g^2), k_g << nb.
    // Multigrid (ngrids>1 && rel_cutoff>0) collocates per-level products; the
    // single grid uses the cbg gather path. (Single-grid products lost an A/B —
    // cbg pre-contracts primitives into χ_u once, products redo primitive work
    // each iter — so products serve only the genuine multigrid.)
    const bool use_mg = cfg.ngrids > 1 && cfg.rel_cutoff > 0.0;
    const bool use_prod = use_mg;
    // Only the single-grid path uses cbg; the multigrid path collocates per-level
    // products instead. Building cbg on the fine grid is prohibitive at production
    // cutoff (CUTOFF 320 → ~8M points × diffuse functions = >100 GB of triples), so
    // skip it entirely under multigrid.
    const CompactBasisGrid cbg =
        use_prod ? CompactBasisGrid{}
                 : build_compact_basis_grid(grid, basis, periodic, eps_default);
    smark("cbg_build");

    // CP2K-style REL_CUTOFF multigrid (gated on ngrids>1 && rel_cutoff>0). Each
    // primitive-pair product of the density is assigned (by its combined exponent)
    // to the coarsest grid level that resolves it; diffuse products collocate on a
    // coarse grid (far fewer points). The per-level densities are prolonged to the
    // finest grid for the (nonlinear) Hartree+XC, and the fine potential is
    // restricted back to each level to integrate the KS matrix. ngrids==1 leaves
    // the single-grid path above untouched (bit-identical). The product list is
    // geometry-fixed (excludes P); P is applied per SCF iteration.
    MultiGrid mg;
    ProductSet products;
    // Inter-level transfer is spectral (band-limited FFT interpolation): a diffuse
    // density collocated on a coarse level is embedded into the fine grid's Fourier
    // spectrum at the same physical frequencies and zero-padded above the coarse
    // Nyquist (shape-exact, unlike trilinear). prolong_fft conserves mass natively
    // (the G=0 mode is preserved) and restrict_fft is its exact adjoint, so the KS
    // integration stays the exact gradient of the collocated energy and the SCF is
    // variational — no per-level mass-conservation correction is needed.
    if (use_prod) {
        const int nlev = use_mg ? cfg.ngrids : 1;
        const double rcut = use_mg
                                ? cfg.rel_cutoff
                                : cfg.cutoff_ry; // 1 level → all products on it
        mg = make_multigrid(cfg.cell, cfg.cutoff_ry, nlev, rcut);
        products =
            build_products(mg, basis, cfg.cell, rcut, eps_default, periodic);
    }

    GpwResult res;
    double e_prev = 0.0;

    // Convergence acceleration. With diis_depth > 0 we run Pulay DIIS on the KS
    // matrix (X = S^{-1/2} is geometry-fixed → built once); with diis_depth == 0
    // we fall back to the legacy linear density mixing. DIIS converges
    // transition-metal / semicore systems (Zn-q12, Na-q9) that linear mixing
    // stalls on, and is the prerequisite for scaling to large cells.
    const bool use_diis = cfg.diis_depth > 0;

    internal::GpwDiis diis(use_diis ? cfg.diis_depth : 1);
    // DIIS commutator-residual threshold: a converged step must have both a
    // stable energy and a near-zero [H,P]_S residual (guards against declaring
    // convergence on an energy plateau before self-consistency).
    constexpr double kDiisErrTol = 1e-5;
    // OT convergence: max element of the projected gradient (the SCF error
    // vector). Paired with the energy-stability test for a tight fixed point.
    constexpr double kOtGradTol = 1e-6;

    // KS matrix + total energy from a density matrix P (collocate -> Poisson +
    // XC -> integrate). Fills res's energy terms; shared by the diagonalization
    // and OT paths.
    const bool prof = std::getenv("ATV_GPW_PROF") != nullptr;
    using ProfClk = std::chrono::steady_clock;
    auto pms = [](ProfClk::time_point a, ProfClk::time_point b) {
        return std::chrono::duration<double, std::milli>(b - a).count();
    };
    auto build_H_and_energy =
        [&](const RealMatrix& Pin) -> std::pair<RealMatrix, double> {
        auto t0p = ProfClk::now();
        const std::vector<double> Pvec = mat_to_vec(Pin, nb);
        // Density on the finest grid. Single-grid: direct compact collocation.
        // Multigrid: collocate each level's products, prolong to the fine grid.
        std::vector<double> rho;
        if (use_prod) {
            rho.assign(static_cast<std::size_t>(mg.levels[0].npts()), 0.0);
            for (std::size_t l = 0; l < mg.levels.size(); ++l) {
                std::vector<double> rho_l = collocate_products(
                    mg.levels[l], products.by_level[l], products.prims,
                    products.contribs, nb, Pvec, eps_default);
                if (l == 0) {
                    rho =
                        std::move(rho_l); // finest level is the reference grid
                } else {
                    // Spectral (band-limited) prolongation to the fine grid:
                    // shape-exact and mass-conserving (no D_l correction).
                    const std::vector<double> up = prolong_fft(
                        mg.levels[l], mg.levels[0], rho_l, periodic);
                    for (std::size_t g = 0; g < rho.size(); ++g)
                        rho[g] += up[g];
                }
            }
        } else {
            rho = collocate_density(cbg, nb, Pvec);
        }
        auto t1p = ProfClk::now();
        if (use_mg && std::getenv("ATV_GPW_MG_TRACE")) {
            double qe = 0.0;
            for (double v : rho)
                qe += v;
            qe *= grid.weight();
            std::fprintf(stderr,
                         "[gpw mg] int rho_fine = %.10f (Tr(P S) check)\n", qe);
        }
        std::vector<double> n_tot(rho.size());
        for (std::size_t g = 0; g < rho.size(); ++g)
            n_tot[g] = rho[g] - n_ion[g];
        const PoissonResult es = cfg.nonperiodic ? solve_poisson_mt(grid, n_tot)
                                                 : solve_poisson(grid, n_tot);
        auto t2p = ProfClk::now();
        const XcOnGrid xc = cfg.xc == XcKind::PBE ? xc_grid_pbe(grid, rho)
                                                  : xc_grid_pade(grid, rho);
        auto t3p = ProfClk::now();
        std::vector<double> veff(rho.size());
        for (std::size_t g = 0; g < rho.size(); ++g)
            veff[g] = es.v_r[g] + xc.vxc_r[g];
        // Add the geometry-fixed local short-range PP and its energy ∫ρ·V_loc^SR
        // (the grid form of the analytic Tr(P·V_loc^SR)).
        double e_locgrid = 0.0;
        if (!v_locsr.empty()) {
            const double gw = grid.weight();
            for (std::size_t g = 0; g < rho.size(); ++g) {
                veff[g] += v_locsr[g];
                e_locgrid += rho[g] * v_locsr[g];
            }
            e_locgrid *= gw;
        }
        // KS grid matrix. Single-grid: integrate veff directly. Multigrid:
        // restrict the fine potential to each level (the transpose of prolong) and
        // integrate each level's products — the adjoint of the density assembly,
        // so the KS matrix stays the exact gradient of the collocated energy.
        RealMatrix Vgrid;
        if (use_prod) {
            std::vector<double> V_ks(static_cast<std::size_t>(nb) * nb, 0.0);
            for (std::size_t l = 0; l < mg.levels.size(); ++l) {
                std::vector<double> V_l;
                if (l == 0) {
                    V_l = veff;
                } else {
                    // Exact adjoint of the spectral prolongation used for the
                    // density: restrict the fine potential by spectral truncation,
                    // so the KS matrix is the exact gradient of the collocated
                    // energy (variational SCF).
                    V_l = restrict_fft(mg.levels[0], mg.levels[l], veff,
                                       periodic);
                }
                const std::vector<double> Vl_ks = integrate_products(
                    mg.levels[l], products.by_level[l], products.prims,
                    products.contribs, nb, V_l, eps_default);
                for (std::size_t kk = 0; kk < V_ks.size(); ++kk)
                    V_ks[kk] += Vl_ks[kk];
            }
            Vgrid = vec_to_mat(V_ks, nb);
        } else {
            Vgrid = vec_to_mat(integrate_potential(cbg, nb, veff), nb);
        }
        auto t4p = ProfClk::now();
        if (prof)
            std::fprintf(stderr,
                         "[gpw prof] collocate=%.0f poisson=%.0f xc=%.0f "
                         "integrate=%.0f ms "
                         "(ngrid=%ld)\n",
                         pms(t0p, t1p), pms(t1p, t2p), pms(t2p, t3p),
                         pms(t3p, t4p), grid.npts());
        const RealMatrix H = H0 + Vgrid;
        res.e_kinetic = trace_product(Pin, T, nb);
        res.e_nonlocal = 0.0; // folded into e_loc_sr below for the first cut
        // Vpp holds the analytic PP terms still computed per pair: V_nl (always)
        // plus V_loc^SR when grid_loc is off. The grid local-PP energy e_locgrid
        // (zero when analytic) is the ∫ρ·V_loc^SR that Tr(P·Vpp) would otherwise
        // carry — together they total the full short-range PP energy.
        res.e_loc_sr = trace_product(Pin, Vpp, nb) + e_locgrid;
        res.e_es = es.e_hartree - e_self;
        res.e_xc = xc.e_xc;
        return {H, res.e_kinetic + res.e_loc_sr + res.e_es + res.e_xc};
    };

    // ── OT (orbital transformation) path ────────────────────────────────────
    // Diagonalization-free SCF: each outer step rebuilds H, then takes one
    // preconditioned-CG step on the OT manifold (gradient O(nb²·n_occ), no
    // O(nb³) eigensolve). Converges to the same gapped ground state as the
    // diagonalization path. The line search uses the frozen-H band energy
    // 2 Tr(CᵀHC) (matrix-only); self-consistency comes from the H rebuild.
    if (cfg.use_ot) {
        RealMatrix Cocc = internal::ot_orthonormalize(
            RealMatrix(xt::view(C0, xt::all(), xt::range(0, cfg.n_occ))), S);
        auto frob = [](const RealMatrix& a, const RealMatrix& b) {
            return xt::sum(a * b)();
        };
        RealMatrix d, pz_prev;
        double denom_prev = 1.0;
        bool have_prev = false;
        const bool prof = std::getenv("ATV_GPW_OT_PROF") != nullptr;
        using Clk = std::chrono::steady_clock;
        auto ms = [](Clk::time_point a, Clk::time_point b) {
            return std::chrono::duration<double, std::milli>(b - a).count();
        };
        for (int iter = 0; iter < cfg.max_iter; ++iter) {
            auto tA = Clk::now();
            P = linalg::build_density_matrix(Cocc, cfg.n_occ, nb,
                                             2.0); // 2 Cocc Coccᵀ
            auto [H, e_total] = build_H_and_energy(P);
            auto tB = Clk::now();
            res.iters = iter + 1;
            res.e_total = e_total;
            int ls_rebuilds = 0;

            const RealMatrix g = internal::ot_gradient(H, S, Cocc);
            double gnorm = 0.0;
            for (double v : g)
                gnorm = std::max(gnorm, std::abs(v));
            if (std::getenv("ATV_GPW_TRACE"))
                std::fprintf(
                    stderr,
                    "[gpw OT %2d] E=%.8f  T=%.4f loc=%.4f es=%.4f xc=%.4f  "
                    "dE=%.2e gnorm=%.2e\n",
                    iter, e_total, res.e_kinetic, res.e_loc_sr, res.e_es,
                    res.e_xc, std::abs(e_total - e_prev), gnorm);
            if (iter > 0 && gnorm < kOtGradTol
                && std::abs(e_total - e_prev) < cfg.tol) {
                res.converged = true;
                store_converged_state(res, H, orth().X, cfg.n_occ, nb);
                break;
            }
            e_prev = e_total;

            // FULL_SINGLE_INVERSE preconditioner: H-aware, rebuilt each step
            // (CP2K's default for gapped systems; FULL_KINETIC T+S was too weak
            // and OT crawled). M = H + a S with a below the spectrum -> SPD.
            const internal::OtPrecondSingleInverse precond(H, S, Cocc);
            const RealMatrix pz =
                internal::ot_tangent(Cocc, S, precond.apply(g));
            auto tC = Clk::now();
            if (!have_prev) {
                d = -pz;
            } else {
                const double beta =
                    std::max(0.0, frob(g, pz - pz_prev) / denom_prev);
                d = internal::ot_tangent(Cocc, S, -pz + beta * d);
            }
            pz_prev = pz;
            denom_prev = frob(g, pz);
            have_prev = true;

            // Monotonic line search on the SCF TOTAL energy E[rho(C)]. The
            // frozen-H band energy 2 Tr(CᵀHC) is only the local quadratic model;
            // minimizing *it* overshoots the self-consistent E and let OT wander
            // (descend then drift up). Armijo backtracking on the rebuilt-H total
            // energy enforces monotonic descent — every accepted step lowers E.
            auto etot = [&](const RealMatrix& C) {
                return build_H_and_energy(
                           linalg::build_density_matrix(C, cfg.n_occ, nb, 2.0))
                    .second;
            };
            double slope = frob(g, d);
            if (slope
                >= 0.0) { // not a descent direction -> restart to (precond) SD
                d = internal::ot_tangent(Cocc, S, -pz);
                slope = frob(g, d);
            }
            // Curvature-aware initial step from the frozen-H band energy (cheap,
            // no grid), then Armijo-backtrack on the rebuilt-H total energy. The
            // band parabola gives a near-optimal first alpha so backtracking
            // (each trial = a full grid build) is rare.
            auto band = [&](const RealMatrix& C) {
                return 2.0 * frob(C, xt::linalg::dot(H, C));
            };
            const double Eb0 = band(Cocc);
            const double Ebt = band(internal::ot_parameterize(Cocc, d, S));
            const double curv = 2.0 * (Ebt - Eb0 - slope); // parabola curvature
            double alpha = (curv > 1e-300) ? -slope / curv : 1.0; // vertex
            if (!(alpha > 0.0))
                alpha = 1.0;
            alpha = std::min(alpha, 2.0);
            constexpr double c1 = 1e-4;
            RealMatrix Cnew = internal::ot_parameterize(Cocc, alpha * d, S);
            for (int bt = 0; bt < 50; ++bt) {
                ++ls_rebuilds;
                if (etot(Cnew) <= e_total + c1 * alpha * slope)
                    break;
                alpha *= 0.5;
                Cnew = internal::ot_parameterize(Cocc, alpha * d, S);
            }
            Cocc = internal::ot_orthonormalize(Cnew, S);
            if (prof) {
                auto tD = Clk::now();
                std::fprintf(stderr,
                             "[OT prof %2d] H_grad=%.0fms precond=%.0fms "
                             "linesearch=%.0fms (%d rebuilds) | total=%.0fms\n",
                             iter, ms(tA, tB), ms(tB, tC), ms(tC, tD),
                             ls_rebuilds, ms(tA, tD));
            }
        }
        // Always expose the final occupied orbitals as res.C (first n_occ columns),
        // so the next step / an ASPC predictor can warm-start even if this solve
        // plateaued short of the tolerance.
        if (static_cast<int>(res.C.shape(0)) != nb) {
            res.C = xt::zeros<double>({nb, nb});
            for (int u = 0; u < nb; ++u)
                for (int i = 0; i < cfg.n_occ; ++i)
                    res.C(u, i) = Cocc(u, i);
        }
        res.e_total += e_d3;
        return res;
    }

    for (int iter = 0; iter < cfg.max_iter; ++iter) {
        const auto [H, e_total] = build_H_and_energy(P);

        res.iters = iter + 1;
        res.e_total = e_total;

        if (use_diis) {
            // Pulay DIIS: extrapolate the KS matrix from the (H, residual)
            // history and rebuild P directly from its eigenvectors (no density
            // mixing — DIIS does the averaging).
            const RealMatrix err = internal::gpw_diis_error(H, P, S, orth().X);
            double err_norm = 0.0;
            for (const double v : err)
                err_norm = std::max(err_norm, std::abs(v));

            if (std::getenv("ATV_GPW_TRACE"))
                std::fprintf(
                    stderr,
                    "[gpw it %2d] E=%.8f  T=%.4f loc=%.4f es=%.4f xc=%.4f  "
                    "dE=%.2e err=%.2e\n",
                    iter, e_total, res.e_kinetic, res.e_loc_sr, res.e_es,
                    res.e_xc, std::abs(e_total - e_prev), err_norm);

            const bool no_stop = std::getenv("ATV_GPW_NOSTOP") != nullptr;
            if (!no_stop && iter > 0 && std::abs(e_total - e_prev) < cfg.tol
                && err_norm < kDiisErrTol) {
                res.converged = true;
                store_converged_state(res, H, orth().X, cfg.n_occ, nb);
                break;
            }
            e_prev = e_total;

            const RealMatrix H_ext = diis.step(H, err);
            auto [eig, C] = diagonalize_fock_shifted(
                H_ext, orth().X, S, C_prev, cfg.n_occ, cfg.level_shift);
            C_prev = C;
            P = linalg::build_density_matrix(C, cfg.n_occ, nb, 2.0);
        } else {
            // Legacy linear density mixing.
            auto [eig, C] = diagonalize_fock_shifted(
                H, orth().X, S, C_prev, cfg.n_occ, cfg.level_shift);
            C_prev = C;
            RealMatrix P_new =
                linalg::build_density_matrix(C, cfg.n_occ, nb, 2.0);
            if (iter > 0 && std::abs(e_total - e_prev) < cfg.tol) {
                res.converged = true;
                store_converged_state(res, H, orth().X, cfg.n_occ, nb);
                break;
            }
            e_prev = e_total;
            P = (1.0 - cfg.mix) * P + cfg.mix * P_new;
        }
    }
    res.e_total += e_d3;
    return res;
}

// ── GPWSolver — the public solver face over run_gpw_scf_impl ────────

GPWSolver::GPWSolver(GpwConfig config) : config_(std::move(config)) {}

void GPWSolver::build(const std::vector<GpwAtom>& reference) {
    if (reference.empty())
        throw std::invalid_argument(
            "GPWSolver::build: empty atom set (nothing to solve)");
    if (config_.n_occ < 1)
        throw std::invalid_argument("GPWSolver::build: n_occ must be >= 1 (got "
                                    + std::to_string(config_.n_occ) + ")");
    if (!config_.nonperiodic) {
        for (int d = 0; d < 3; ++d)
            if (!(config_.cell[d] > 0.0))
                throw std::invalid_argument(
                    "GPWSolver::build: periodic cell edge " + std::to_string(d)
                    + " must be > 0 Bohr (got "
                    + std::to_string(config_.cell[d]) + ")");
    }
    built_ = true;
}

GpwResult& GPWSolver::run(const std::vector<GpwAtom>& atoms,
                          const RealMatrix* guess_C) {
    if (!built_)
        throw std::logic_error(
            "GPWSolver::run called before build() — validate the "
            "configuration first (no silent auto-build)");
    result_ = run_gpw_scf_impl(atoms, config_, guess_C);
    return result_;
}

namespace {
// Displace an atom (its position AND all its basis-function centers) by d.
GpwAtom displaced(GpwAtom a, int axis, double d) {
    a.pos[axis] += d;
    for (auto& bf : a.basis)
        for (auto& p : bf)
            p.center[axis] += d;
    return a;
}
} // namespace

std::vector<std::array<double, 3>>
gpw_forces_fd(const std::vector<GpwAtom>& atoms, const GpwConfig& cfg,
              double h) {
    const int na = static_cast<int>(atoms.size());
    std::vector<std::array<double, 3>> forces(na, {0.0, 0.0, 0.0});
    for (int a = 0; a < na; ++a) {
        for (int k = 0; k < 3; ++k) {
            std::vector<GpwAtom> plus = atoms, minus = atoms;
            plus[a] = displaced(atoms[a], k, +h);
            minus[a] = displaced(atoms[a], k, -h);
            GPWSolver plus_solver{cfg};
            plus_solver.build(plus);
            const double e_plus = plus_solver.run(plus).e_total;
            GPWSolver minus_solver{cfg};
            minus_solver.build(minus);
            const double e_minus = minus_solver.run(minus).e_total;
            forces[a][k] = -(e_plus - e_minus) / (2.0 * h); // F = -dE/dR
        }
    }
    return forces;
}

} // namespace gpw
} // namespace cpu
} // namespace ATV
