#include <chrono>
#include <cstdio>
#include <cstdlib>
/// @file src/cpu/gpw/gpw_solver.cpp
/// @brief Implementation of the GPW SCF — RKS and UKS (see header).
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
#include <tuple>
#include <utility>
#include <vector>

#include "atomiverse/core/real_harmonics.h"
#include "atomiverse/core/units.h"
#include "atomiverse/cpu/gpw/internal/dft_plus_u.h"
#include "atomiverse/cpu/gpw/internal/dftd3.h"
#include "atomiverse/cpu/gpw/internal/diis.h"
#include "atomiverse/cpu/gpw/internal/gpw_workspace.h"
#include "atomiverse/cpu/gpw/internal/grid_fft.h"
#include "atomiverse/cpu/gpw/internal/ks_grid_build.h"
#include "atomiverse/cpu/gpw/internal/gth_integrals.h"
#include "atomiverse/cpu/gpw/internal/hartree_grid.h"
#include "atomiverse/cpu/gpw/internal/hirshfeld_volumes.h"
#include "atomiverse/cpu/gpw/internal/kpoints.h"
#include "atomiverse/cpu/gpw/internal/local_pp.h"
#include "atomiverse/cpu/gpw/internal/mbd_dispersion.h"
#include "atomiverse/cpu/gpw/internal/mt_poisson.h"
#include "atomiverse/cpu/gpw/internal/multigrid.h"
#include "atomiverse/cpu/gpw/internal/one_electron.h"
#include "atomiverse/cpu/gpw/internal/ot.h"
#include "atomiverse/cpu/gpw/internal/periodic.h"
#include "atomiverse/cpu/gpw/internal/poisson_fft.h"
#include "atomiverse/cpu/gpw/internal/rvv10.h"
#include "atomiverse/cpu/gpw/internal/ts_dispersion.h"
#include "atomiverse/cpu/gpw/internal/vdw_df.h"
#include "atomiverse/cpu/gpw/internal/xc_grid.h"
#include "atomiverse/cpu/gpw/internal/xc_grid_gga.h"
#include "atomiverse/cpu/gpw/internal/xdm_dispersion.h"
#include "atomiverse/cpu/linalg/density.h"    // build_density_matrix
#include "atomiverse/cpu/linalg/eigensolve.h" // diagonalize_fock_generalized
#include "atomiverse/cpu/linalg/matrix.h"     // RealMatrix
#include "atomiverse/cpu/planewave/internal/smearing.h" // metallic occupations
#include "atomiverse/cpu/scf/dftd4.h"

#include <xtensor-blas/xlinalg.hpp>
#include <xtensor/views/xview.hpp>

#ifdef ATV_GPW_FFTW_THREADS
#include <fftw3.h>
#include <omp.h>
#endif

#ifdef ATV_HAS_MPI
#include <mpi.h>
#endif

namespace ATV {
namespace cpu {
namespace gpw {

namespace {

using ATV::cpu::linalg::RealMatrix;
using ATV::cpu::linalg::RealVector;

/// @brief MPI layout for k-point round-robin (mirrors `cpu::pw` scf_loop).
struct GpwMpiLayout {
    int rank = 0;
    int nprocs = 1;
    bool active = false; ///< true only when MPI is built AND nprocs > 1.
};

GpwMpiLayout gpw_mpi_layout() {
    GpwMpiLayout layout;
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

// Flatten all atoms' basis functions into one global list.
std::vector<GridBasisFunction> global_basis(const std::vector<GpwAtom>& atoms) {
    std::vector<GridBasisFunction> b;
    for (const auto& a : atoms)
        for (const auto& bf : a.basis)
            b.push_back(bf);
    return b;
}

// Parallel AO tags (atom index + ℓ) in the same order as global_basis.
std::vector<internal::AoShellTag>
global_ao_tags(const std::vector<GpwAtom>& atoms) {
    std::vector<internal::AoShellTag> tags;
    for (int ia = 0; ia < static_cast<int>(atoms.size()); ++ia) {
        const auto& a = atoms[static_cast<std::size_t>(ia)];
        const int nbf = static_cast<int>(a.basis.size());
        const int nl = static_cast<int>(a.basis_l.size());
        for (int k = 0; k < nbf; ++k) {
            internal::AoShellTag t;
            t.atom = ia;
            t.l = (k < nl) ? a.basis_l[static_cast<std::size_t>(k)] : -1;
            tags.push_back(t);
        }
    }
    return tags;
}

// Convert public PlusU (eV) list into solver-side (Z, l, U_ha) tables.
void pack_plus_u(const std::vector<PlusU>& plus_u, std::vector<int>& z_keys,
                 std::vector<internal::PlusUHa>& slots) {
    z_keys.clear();
    slots.clear();
    for (const auto& pu : plus_u) {
        if (!(pu.u_minus_j_ev > 0.0) || pu.l < 0)
            continue;
        z_keys.push_back(static_cast<int>(pu.element));
        internal::PlusUHa s;
        s.l = pu.l;
        s.U_ha = pu.u_minus_j_ev / units::kHartreeInEv;
        slots.push_back(s);
    }
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
// the fix for the hundreds-× setup gap. The atom loop is parallel; each atom
// builds its own dense (μ,ν) block and the blocks are ADDED IN ASCENDING ATOM
// ORDER afterwards. The `#pragma omp atomic` this replaces let two atoms hitting
// one (μ,ν) land in thread-completion order, so V_nl — and every SCF that reads
// it — differed run to run at a fixed thread count.
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
    std::vector<std::vector<int>> atom_mus(static_cast<std::size_t>(natoms));
    std::vector<std::vector<double>> atom_block(
        static_cast<std::size_t>(natoms));
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
        std::vector<double> blk(static_cast<std::size_t>(ng) * ng, 0.0);
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
                blk[static_cast<std::size_t>(k1) * ng
                    + static_cast<std::size_t>(k2)] = s;
            }
        atom_mus[static_cast<std::size_t>(ai)] = std::move(mus);
        atom_block[static_cast<std::size_t>(ai)] = std::move(blk);
    }
    // Fixed accumulation order: ascending atom, then (k1, k2) within the atom.
    for (int ai = 0; ai < natoms; ++ai) {
        const std::vector<int>& mus = atom_mus[static_cast<std::size_t>(ai)];
        const std::vector<double>& blk =
            atom_block[static_cast<std::size_t>(ai)];
        const int ng = static_cast<int>(mus.size());
        for (int k1 = 0; k1 < ng; ++k1)
            for (int k2 = 0; k2 < ng; ++k2) {
                const double s = blk[static_cast<std::size_t>(k1) * ng
                                     + static_cast<std::size_t>(k2)];
                if (s == 0.0)
                    continue;
                Vpp(mus[static_cast<std::size_t>(k1)],
                    mus[static_cast<std::size_t>(k2)]) += s;
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

// Geometry-derived AO setup plus the image-summed Γ one-electron loop — the
// SINGLE definition shared by the SCF (`run_gpw_scf_impl`) and the standalone
// public entry points `GpwDft::overlap` / `GpwDft::overlap_k`. The SCF asks for
// S together with T and Vpp; a standalone overlap asks for S alone. Every
// matrix element is an independent screened pair contraction over the same cell
// list, the same screen and the same image reach, so the S written either way
// is bit-identical — which is what lets a CPMD step align its propagated
// coefficients against a standalone overlap and then evaluate the KS matrix
// exactly ONCE instead of twice (esmm-cpmd-06-single-eval).
struct OneElectronSetup {
    std::vector<GridBasisFunction> basis; ///< flattened AO basis.
    std::vector<double> radii; ///< per-AO Gaussian-tail radius [Bohr].
    double rmax = 0.0;         ///< widest AO radius [Bohr].
    int mimg[3] = {0, 0, 0};   ///< Γ pair-image half-counts per axis.

    // Γ-point lattice-image extent, shared by the analytic S/T/Vpp lattice sums
    // and the grid collocation. A Gaussian *pair* reaches up to twice the largest
    // single-function radius; images within that distance contribute. EPS_DEFAULT
    // sets the radius (CP2K's Gaussian-tail tolerance).
    OneElectronSetup(const std::vector<GpwAtom>& atoms, const double cell[3],
                     bool periodic, double eps_default)
        : basis(global_basis(atoms)), radii(basis.size(), 0.0) {
        for (std::size_t i = 0; i < basis.size(); ++i) {
            radii[i] = basis_function_radius(basis[i], eps_default);
            rmax = std::max(rmax, radii[i]);
        }
        pair_image_counts(cell, periodic, 2.0 * rmax, mimg);
    }

    int nb() const { return static_cast<int>(basis.size()); }

    // Real Γ lattice sums into pre-zeroed nb × nb matrices. `S` is always
    // written; `T` and `Vpp` only when the caller supplies them. Vpp covers the
    // analytic local GTH term (skipped when `grid_loc`, i.e. V_loc^SR is
    // collocated instead) plus the separable Kleinman-Bylander nonlocal.
    void gamma(const std::vector<GpwAtom>& atoms, const double cell[3],
               bool periodic, bool grid_loc, RealMatrix& S, RealMatrix* T,
               RealMatrix* Vpp) const {
        const int n = nb();
        // Atom cell list + widest PP width, for the O(nb²) (was O(nb²·natoms))
        // GTH build: each basis-pair queries only atoms near its product centre.
        double rmax_pp = 0.0;
        AtomCells acells;
        if (Vpp != nullptr) {
            for (const auto& at : atoms) {
                double r = at.rloc;
                for (const auto& ch : at.nonlocal)
                    r = std::max(r, ch.r);
                rmax_pp = std::max(rmax_pp, r);
            }
            // ATV_GPW_NOCELLS forces one giant cell (query returns all atoms) —
            // the pre-cell-list path, for bit-identity A/B verification.
            acells.build(atoms, cell,
                         std::getenv("ATV_GPW_NOCELLS") ? 1e30 : 3.0, periodic);
        }
        // Basis-center cell list: the outer S/T/Vpp build is O(nb²) only because
        // every pair is screened; the overlaps are short-ranged, so visit only
        // neighbours. ATV_GPW_NOCELLS forces one giant cell (query returns all j).
        BasisCells bcells;
        bcells.build(basis, cell, std::getenv("ATV_GPW_NOCELLS") ? 1e30 : rmax,
                     periodic);
#pragma omp parallel for schedule(dynamic, 8)
        for (int i = 0; i < n; ++i) {
            if (basis[i].empty())
                continue;
            const double* ci = basis[i][0].center;
            const double Rq =
                radii[static_cast<std::size_t>(i)] + rmax; // conservative
            bcells.query(ci, Rq, [&](int j) {
                // Overlap pair screening: S, T, Vpp are all bounded by the
                // Gaussian product magnitude of (i,j). If even the best
                // primitive pair is below the screen, the whole element is
                // negligible -> leave 0.
                if (pair_overlap_bound(basis[i], basis[j], cell, periodic)
                    < kBuildScreen)
                    return;
                const double reach = radii[static_cast<std::size_t>(i)]
                                     + radii[static_cast<std::size_t>(j)];
                S(i, j) = overlap(basis[i], basis[j], cell, mimg, reach);
                if (T != nullptr)
                    (*T)(i, j) = kinetic(basis[i], basis[j], cell, mimg, reach);
                if (Vpp != nullptr && !grid_loc) // analytic local PP only when
                                                 // the grid route is disabled
                    (*Vpp)(i, j) =
                        gth_matrix_element(basis[i], basis[j], atoms, acells,
                                           rmax_pp, cell, mimg, reach);
            });
        }
        // GTH nonlocal V_nl via the separable Kleinman-Bylander form
        // (projectors computed once per (basis,atom)) — adds onto Vpp.
        if (Vpp != nullptr)
            add_nonlocal_kb(*Vpp, basis, atoms, radii, cell, mimg, bcells);
    }
};

// Sampled k-points of a configuration: the explicit `k_points` list when
// non-empty, else the Monkhorst-Pack mesh. One definition, so the SCF k branch
// and the standalone `GpwDft::overlap_k` sample the same points in the same
// order (S_k parity is per-index).
std::vector<internal::KPoint> sampled_kpoints(const GpwDft& cfg) {
    std::vector<internal::KPoint> kpts;
    if (!cfg.k_points.empty()) {
        for (std::size_t i = 0; i < cfg.k_points.size(); ++i) {
            internal::KPoint kp;
            kp.frac[0] = cfg.k_points[i][0];
            kp.frac[1] = cfg.k_points[i][1];
            kp.frac[2] = cfg.k_points[i][2];
            kp.weight = cfg.k_point_weights[i];
            kpts.push_back(kp);
        }
        return kpts;
    }
    return internal::monkhorst_pack(cfg.k_mesh, cfg.k_time_reversal);
}

// True when this configuration solves on the k-point path (any non-Γ mesh,
// explicit k list, or metallic smearing — occupations live on the multi-k
// path, including effective Γ via k_mesh = {1,1,1}). Shared by
// `run_gpw_scf_impl` and the standalone overlap entry points, which are the
// Γ / k halves of the same split.
bool uses_kpath(const GpwDft& cfg) {
    return !cfg.k_points.empty() || cfg.k_mesh[0] != 1 || cfg.k_mesh[1] != 1
           || cfg.k_mesh[2] != 1 || cfg.smearing != Smearing::none;
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

// UKS: store α/β orbitals and Löwdin ⟨S²⟩.
// ⟨S²⟩ = s_z(s_z+1) + N_β − Σ_{i∈α,j∈β} |⟨φ_i^α|φ_j^β⟩|²
// with ⟨φ_i^α|φ_j^β⟩ = C_αᵀ S C_β (Löwdin 1955; Szabo & Ostlund §2.5).
void store_converged_state_uks(GpwResult& res, const RealMatrix& Ha,
                               const RealMatrix& Hb, const RealMatrix& X,
                               const RealMatrix& S, int n_alpha, int n_beta,
                               int nb) {
    auto [eiga, Ca] = linalg::diagonalize_fock(Ha, X);
    auto [eigb, Cb] = linalg::diagonalize_fock(Hb, X);
    res.C = Ca;
    res.C_beta = Cb;
    res.eps.assign(eiga.begin(), eiga.end());
    res.eps_beta.assign(eigb.begin(), eigb.end());
    res.P = linalg::build_density_matrix(Ca, n_alpha, nb, 1.0);
    res.P_beta = linalg::build_density_matrix(Cb, n_beta, nb, 1.0);

    const double sz = 0.5 * static_cast<double>(n_alpha - n_beta);
    double ov = 0.0;
    if (n_alpha > 0 && n_beta > 0) {
        const RealMatrix Ca_occ =
            xt::view(Ca, xt::all(), xt::range(0, n_alpha));
        const RealMatrix Cb_occ = xt::view(Cb, xt::all(), xt::range(0, n_beta));
        // Sab = C_a† S C_b  (n_alpha × n_beta)
        const RealMatrix Sab =
            xt::linalg::dot(xt::transpose(Ca_occ), xt::linalg::dot(S, Cb_occ));
        for (int i = 0; i < n_alpha; ++i)
            for (int j = 0; j < n_beta; ++j) {
                const double s = Sab(i, j);
                ov += s * s;
            }
    }
    res.s2 = sz * (sz + 1.0) + static_cast<double>(n_beta) - ov;
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
                    if (!bf.empty()) {
                        this->basis.push_back(std::move(bf));
                        this->basis_l.push_back(L);
                    }
                }
            }
        }
    }
}

/// k-point RKS SCF (Monkhorst-Pack / explicit list). Shares the Γ path's grid
/// pipeline (Poisson, XC, local-PP-on-grid) — the k-resolved pieces are the
/// complex Bloch matrices S_k/T_k/V_nl^k, the per-k pre-summed basis-on-grid
/// values, the per-k complex Hermitian diagonalization, and the k-summed
/// density / Pulay DIIS. Reached only through `run_gpw_scf_impl`.
///
/// MPI: when built with `ATV_HAS_MPI` and MPI is initialized with nprocs > 1,
/// k-points are distributed round-robin over `MPI_COMM_WORLD` (same layout as
/// `cpu::pw`). Local ranks collocate / diagonalize only their k; density,
/// kinetic/nonlocal traces, DIIS B-matrix, and residual norms are Allreduced
/// so every rank holds identical `e_total`. Setup matrices stay replicated
/// (foothold; not full domain decomp). nprocs==1 is bit-identical to serial.
static GpwResult run_gpw_scf_kpoints(
    const std::vector<GpwAtom>& atoms, const GpwDft& cfg,
    const std::vector<GridBasisFunction>& basis,
    const std::vector<double>& radii, const int mimg[3],
    const UniformGrid& grid, const std::vector<double>& n_ion,
    const std::vector<double>& v_locsr, double e_self, double e_d3,
    const std::vector<internal::KPoint>& kpts,
    const std::vector<linalg::CplxMatrix>* guess_P_k = nullptr) {
    using internal::CplxMatrix;
    using Cplx = std::complex<double>;
    const int nb = static_cast<int>(basis.size());
    const int nk = static_cast<int>(kpts.size());
    const GpwMpiLayout layout = gpw_mpi_layout();
    int n_kpts_local = 0;
    for (int ik = layout.rank; ik < nk; ik += layout.nprocs)
        ++n_kpts_local;

    // Validate multi-k XL density guess before any heavy setup.
    if (guess_P_k != nullptr) {
        if (static_cast<int>(guess_P_k->size()) != nk)
            throw std::invalid_argument("GPW kpoints: guess_P_k size ("
                                        + std::to_string(guess_P_k->size())
                                        + ") != n_kpts (" + std::to_string(nk)
                                        + ")");
        for (int ik = 0; ik < nk; ++ik) {
            const auto& Pk = (*guess_P_k)[static_cast<std::size_t>(ik)];
            if (static_cast<int>(Pk.shape(0)) != nb
                || static_cast<int>(Pk.shape(1)) != nb)
                throw std::invalid_argument(
                    "GPW kpoints: guess_P_k[" + std::to_string(ik)
                    + "] shape does not match AO basis (nb="
                    + std::to_string(nb) + ")");
        }
    }

    // Phased analytic matrices: S_k, T_k in one image sweep; nonlocal KB.
    // Replicated on every rank for this foothold (setup not parallelized).
    // ATV_GPW_PROF: [gpw setup k] for STVnl / products / kbg (matches Γ setup).
    const bool k_setup_prof_early = std::getenv("ATV_GPW_PROF") != nullptr;
    auto k_setup_clk_early = std::chrono::steady_clock::now();
    auto k_setup_mark_early = [&](const char* what) {
        if (!k_setup_prof_early || layout.rank != 0)
            return;
        auto now = std::chrono::steady_clock::now();
        std::fprintf(
            stderr, "[gpw setup k] %-14s %.0f ms\n", what,
            std::chrono::duration<double, std::milli>(now - k_setup_clk_early)
                .count());
        k_setup_clk_early = now;
    };
    std::vector<CplxMatrix> S_k, T_k;
    internal::build_overlap_kinetic_k(basis, radii, cfg.cell, mimg,
                                      kBuildScreen, kpts, S_k, T_k);
    std::vector<CplxMatrix> Vnl_k(static_cast<std::size_t>(nk),
                                  CplxMatrix(xt::zeros<Cplx>({nb, nb})));
    internal::add_nonlocal_kb_k(Vnl_k, basis, atoms, radii, cfg.cell, mimg,
                                kBuildScreen, kpts);
    k_setup_mark_early("STVnl_k");

    // Canonical orthogonalization per k (drops near-null Bloch directions).
    std::vector<internal::KOrth> orth;
    orth.reserve(static_cast<std::size_t>(nk));
    int m_min = 0;
    for (int ik = 0; ik < nk; ++ik) {
        orth.emplace_back(S_k[static_cast<std::size_t>(ik)],
                          cfg.lindep_threshold);
        const int m = static_cast<int>(orth.back().X.shape(1));
        if (ik == 0)
            m_min = m;
        else
            m_min = std::min(m_min, m);
        if (cfg.n_occ > m)
            throw std::runtime_error(
                "GPW kpoints: n_occ=" + std::to_string(cfg.n_occ)
                + " exceeds the " + std::to_string(m)
                + " independent basis directions at k #" + std::to_string(ik)
                + " (no-fallback rule)");
    }
    k_setup_mark_early("orth_k");

    // Metallic smearing (FD / MP) — shared μ over all k, Mermin free energy.
    // Occupation math is `cpu::pw::internal` (unit-tested); units: Ha inside.
    const bool use_smear = (cfg.smearing != Smearing::none);
    const double n_elec = 2.0 * static_cast<double>(cfg.n_occ);
    const double degauss_ha =
        use_smear ? cfg.degauss * 0.5 : 0.0; // Ry → Ha (1 Ry = 0.5 Ha)
    using pw::internal::SmearKind;
    SmearKind smear_kind = SmearKind::fermi_dirac;
    if (cfg.smearing == Smearing::methfessel_paxton)
        smear_kind = SmearKind::methfessel_paxton;
    const int mp_order = cfg.mp_order;
    if (use_smear && m_min < cfg.n_occ + std::max(cfg.extra_bands, 0)
        && std::getenv("ATV_GPW_TRACE") && layout.rank == 0)
        std::fprintf(stderr,
                     "[gpw k] smear: m_min=%d n_occ=%d extra_bands=%d "
                     "(full AO spectrum used)\n",
                     m_min, cfg.n_occ, cfg.extra_bands);

    std::vector<double> weights(static_cast<std::size_t>(nk));
    for (int ik = 0; ik < nk; ++ik)
        weights[static_cast<std::size_t>(ik)] =
            kpts[static_cast<std::size_t>(ik)].weight;

    // CP2K REL_CUTOFF multigrid on the k path (gated ngrids>1 && rel_cutoff>0):
    // the geometry-fixed product list is the SAME as the Γ multigrid — the ket
    // image shift each product already carries becomes the Bloch phase index.
    // Per iteration the k-summed real-space density-matrix stack
    //   D(t) = Σ_k w_k Re[P^k e^{−2πi κ·t}]
    // weighs the per-level product collocation (Γ limit: D(t) = P, the shipped
    // weight), the per-level densities prolong spectrally to the finest grid
    // for Poisson + XC, and the fine potential restricts back to each level to
    // integrate the real per-cell stack V_t — phase-summed per k into the
    // Hermitian grid KS matrix V^k = Σ_t e^{+2πi κ·t} V_t (exact transpose of
    // the density assembly, so the SCF stays variational). MPI: D and V_t are
    // k-independent, so each rank holds the full stacks (D Allreduced over the
    // local-k partials; the product sweep itself is replicated — foothold,
    // same replication class as the analytic S_k/T_k setup).
    const bool use_mg = cfg.ngrids > 1 && cfg.rel_cutoff > 0.0;
    // Solver-private bags (products / kbg / last_veff); grid is the caller's.
    internal::GpwWorkspace ws;
    ws.grid = grid;
    MultiGrid& mg = ws.mg;
    ProductSet& products = ws.products;
    if (use_mg) {
        k_setup_clk_early = std::chrono::steady_clock::now();
        mg = make_multigrid(cfg.cell, cfg.cutoff_ry, cfg.ngrids,
                            cfg.rel_cutoff);
        if (mg.levels[0].n[0] != grid.n[0] || mg.levels[0].n[1] != grid.n[1]
            || mg.levels[0].n[2] != grid.n[2])
            throw std::logic_error(
                "GPW kpoints: multigrid finest level n[] != SCF grid n[]");
        products = build_products(mg, basis, cfg.cell, cfg.rel_cutoff,
                                  cfg.eps_default, /*periodic=*/true);
        k_setup_mark_early("products_mg");
    }
    const std::size_t n_tcells = products.cells.size();
    // Bloch phase per (k, image cell): e^{−2πi κ·t} matches the χ^k grid
    // convention (kpoints.h); the KS assembly uses its conjugate.
    std::vector<std::vector<Cplx>> tcell_phase(static_cast<std::size_t>(nk));
    if (use_mg) {
        constexpr double kTwoPi = 6.283185307179586476925286766559;
        for (int ik = 0; ik < nk; ++ik) {
            auto& row = tcell_phase[static_cast<std::size_t>(ik)];
            row.resize(n_tcells);
            for (std::size_t c = 0; c < n_tcells; ++c) {
                const auto& t = products.cells[c];
                const double arg =
                    -kTwoPi
                    * (kpts[static_cast<std::size_t>(ik)].frac[0] * t[0]
                       + kpts[static_cast<std::size_t>(ik)].frac[1] * t[1]
                       + kpts[static_cast<std::size_t>(ik)].frac[2] * t[2]);
                row[c] = Cplx(std::cos(arg), std::sin(arg));
            }
        }
    }
    // Full (Allreduced) D stack from {P_k}: local-k partials, then the sum.
    auto mg_density_stack = [&](const std::vector<CplxMatrix>& Pk) {
        std::vector<std::vector<double>> D(
            n_tcells,
            std::vector<double>(static_cast<std::size_t>(nb) * nb, 0.0));
        for (int ik = layout.rank; ik < nk; ik += layout.nprocs) {
            const double wk = weights[static_cast<std::size_t>(ik)];
            const auto& P = Pk[static_cast<std::size_t>(ik)];
            for (std::size_t c = 0; c < n_tcells; ++c) {
                const Cplx ph = tcell_phase[static_cast<std::size_t>(ik)][c];
                double* Dc = D[c].data();
                const Cplx* Pd = P.data();
                const std::size_t nn = static_cast<std::size_t>(nb) * nb;
                for (std::size_t e = 0; e < nn; ++e)
                    Dc[e] += wk
                             * (Pd[e].real() * ph.real()
                                - Pd[e].imag() * ph.imag());
            }
        }
#ifdef ATV_HAS_MPI
        if (layout.active)
            for (auto& Dc : D)
                MPI_Allreduce(MPI_IN_PLACE, Dc.data(),
                              static_cast<int>(Dc.size()), MPI_DOUBLE, MPI_SUM,
                              MPI_COMM_WORLD);
#endif
        return D;
    };
    // Fine-grid density from {P_k}: per-level product collocation weighted by
    // the (full) D stack + spectral prolongation — every rank returns the
    // GLOBAL density (no further reduction).
    auto mg_density = [&](const std::vector<CplxMatrix>& Pk) {
        const auto D = mg_density_stack(Pk);
        std::vector<double> rho(static_cast<std::size_t>(grid.npts()), 0.0);
        for (std::size_t l = 0; l < mg.levels.size(); ++l) {
            std::vector<double> rho_l = collocate_products_cells(
                mg.levels[l], products.by_level[l], products.prims,
                products.contribs, nb, D, cfg.eps_default);
            if (l == 0) {
                rho = std::move(rho_l);
            } else {
                const std::vector<double> up =
                    prolong_fft(mg.levels[l], mg.levels[0], rho_l,
                                /*spectral*/ true);
                for (std::size_t g = 0; g < rho.size(); ++g)
                    rho[g] += up[g];
            }
        }
        return rho;
    };
    // Real per-cell KS stack V_t from the fine effective potential: restrict
    // per level (exact adjoint of the prolongation) + integrate each level's
    // products. k-independent — one stack serves every k via the phase sum.
    auto mg_vt_stack = [&](const std::vector<double>& veff) {
        std::vector<std::vector<double>> Vt(
            n_tcells,
            std::vector<double>(static_cast<std::size_t>(nb) * nb, 0.0));
        for (std::size_t l = 0; l < mg.levels.size(); ++l) {
            std::vector<double> V_l;
            if (l == 0) {
                V_l = veff;
            } else {
                V_l = restrict_fft(mg.levels[0], mg.levels[l], veff,
                                   /*spectral*/ true);
            }
            const auto Vt_l = integrate_products_cells(
                mg.levels[l], products.by_level[l], products.prims,
                products.contribs, nb, V_l, n_tcells, cfg.eps_default);
            for (std::size_t c = 0; c < n_tcells; ++c)
                for (std::size_t e = 0; e < Vt[c].size(); ++e)
                    Vt[c][e] += Vt_l[c][e];
        }
        return Vt;
    };
    // Per-k Hermitian grid KS matrix V^k = Σ_t e^{+2πi κ·t} V_t (conjugate of
    // the collocation phase).
    auto mg_phase_V = [&](const std::vector<std::vector<double>>& Vt, int ik) {
        CplxMatrix V = xt::zeros<Cplx>({nb, nb});
        const std::size_t nn = static_cast<std::size_t>(nb) * nb;
        for (std::size_t c = 0; c < n_tcells; ++c) {
            const Cplx ph =
                std::conj(tcell_phase[static_cast<std::size_t>(ik)][c]);
            const double* Vc = Vt[c].data();
            Cplx* out = V.data();
            for (std::size_t e = 0; e < nn; ++e)
                out[e] += ph * Vc[e];
        }
        return V;
    };

    // Per-k pre-summed complex basis-on-grid values (image sum + Bloch phase
    // folded once; per-iteration cost is O(npts * nb^2) per k). Skipped under
    // multigrid — the product path replaces it (dense χ^k on the fine grid is
    // exactly the cost multigrid exists to avoid).
    if (!use_mg)
        k_setup_clk_early = std::chrono::steady_clock::now();
    if (!use_mg)
        ws.kbg = internal::build_kpoint_basis_grid(grid, basis, kpts,
                                                   cfg.eps_default);
    const internal::KpointBasisGrid& kbg = ws.kbg;
    if (!use_mg)
        k_setup_mark_early("kbg_build");

    // Density from band occupations f_b ∈ [0, 2] (closed-shell band form).
    auto density_from_occ = [&](const CplxMatrix& C,
                                const std::vector<double>& occ) {
        const int m = static_cast<int>(C.shape(1));
        CplxMatrix P = xt::zeros<Cplx>({nb, nb});
        const int nfill = std::min(m, static_cast<int>(occ.size()));
        for (int o = 0; o < nfill; ++o) {
            const double fo = occ[static_cast<std::size_t>(o)];
            if (!(fo > 1e-14))
                continue;
            for (int i = 0; i < nb; ++i) {
                const Cplx ci = fo * C(i, o);
                for (int j = 0; j < nb; ++j)
                    P(i, j) += ci * std::conj(C(j, o));
            }
        }
        return P;
    };
    auto density_from_fixed = [&](const CplxMatrix& C, int n_occ) {
        std::vector<double> occ(static_cast<std::size_t>(n_occ), 2.0);
        return density_from_occ(C, occ);
    };

    // Solve μ from the full (k,band) spectrum; return per-k band occupations.
    auto fill_metallic = [&](const std::vector<std::vector<double>>& evals_k,
                             double& mu_ha, double& n_occ_sum,
                             double& entropy_sum) {
        std::vector<AccReal> all_evals;
        std::vector<AccReal> all_weights;
        for (int ik = 0; ik < nk; ++ik) {
            const double wk = weights[static_cast<std::size_t>(ik)];
            const auto& ev = evals_k[static_cast<std::size_t>(ik)];
            for (double e : ev) {
                all_evals.push_back(static_cast<AccReal>(e));
                all_weights.push_back(static_cast<AccReal>(wk));
            }
        }
        mu_ha = static_cast<double>(pw::internal::find_fermi_level(
            all_evals, all_weights, static_cast<AccReal>(n_elec),
            static_cast<AccReal>(degauss_ha), smear_kind, mp_order));
        n_occ_sum = 0.0;
        entropy_sum = 0.0;
        std::vector<std::vector<double>> occ_k(static_cast<std::size_t>(nk));
        for (int ik = 0; ik < nk; ++ik) {
            const double wk = weights[static_cast<std::size_t>(ik)];
            const auto& ev = evals_k[static_cast<std::size_t>(ik)];
            occ_k[static_cast<std::size_t>(ik)].resize(ev.size());
            std::vector<AccReal> ev_a(ev.begin(), ev.end());
            std::vector<AccReal> w_one(ev.size(), static_cast<AccReal>(wk));
            auto fill = pw::internal::fill_occupations(
                ev_a, w_one, static_cast<AccReal>(mu_ha),
                static_cast<AccReal>(degauss_ha), smear_kind, mp_order);
            n_occ_sum += static_cast<double>(fill.n_occ);
            entropy_sum += static_cast<double>(fill.entropy);
            for (std::size_t b = 0; b < ev.size(); ++b)
                occ_k[static_cast<std::size_t>(ik)][b] = static_cast<double>(
                    pw::internal::smear_f_band(static_cast<AccReal>(ev[b]),
                                               static_cast<AccReal>(mu_ha),
                                               static_cast<AccReal>(degauss_ha),
                                               smear_kind, mp_order));
        }
        return occ_k;
    };

    std::vector<CplxMatrix> P_k(static_cast<std::size_t>(nk));
    // Occupied Bloch coefficients (OT path); DIIS rebuilds density from full C.
    std::vector<CplxMatrix> Cocc_k(static_cast<std::size_t>(nk));
    double last_mu = 0.0;
    double last_n_elec = 0.0;
    double last_entropy = 0.0;
    // Cold start: fixed integer fill (smear path re-fills after first diag).
    // XL / SGCP multi-k warm start: replace P_k with the propagated /
    // predicted density when guess_P_k is provided. OT + guess: purify each
    // predicted P_k to natural orbitals so the per-k OT manifold starts on
    // the predicted density (complex mirror of the Γ warm start; not a
    // silent ignore of the guess, and not the cold H0 manifold — a
    // budget-capped SGCP corrector from H0 would give garbage forces). The
    // smear path re-expands Cocc_k from H0 below, but the evaluator exposes
    // no smearing, so the SGCP corrector never reaches it.
    for (int ik = layout.rank; ik < nk; ik += layout.nprocs) {
        const CplxMatrix H0k = T_k[static_cast<std::size_t>(ik)]
                               + Vnl_k[static_cast<std::size_t>(ik)];
        auto [eps0, C0k] =
            internal::diagonalize_k(H0k, orth[static_cast<std::size_t>(ik)]);
        (void)eps0;
        CplxMatrix Cocc = xt::zeros<Cplx>({nb, cfg.n_occ});
        for (int mu = 0; mu < nb; ++mu)
            for (int o = 0; o < cfg.n_occ; ++o)
                Cocc(mu, o) = C0k(mu, o);
        Cocc = linalg::lowdin_orthonormalize_k(
            Cocc, S_k[static_cast<std::size_t>(ik)]);
        Cocc_k[static_cast<std::size_t>(ik)] = Cocc;
        if (guess_P_k != nullptr) {
            P_k[static_cast<std::size_t>(ik)] =
                CplxMatrix((*guess_P_k)[static_cast<std::size_t>(ik)]);
            if (cfg.use_ot) {
                // Natural orbitals of the guess density: eigenvalues of
                // X† P X ascend, so the largest occupations sit in the LAST
                // columns (same convention as the Γ diagonalize_fock path).
                auto [n_occ_vals, Cnat] = internal::diagonalize_k(
                    P_k[static_cast<std::size_t>(ik)],
                    orth[static_cast<std::size_t>(ik)]);
                (void)n_occ_vals;
                const int m = static_cast<int>(Cnat.shape(1));
                if (m < cfg.n_occ)
                    throw std::runtime_error(
                        "GPW kpoints OT: natural-orbital warm start needs at "
                        "least n_occ kept orthogonal directions (m="
                        + std::to_string(m)
                        + ", n_occ=" + std::to_string(cfg.n_occ) + ")");
                CplxMatrix Cw = xt::zeros<Cplx>({nb, cfg.n_occ});
                for (int i = 0; i < cfg.n_occ; ++i) {
                    const int src = m - cfg.n_occ + i;
                    for (int mu = 0; mu < nb; ++mu)
                        Cw(mu, i) = Cnat(mu, src);
                }
                Cocc_k[static_cast<std::size_t>(ik)] =
                    linalg::lowdin_orthonormalize_k(
                        Cw, S_k[static_cast<std::size_t>(ik)]);
            }
        } else {
            P_k[static_cast<std::size_t>(ik)] =
                density_from_fixed(C0k, cfg.n_occ);
        }
    }

    auto ctrace = [&](const CplxMatrix& P, const CplxMatrix& M) {
        double s = 0.0;
        for (int i = 0; i < nb; ++i)
            for (int j = 0; j < nb; ++j)
                s += (P(i, j) * M(j, i)).real();
        return s;
    };

    GpwResult res;
    res.n_kpts = nk;
    res.n_kpts_local = n_kpts_local;
    // Force path reuses geometry-fixed χ^k (single-grid only; MG has no kbg).
    if (!use_mg)
        res.scf_kbg = std::make_shared<internal::KpointBasisGrid>(ws.kbg);
    const bool use_diis = cfg.diis_depth > 0;
    internal::KDiis diis(use_diis ? cfg.diis_depth : 1, weights);
    constexpr double kDiisErrTolK = 1e-5; // same gate as the Γ DIIS path
    double e_prev = 0.0;
    const bool periodic_cell = !cfg.nonperiodic;

    // k-summed real density on the fine grid from {P_k}: multigrid products or
    // the pre-summed χ^k path. Every rank returns the GLOBAL (reduced) density.
    auto collocate_rho_k = [&](const std::vector<CplxMatrix>& Pk) {
        if (use_mg)
            return mg_density(Pk);
        std::vector<double> rho(static_cast<std::size_t>(grid.npts()), 0.0);
        for (int ik = layout.rank; ik < nk; ik += layout.nprocs)
            internal::collocate_density_k(kbg, ik,
                                          Pk[static_cast<std::size_t>(ik)],
                                          weights[static_cast<std::size_t>(ik)],
                                          rho);
#ifdef ATV_HAS_MPI
        if (layout.active) {
            MPI_Allreduce(MPI_IN_PLACE, rho.data(),
                          static_cast<int>(rho.size()), MPI_DOUBLE, MPI_SUM,
                          MPI_COMM_WORLD);
        }
#endif
        return rho;
    };

    // Fold nonlocal vdW (DRSLL/LMKLL/rVV10), free-atom e_d3, or
    // Hirshfeld-scaled TS/MBD from the final k-summed density (post-SCF).
    // Mirrors Γ `fold_dispersion` (incl. multigrid product density via
    // `collocate_rho_k`).
    auto fold_dispersion_k = [&](GpwResult& out,
                                 const std::vector<CplxMatrix>& Pk) {
        // Final nonlocal E_c^nl on the (self-consistent) k-summed total
        // density. V_nl was already applied every SCF iteration inside
        // build_k_energy; this mirrors the Γ runner's post-SCF evaluation.
        if (cfg.vdw_df || cfg.vdw_df2 || cfg.rvv10) {
            const std::vector<double> rho = collocate_rho_k(Pk);
            out.e_disp = nonlocal_vdw_on_grid(grid, rho, cfg.vdw_df,
                                              cfg.vdw_df2, cfg.rvv10,
                                              /*potential=*/false)
                             .energy;
            out.e_total += out.e_disp;
            return;
        }
        if (cfg.hirshfeld && (cfg.ts || cfg.mbd)) {
            const std::vector<double> rho = collocate_rho_k(Pk);
            std::vector<int> Z(atoms.size());
            std::vector<std::array<double, 3>> pos(Z.size());
            for (std::size_t i = 0; i < Z.size(); ++i) {
                Z[i] = atoms[i].z;
                pos[i] = {atoms[i].pos[0], atoms[i].pos[1], atoms[i].pos[2]};
            }
            out.volume_ratios = hirshfeld_volume_ratios(
                grid, rho, Z, pos, cfg.cell, periodic_cell);
            if (cfg.mbd) {
                const bool mbd_per = cfg.mbd_k_mesh[0] > 0
                                     || cfg.mbd_k_mesh[1] > 0
                                     || cfg.mbd_k_mesh[2] > 0;
                if (mbd_per && cfg.nonperiodic)
                    throw std::runtime_error(
                        "GPW MBD: mbd_k_mesh requires a periodic cell "
                        "(!nonperiodic)");
                out.e_disp = mbd_rsscs_pbe(Z, pos, cfg.cell, mbd_per, 0.83, 6.0,
                                           out.volume_ratios,
                                           mbd_per ? cfg.mbd_k_mesh : nullptr)
                                 .energy;
            } else {
                constexpr double kAng2Bohr = 1.0 / 0.52917720859;
                constexpr double kRcutBohr = 50.0 * kAng2Bohr;
                out.e_disp = ts_pbe(Z, pos, cfg.cell, periodic_cell, kRcutBohr,
                                    0.94, 20.0, out.volume_ratios)
                                 .energy;
            }
            out.e_total += out.e_disp;
            return;
        }
        out.e_disp = e_d3;
        out.e_total += e_d3;
    };

    // Dudarev multi-k setup (empty plus_u → no-op).
    std::vector<int> plus_u_z;
    std::vector<internal::PlusUHa> plus_u_slots;
    pack_plus_u(cfg.plus_u, plus_u_z, plus_u_slots);
    const bool do_dftu = !plus_u_slots.empty();
    const std::vector<internal::AoShellTag> ao_tags = global_ao_tags(atoms);
    std::vector<int> z_of_atom(atoms.size());
    for (std::size_t i = 0; i < atoms.size(); ++i)
        z_of_atom[i] = atoms[i].z;
    // Closed-shell: n_spin = n/2 → E_U = 2 E_U(n_spin); V added once to H.
    linalg::RealMatrix V_dftu; // last-iteration potential (for rebuild_all_H)
    bool have_V_dftu = false;

    // Rebuild H_k for *all* k from Veff so every rank can emit the full band
    // payload after a distributed SCF (local H_k alone is incomplete). Under
    // multigrid the k-independent V_t stack is integrated once and phase-
    // summed per k.
    auto rebuild_all_H = [&](const std::vector<double>& veff) {
        std::vector<CplxMatrix> full(static_cast<std::size_t>(nk));
        std::vector<std::vector<double>> Vt;
        if (use_mg)
            Vt = mg_vt_stack(veff);
        for (int ik = 0; ik < nk; ++ik) {
            const CplxMatrix Vg =
                use_mg ? mg_phase_V(Vt, ik)
                       : internal::integrate_potential_k(kbg, ik, veff);
            full[static_cast<std::size_t>(ik)] =
                T_k[static_cast<std::size_t>(ik)]
                + Vnl_k[static_cast<std::size_t>(ik)] + Vg;
            if (do_dftu && have_V_dftu)
                internal::add_dftu_potential_cplx(
                    full[static_cast<std::size_t>(ik)], V_dftu,
                    S_k[static_cast<std::size_t>(ik)]);
        }
        return full;
    };

    auto store_k_payload = [&](const std::vector<CplxMatrix>& H_full) {
        res.k_frac.clear();
        res.k_weights.clear();
        res.eps_k.clear();
        res.eps_k_beta.clear();
        res.C_k.clear();
        res.C_k.reserve(static_cast<std::size_t>(nk));
        res.C_k_beta.clear();
        res.C_occ_k.clear();
        res.C_occ_k.resize(static_cast<std::size_t>(nk));
        res.C_occ_k_beta.clear();
        res.f_k.clear();
        res.f_k_beta.clear();
        res.P_k.clear();
        res.P_k.resize(static_cast<std::size_t>(nk));
        // First pass: diagonalize every k (all ranks rebuild — no MPI).
        std::vector<std::vector<double>> evals_store(
            static_cast<std::size_t>(nk));
        std::vector<CplxMatrix> C_store(static_cast<std::size_t>(nk));
        for (int ik = 0; ik < nk; ++ik) {
            const auto& kp = kpts[static_cast<std::size_t>(ik)];
            res.k_frac.push_back({kp.frac[0], kp.frac[1], kp.frac[2]});
            res.k_weights.push_back(kp.weight);
            auto [eps, C] =
                internal::diagonalize_k(H_full[static_cast<std::size_t>(ik)],
                                        orth[static_cast<std::size_t>(ik)]);
            evals_store[static_cast<std::size_t>(ik)] = eps;
            res.eps_k.push_back(std::move(eps));
            C_store[static_cast<std::size_t>(ik)] = C;
            // Full Bloch C_k for multi-k mp2() and free-energy force/stress.
            res.C_k.push_back(linalg::CplxMatrix(C));
        }
        // Metallic smear: shared μ over the final spectrum; P_k and f_k must
        // match the free-energy fixed point (not integer n_occ truncation).
        std::vector<std::vector<double>> occ_store;
        if (use_smear) {
            double mu_tmp = last_mu, ne_tmp = last_n_elec, s_tmp = last_entropy;
            occ_store = fill_metallic(evals_store, mu_tmp, ne_tmp, s_tmp);
            last_mu = mu_tmp;
            last_n_elec = ne_tmp;
            last_entropy = s_tmp;
            res.f_k = occ_store;
        }
        for (int ik = 0; ik < nk; ++ik) {
            const CplxMatrix& C = C_store[static_cast<std::size_t>(ik)];
            // Integer C_occ_k (n_occ columns) kept for fixed-occ force/stress
            // and OT consumers; free-energy forces use C_k + f_k.
            std::vector<Cplx> occ(static_cast<std::size_t>(nb)
                                  * static_cast<std::size_t>(cfg.n_occ));
            for (int mu = 0; mu < nb; ++mu)
                for (int o = 0; o < cfg.n_occ; ++o)
                    occ[static_cast<std::size_t>(mu) * cfg.n_occ
                        + static_cast<std::size_t>(o)] = C(mu, o);
            res.C_occ_k[static_cast<std::size_t>(ik)] = std::move(occ);
            if (use_smear) {
                res.P_k[static_cast<std::size_t>(ik)] = density_from_occ(
                    C, occ_store[static_cast<std::size_t>(ik)]);
            } else {
                // XL-BOMD payload: P_k from final MOs (occ = 2) so the
                // auxiliary tracks the SCF fixed point, not mixed DIIS dens.
                res.P_k[static_cast<std::size_t>(ik)] =
                    density_from_fixed(C, cfg.n_occ);
            }
        }
    };

    std::vector<CplxMatrix> H_k(static_cast<std::size_t>(nk));
    std::vector<double>& last_veff = ws.last_veff;

    // Shared H/energy rebuild from {P_k} (used by DIIS and OT).
    // ATV_GPW_PROF: per-call [gpw prof k] timings (collocate / Poisson / XC /
    // integrate) so multi-k twins match the Γ [gpw prof] shape.
    const bool kprof = std::getenv("ATV_GPW_PROF") != nullptr;
    using KProfClk = std::chrono::steady_clock;
    auto k_pms = [](KProfClk::time_point a, KProfClk::time_point b) {
        return std::chrono::duration<double, std::milli>(b - a).count();
    };
    int k_energy_builds = 0;
    auto build_k_energy = [&](const std::vector<CplxMatrix>& Pk) {
        ++k_energy_builds;
        auto t0p = KProfClk::now();
        const std::vector<double> rho = collocate_rho_k(Pk);
        auto t1p = KProfClk::now();
        // Shared ρ→Poisson→XC→V_eff (k path is always periodic Poisson).
        const internal::KsGridRks ks = internal::ks_grid_build_rks(
            grid, rho, n_ion, v_locsr, cfg.xc, /*nonperiodic=*/false,
            cfg.vdw_df, cfg.vdw_df2, cfg.rvv10);
        auto t3p = KProfClk::now();
        auto t2p = t1p; // poisson+xc folded into ks_grid_build (prof bucket)
        (void)t2p;
        const double e_locgrid = ks.e_locgrid;
        last_veff = ks.veff;
        res.veff = last_veff; // force assembly reuses SCF V_eff
        res.v_es = ks.v_es;
        const std::vector<double>& veff = last_veff;

        std::vector<std::vector<double>> Vt;
        if (use_mg)
            Vt = mg_vt_stack(veff);
        double e_kin = 0.0, e_nl = 0.0;
        for (int ik = layout.rank; ik < nk; ik += layout.nprocs) {
            const double wk = weights[static_cast<std::size_t>(ik)];
            const CplxMatrix Vg =
                use_mg ? mg_phase_V(Vt, ik)
                       : internal::integrate_potential_k(kbg, ik, veff);
            H_k[static_cast<std::size_t>(ik)] =
                T_k[static_cast<std::size_t>(ik)]
                + Vnl_k[static_cast<std::size_t>(ik)] + Vg;
            e_kin += wk
                     * ctrace(Pk[static_cast<std::size_t>(ik)],
                              T_k[static_cast<std::size_t>(ik)]);
            e_nl += wk
                    * ctrace(Pk[static_cast<std::size_t>(ik)],
                             Vnl_k[static_cast<std::size_t>(ik)]);
        }
        auto t4p = KProfClk::now();
        if (kprof)
            std::fprintf(stderr,
                         "[gpw prof k] collocate=%.0f poisson=%.0f xc=%.0f "
                         "integrate=%.0f ms "
                         "(ngrid=%ld nk=%d mg=%d build=%d)\n",
                         k_pms(t0p, t1p), k_pms(t1p, t2p), k_pms(t2p, t3p),
                         k_pms(t3p, t4p), grid.npts(), nk, use_mg ? 1 : 0,
                         k_energy_builds);
        double e_dftu = 0.0;
        if (do_dftu) {
            std::vector<CplxMatrix> Pspin_k(static_cast<std::size_t>(nk));
            for (int ik = layout.rank; ik < nk; ik += layout.nprocs)
                Pspin_k[static_cast<std::size_t>(ik)] =
                    0.5 * Pk[static_cast<std::size_t>(ik)];
            e_dftu = 2.0
                     * internal::apply_dft_plus_u_k(
                         H_k, Pspin_k, S_k, weights, layout.rank, layout.nprocs,
                         layout.active, ao_tags, z_of_atom, plus_u_slots,
                         plus_u_z, &V_dftu);
            have_V_dftu = true;
        }
#ifdef ATV_HAS_MPI
        if (layout.active) {
            MPI_Allreduce(MPI_IN_PLACE, &e_kin, 1, MPI_DOUBLE, MPI_SUM,
                          MPI_COMM_WORLD);
            MPI_Allreduce(MPI_IN_PLACE, &e_nl, 1, MPI_DOUBLE, MPI_SUM,
                          MPI_COMM_WORLD);
        }
#endif
        res.e_dftu = e_dftu;
        res.e_kinetic = e_kin;
        res.e_nonlocal = 0.0;
        res.e_loc_sr = e_nl + e_locgrid;
        res.e_es = ks.e_hartree - e_self;
        res.e_xc = ks.e_xc;
        const double e_ks =
            res.e_kinetic + res.e_loc_sr + res.e_es + res.e_xc + e_dftu;
        const double e_ent = use_smear ? (-degauss_ha * last_entropy) : 0.0;
        res.e_entropy = e_ent;
        res.e_fermi = use_smear ? last_mu : 0.0;
        res.n_elec_occ = use_smear ? last_n_elec : n_elec;
        return e_ks + e_ent;
    };

    // ── Multi-k fixed-density KS eval (Car–Parrinello electronic step) ───
    // max_iter == 0: one collocate→Poisson→XC→integrate at the supplied
    // {P_k}; no diagonalization, no mixing, no orbital update. Populates
    // H_ks_k / S_k for the complex electronic EOM (multi-k CPMD).
    if (cfg.max_iter == 0) {
        if (guess_P_k == nullptr)
            throw std::invalid_argument(
                "GpwDft: multi-k max_iter==0 (fixed-density) requires "
                "guess_P_k (the Car–Parrinello density per k)");
        if (cfg.use_ot)
            throw std::invalid_argument(
                "GpwDft: multi-k fixed-density eval is incompatible with "
                "use_ot");
        // Ensure every rank holds the full {P_k} (init loop only fills local k).
        for (int ik = 0; ik < nk; ++ik)
            P_k[static_cast<std::size_t>(ik)] =
                CplxMatrix((*guess_P_k)[static_cast<std::size_t>(ik)]);
        const double e_total = build_k_energy(P_k);
        res.e_total = e_total;
        // Full H_k on every rank from the shared V_eff.
        const std::vector<CplxMatrix> H_full = rebuild_all_H(last_veff);
        res.H_ks_k.resize(static_cast<std::size_t>(nk));
        res.S_k.resize(static_cast<std::size_t>(nk));
        res.P_k.resize(static_cast<std::size_t>(nk));
        res.k_frac.clear();
        res.k_weights.clear();
        res.k_frac.reserve(static_cast<std::size_t>(nk));
        res.k_weights.reserve(static_cast<std::size_t>(nk));
        for (int ik = 0; ik < nk; ++ik) {
            const auto& kp = kpts[static_cast<std::size_t>(ik)];
            res.k_frac.push_back({kp.frac[0], kp.frac[1], kp.frac[2]});
            res.k_weights.push_back(kp.weight);
            res.H_ks_k[static_cast<std::size_t>(ik)] =
                H_full[static_cast<std::size_t>(ik)];
            res.S_k[static_cast<std::size_t>(ik)] =
                S_k[static_cast<std::size_t>(ik)];
            res.P_k[static_cast<std::size_t>(ik)] =
                P_k[static_cast<std::size_t>(ik)];
        }
        res.converged = true;
        res.iters = 0;
        fold_dispersion_k(res, P_k);
        return res;
    }

    // ── Multi-k OT (closed-shell RKS; integer / metallic / +DFT+U) ─────────
    // Diagonalization-free: active C_k live on the complex OT manifold.
    // Integer occupations + PLUS_U: CP2K-style OUTER_SCF — rebuild H (incl.
    // V_U), then fixed-H OT band minimize on each local k (avoids stalled
    // Armijo when the SCF total-energy gradient is not the frozen-H band
    // gradient). Metallic smear (FD/MP), with or without PLUS_U: joint PR+
    // Armijo on the free energy; build_k_energy already folds V_U / E_U when
    // plus_u is set (product: OT+U+smear ≡ DIIS+U+smear fixed point).
    if (cfg.use_ot) {
        auto dens_from_cocc_int = [&](const CplxMatrix& C) {
            return density_from_fixed(C, cfg.n_occ);
        };

        // ── OUTER_SCF + fixed-H OT for multi-k OT+DFT+U (integer occ) ────
        if (do_dftu && !use_smear) {
            constexpr double kInnerTol = 1e-7;
            constexpr int kInnerMax = 80;
            for (int outer = 0; outer < cfg.max_iter; ++outer) {
                for (int ik = layout.rank; ik < nk; ik += layout.nprocs)
                    P_k[static_cast<std::size_t>(ik)] = dens_from_cocc_int(
                        Cocc_k[static_cast<std::size_t>(ik)]);
                const double e_total = build_k_energy(P_k);
                res.iters = outer + 1;
                res.e_total = e_total;

                if (std::getenv("ATV_GPW_TRACE") && layout.rank == 0)
                    std::fprintf(
                        stderr,
                        "[gpw k OT+U outer %2d] E=%.8f  U=%.4f  dE=%.2e "
                        "(nk=%d)\n",
                        outer, e_total, res.e_dftu, std::abs(e_total - e_prev),
                        nk);

                if (outer > 0 && std::abs(e_total - e_prev) < cfg.tol) {
                    res.converged = true;
                    store_k_payload(rebuild_all_H(last_veff));
                    fold_dispersion_k(res, P_k);
                    return res;
                }
                e_prev = e_total;

                // Inner: fixed-H OT → occupied eigenspace of current H_k.
                for (int ik = layout.rank; ik < nk; ik += layout.nprocs) {
                    const auto ot = internal::ot_minimize_fixed_k(
                        H_k[static_cast<std::size_t>(ik)],
                        S_k[static_cast<std::size_t>(ik)],
                        T_k[static_cast<std::size_t>(ik)],
                        Cocc_k[static_cast<std::size_t>(ik)], kInnerTol,
                        kInnerMax);
                    Cocc_k[static_cast<std::size_t>(ik)] = ot.C;
                    P_k[static_cast<std::size_t>(ik)] =
                        dens_from_cocc_int(ot.C);
                }
            }
            if (!last_veff.empty())
                store_k_payload(rebuild_all_H(last_veff));
            fold_dispersion_k(res, P_k);
            res.n_elec_occ = n_elec;
            return res;
        }

        // ── One PR+ Armijo step per outer H rebuild (smear and/or no U) ──
        // Active OT manifold: integer path keeps n_occ double-occupied columns;
        // metallic smear uses the full AO spectrum (same as DIIS + smear) so
        // the free-energy fixed point matches; extra_bands remains API parity.
        // PLUS_U + smear rides this path (V_U via build_k_energy).
        const int n_active = use_smear ? m_min : cfg.n_occ;
        if (n_active < 1)
            throw std::runtime_error(
                "GPW kpoints OT: n_active < 1 (n_occ / basis)");
        // Expand cold-start Cocc to the active manifold when smearing.
        if (use_smear) {
            for (int ik = layout.rank; ik < nk; ik += layout.nprocs) {
                const CplxMatrix H0k = T_k[static_cast<std::size_t>(ik)]
                                       + Vnl_k[static_cast<std::size_t>(ik)];
                auto [eps0, C0k] = internal::diagonalize_k(
                    H0k, orth[static_cast<std::size_t>(ik)]);
                (void)eps0;
                const int mcols =
                    std::min(n_active, static_cast<int>(C0k.shape(1)));
                CplxMatrix Cocc = xt::zeros<Cplx>({nb, n_active});
                for (int mu = 0; mu < nb; ++mu)
                    for (int o = 0; o < mcols; ++o)
                        Cocc(mu, o) = C0k(mu, o);
                Cocc = linalg::lowdin_orthonormalize_k(
                    Cocc, S_k[static_cast<std::size_t>(ik)]);
                Cocc_k[static_cast<std::size_t>(ik)] = Cocc;
                // Cold density: integer fill of n_occ (smear re-fills after H).
                P_k[static_cast<std::size_t>(ik)] =
                    density_from_fixed(C0k, cfg.n_occ);
            }
        }
        constexpr double kOtGradTolK = 1e-5;
        const double ot_grad_tol_k = kOtGradTolK;
        auto frob_k = [](const CplxMatrix& a, const CplxMatrix& b) {
            double s = 0.0;
            const int n0 = static_cast<int>(a.shape(0));
            const int n1 = static_cast<int>(a.shape(1));
            for (int i = 0; i < n0; ++i)
                for (int j = 0; j < n1; ++j)
                    s += (std::conj(a(i, j)) * b(i, j)).real();
            return s;
        };
        // Per-k band occupations (f ∈ [0,2] closed-shell); integer path all 2.
        std::vector<std::vector<double>> occ_k(
            static_cast<std::size_t>(nk),
            std::vector<double>(static_cast<std::size_t>(n_active), 2.0));
        if (!use_smear) {
            for (auto& o : occ_k)
                o.assign(static_cast<std::size_t>(n_active), 2.0);
        }
        auto dens_from_cocc = [&](const CplxMatrix& C, int ik) {
            if (use_smear)
                return density_from_occ(C, occ_k[static_cast<std::size_t>(ik)]);
            return density_from_fixed(C, cfg.n_occ);
        };
        // Subspace-canonicalize C† H C → (ε, U), C ← C U; fill metallic f.
        auto refresh_occupations = [&]() {
            std::vector<std::vector<double>> evals_k(
                static_cast<std::size_t>(nk));
            for (int ik = layout.rank; ik < nk; ik += layout.nprocs) {
                auto& C = Cocc_k[static_cast<std::size_t>(ik)];
                const auto& H = H_k[static_cast<std::size_t>(ik)];
                const auto& S = S_k[static_cast<std::size_t>(ik)];
                // h = C† H C (n_active × n_active Hermitian).
                CplxMatrix h = xt::zeros<Cplx>({n_active, n_active});
                for (int i = 0; i < n_active; ++i)
                    for (int j = 0; j < n_active; ++j) {
                        Cplx s(0, 0);
                        for (int mu = 0; mu < nb; ++mu) {
                            Cplx hcv(0, 0);
                            for (int nu = 0; nu < nb; ++nu)
                                hcv += H(mu, nu) * C(nu, j);
                            s += std::conj(C(mu, i)) * hcv;
                        }
                        h(i, j) = s;
                    }
                std::vector<double> eps;
                CplxMatrix U;
                linalg::eigh_hermitian(h, eps, U);
                // C ← C U
                CplxMatrix Cnew = xt::zeros<Cplx>({nb, n_active});
                for (int mu = 0; mu < nb; ++mu)
                    for (int o = 0; o < n_active; ++o) {
                        Cplx s(0, 0);
                        for (int p = 0; p < n_active; ++p)
                            s += C(mu, p) * U(p, o);
                        Cnew(mu, o) = s;
                    }
                C = linalg::lowdin_orthonormalize_k(Cnew, S);
                evals_k[static_cast<std::size_t>(ik)] = std::move(eps);
            }
#ifdef ATV_HAS_MPI
            if (layout.active && use_smear) {
                std::vector<double> pack(
                    static_cast<std::size_t>(nk * n_active), 0.0);
                std::vector<int> nband(static_cast<std::size_t>(nk), 0);
                for (int ik = layout.rank; ik < nk; ik += layout.nprocs) {
                    const auto& ev = evals_k[static_cast<std::size_t>(ik)];
                    nband[static_cast<std::size_t>(ik)] =
                        static_cast<int>(ev.size());
                    for (std::size_t b = 0; b < ev.size(); ++b)
                        pack[static_cast<std::size_t>(ik * n_active) + b] =
                            ev[b];
                }
                MPI_Allreduce(MPI_IN_PLACE, pack.data(), nk * n_active,
                              MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
                MPI_Allreduce(MPI_IN_PLACE, nband.data(), nk, MPI_INT, MPI_SUM,
                              MPI_COMM_WORLD);
                for (int ik = 0; ik < nk; ++ik) {
                    const int m = nband[static_cast<std::size_t>(ik)];
                    evals_k[static_cast<std::size_t>(ik)].resize(
                        static_cast<std::size_t>(m));
                    for (int b = 0; b < m; ++b)
                        evals_k[static_cast<std::size_t>(
                            ik)][static_cast<std::size_t>(b)] =
                            pack[static_cast<std::size_t>(ik * n_active + b)];
                }
            }
#endif
            if (use_smear) {
                auto filled =
                    fill_metallic(evals_k, last_mu, last_n_elec, last_entropy);
                for (int ik = 0; ik < nk; ++ik) {
                    occ_k[static_cast<std::size_t>(ik)] =
                        std::move(filled[static_cast<std::size_t>(ik)]);
                    // Pad / trim to n_active.
                    occ_k[static_cast<std::size_t>(ik)].resize(
                        static_cast<std::size_t>(n_active), 0.0);
                }
            }
        };

        std::vector<CplxMatrix> d_k(static_cast<std::size_t>(nk));
        std::vector<CplxMatrix> pz_prev_k(static_cast<std::size_t>(nk));
        double denom_prev = 1.0;
        bool have_prev = false;
        // After an accepted Armijo trial, H_k already matches dens(C,f). Skip
        // the pre-refresh ρ→H on the next outer when true (metallic only).
        bool h_matches_manifold = false;

        for (int iter = 0; iter < cfg.max_iter; ++iter) {
            // One ρ→H when occupations are frozen (integer). Metallic needs
            // H → fill f → rebuild P → H so F and g see a consistent triple;
            // the first of those two builds is skipped when H already matches
            // dens(C,f_old) from the previous Armijo accept.
            for (int ik = layout.rank; ik < nk; ik += layout.nprocs)
                P_k[static_cast<std::size_t>(ik)] =
                    dens_from_cocc(Cocc_k[static_cast<std::size_t>(ik)], ik);
            double e_total = 0.0;
            if (use_smear) {
                if (!h_matches_manifold)
                    (void)build_k_energy(P_k);
                refresh_occupations();
                for (int ik = layout.rank; ik < nk; ik += layout.nprocs)
                    P_k[static_cast<std::size_t>(ik)] =
                        dens_from_cocc(Cocc_k[static_cast<std::size_t>(ik)],
                                       ik);
                e_total = build_k_energy(P_k);
            } else if (!h_matches_manifold) {
                // Integer: one KS rebuild unless the previous Armijo accept
                // already left H_k matching dens(C) (Γ RKS OT economics).
                e_total = build_k_energy(P_k);
                refresh_occupations();
            } else {
                e_total = res.e_total; // carried from accepted trial
                refresh_occupations();
            }
            h_matches_manifold = true;
            res.iters = iter + 1;
            res.e_total = e_total;

            // Projected OT gradient (local k); max-norm + weighted PR denom.
            double gnorm = 0.0;
            double slope_dot = 0.0;
            std::vector<CplxMatrix> g_k(static_cast<std::size_t>(nk));
            std::vector<CplxMatrix> pz_k(static_cast<std::size_t>(nk));
            for (int ik = layout.rank; ik < nk; ik += layout.nprocs) {
                const auto& S = S_k[static_cast<std::size_t>(ik)];
                const auto& H = H_k[static_cast<std::size_t>(ik)];
                const auto& C = Cocc_k[static_cast<std::size_t>(ik)];
                const CplxMatrix g =
                    use_smear
                        ? linalg::band_gradient_k(
                              H, S, C, occ_k[static_cast<std::size_t>(ik)])
                        : linalg::band_gradient_k(H, S, C, 2.0);
                for (const Cplx v : g)
                    gnorm = std::max(gnorm, std::abs(v));
                const internal::OtPrecondSingleInverseK pre(H, S, C);
                const CplxMatrix pz = linalg::s_tangent_k(C, S, pre.apply(g));
                g_k[static_cast<std::size_t>(ik)] = g;
                pz_k[static_cast<std::size_t>(ik)] = pz;
            }
#ifdef ATV_HAS_MPI
            if (layout.active) {
                double gglob = gnorm;
                MPI_Allreduce(&gnorm, &gglob, 1, MPI_DOUBLE, MPI_MAX,
                              MPI_COMM_WORLD);
                gnorm = gglob;
            }
#endif
            if (std::getenv("ATV_GPW_TRACE") && layout.rank == 0) {
                if (use_smear)
                    std::fprintf(
                        stderr,
                        "[gpw k OT %2d] F=%.8f  T=%.4f loc=%.4f es=%.4f "
                        "xc=%.4f  dE=%.2e gnorm=%.2e mu=%.6f Ne=%.6f "
                        "(nk=%d n_act=%d)\n",
                        iter, e_total, res.e_kinetic, res.e_loc_sr, res.e_es,
                        res.e_xc, std::abs(e_total - e_prev), gnorm, last_mu,
                        last_n_elec, nk, n_active);
                else
                    std::fprintf(
                        stderr,
                        "[gpw k OT %2d] E=%.8f  T=%.4f loc=%.4f es=%.4f "
                        "xc=%.4f  dE=%.2e gnorm=%.2e (nk=%d)\n",
                        iter, e_total, res.e_kinetic, res.e_loc_sr, res.e_es,
                        res.e_xc, std::abs(e_total - e_prev), gnorm, nk);
            }

            // Metallic: free-energy plateau + moderate gnorm is enough (FD
            // occupations make the projected gradient softer than gapped OT).
            // PLUS_U softens g further — accept energy plateau with soft gtol.
            const double gtol = (use_smear || do_dftu)
                                    ? std::max(kOtGradTolK, 5e-4)
                                    : kOtGradTolK;
            if (iter > 0 && gnorm < gtol
                && std::abs(e_total - e_prev) < cfg.tol) {
                res.converged = true;
                store_k_payload(rebuild_all_H(last_veff));
                fold_dispersion_k(res, P_k);
                res.e_fermi = use_smear ? last_mu : 0.0;
                res.e_entropy = use_smear ? (-degauss_ha * last_entropy) : 0.0;
                res.n_elec_occ = use_smear ? last_n_elec : n_elec;
                return res;
            }
            // Energy plateau fallback for metals / OT+U+smear (budget-friendly).
            if ((use_smear || do_dftu) && iter > 15
                && std::abs(e_total - e_prev) < cfg.tol) {
                res.converged = true;
                store_k_payload(rebuild_all_H(last_veff));
                fold_dispersion_k(res, P_k);
                res.e_fermi = use_smear ? last_mu : 0.0;
                res.e_entropy = use_smear ? (-degauss_ha * last_entropy) : 0.0;
                res.n_elec_occ = use_smear ? last_n_elec : n_elec;
                return res;
            }
            e_prev = e_total;

            // Polak–Ribière+ search direction (joint over local k).
            double num = 0.0, den = 0.0;
            for (int ik = layout.rank; ik < nk; ik += layout.nprocs) {
                const double wk = weights[static_cast<std::size_t>(ik)];
                const auto& g = g_k[static_cast<std::size_t>(ik)];
                const auto& pz = pz_k[static_cast<std::size_t>(ik)];
                den += wk * frob_k(g, pz);
                if (have_prev)
                    num += wk
                           * frob_k(
                               g, pz - pz_prev_k[static_cast<std::size_t>(ik)]);
            }
#ifdef ATV_HAS_MPI
            if (layout.active) {
                double pack[2] = {num, den};
                MPI_Allreduce(MPI_IN_PLACE, pack, 2, MPI_DOUBLE, MPI_SUM,
                              MPI_COMM_WORLD);
                num = pack[0];
                den = pack[1];
            }
#endif
            const double beta = (!have_prev || !(denom_prev > 0.0))
                                    ? 0.0
                                    : std::max(0.0, num / denom_prev);
            denom_prev = (den > 1e-300) ? den : 1.0;
            slope_dot = 0.0;
            for (int ik = layout.rank; ik < nk; ik += layout.nprocs) {
                const auto& S = S_k[static_cast<std::size_t>(ik)];
                const auto& C = Cocc_k[static_cast<std::size_t>(ik)];
                const auto& g = g_k[static_cast<std::size_t>(ik)];
                const auto& pz = pz_k[static_cast<std::size_t>(ik)];
                CplxMatrix d;
                if (!have_prev)
                    d = -pz;
                else
                    d = linalg::s_tangent_k(
                        C, S, -pz + beta * d_k[static_cast<std::size_t>(ik)]);
                double sl = frob_k(g, d);
                if (sl >= 0.0) {
                    d = -pz;
                    sl = frob_k(g, d);
                }
                d_k[static_cast<std::size_t>(ik)] = d;
                pz_prev_k[static_cast<std::size_t>(ik)] = pz;
                slope_dot += weights[static_cast<std::size_t>(ik)] * sl;
            }
#ifdef ATV_HAS_MPI
            if (layout.active)
                MPI_Allreduce(MPI_IN_PLACE, &slope_dot, 1, MPI_DOUBLE, MPI_SUM,
                              MPI_COMM_WORLD);
#endif
            have_prev = true;

            // Curvature from frozen-H band energy (weighted by f), Armijo on F.
            auto band_k = [&](const CplxMatrix& H, const CplxMatrix& Cc,
                              int ik) {
                const auto& f = occ_k[static_cast<std::size_t>(ik)];
                const int no = static_cast<int>(Cc.shape(1));
                double s = 0.0;
                for (int o = 0; o < no; ++o) {
                    const double fo = (o < static_cast<int>(f.size()))
                                          ? f[static_cast<std::size_t>(o)]
                                          : (use_smear ? 0.0 : 2.0);
                    if (!(fo > 1e-14))
                        continue;
                    double eorb = 0.0;
                    for (int mu = 0; mu < nb; ++mu) {
                        Cplx hcv(0, 0);
                        for (int nu = 0; nu < nb; ++nu)
                            hcv += H(mu, nu) * Cc(nu, o);
                        eorb += (std::conj(Cc(mu, o)) * hcv).real();
                    }
                    s += fo * eorb;
                }
                return s;
            };
            double Eb0 = 0.0, Ebt = 0.0;
            for (int ik = layout.rank; ik < nk; ik += layout.nprocs) {
                const double wk = weights[static_cast<std::size_t>(ik)];
                const auto& H = H_k[static_cast<std::size_t>(ik)];
                const auto& S = S_k[static_cast<std::size_t>(ik)];
                const auto& C = Cocc_k[static_cast<std::size_t>(ik)];
                const auto& d = d_k[static_cast<std::size_t>(ik)];
                Eb0 += wk * band_k(H, C, ik);
                Ebt += wk * band_k(H, internal::ot_parameterize_k(C, d, S), ik);
            }
#ifdef ATV_HAS_MPI
            if (layout.active) {
                double pack[2] = {Eb0, Ebt};
                MPI_Allreduce(MPI_IN_PLACE, pack, 2, MPI_DOUBLE, MPI_SUM,
                              MPI_COMM_WORLD);
                Eb0 = pack[0];
                Ebt = pack[1];
            }
#endif
            const double curv = 2.0 * (Ebt - Eb0 - slope_dot);
            double alpha = (curv > 1e-300) ? -slope_dot / curv : 1.0;
            if (!(alpha > 0.0))
                alpha = 1.0;
            alpha = std::min(alpha, 2.0);
            constexpr double c1 = 1e-4;

            auto trial_energy = [&](double a) {
                std::vector<CplxMatrix> Ptry(static_cast<std::size_t>(nk));
                for (int ik = layout.rank; ik < nk; ik += layout.nprocs) {
                    const auto& S = S_k[static_cast<std::size_t>(ik)];
                    const auto& C = Cocc_k[static_cast<std::size_t>(ik)];
                    const auto& d = d_k[static_cast<std::size_t>(ik)];
                    CplxMatrix Cnew = internal::ot_parameterize_k(C, a * d, S);
                    Cnew = linalg::lowdin_orthonormalize_k(Cnew, S);
                    Ptry[static_cast<std::size_t>(ik)] =
                        dens_from_cocc(Cnew, ik);
                }
                return build_k_energy(Ptry);
            };

            for (int bt = 0; bt < 50; ++bt) {
                const double et = trial_energy(alpha);
                if (et <= e_total + c1 * alpha * slope_dot || bt == 49) {
                    e_total = et;
                    res.e_total = et;
                    break;
                }
                const double denom = et - e_total - slope_dot * alpha;
                const double a_q =
                    (denom > 1e-300) ? -slope_dot * alpha * alpha / (2.0 * denom)
                                     : 0.5 * alpha;
                alpha = std::min(std::max(a_q, 0.1 * alpha), 0.5 * alpha);
            }
            // Accept step: C/P from accepted alpha; H_k already from last trial.
            for (int ik = layout.rank; ik < nk; ik += layout.nprocs) {
                const auto& S = S_k[static_cast<std::size_t>(ik)];
                auto& C = Cocc_k[static_cast<std::size_t>(ik)];
                const auto& d = d_k[static_cast<std::size_t>(ik)];
                C = linalg::lowdin_orthonormalize_k(
                    internal::ot_parameterize_k(C, alpha * d, S), S);
                P_k[static_cast<std::size_t>(ik)] = dens_from_cocc(C, ik);
            }
            h_matches_manifold = true; // next outer skips integer KS rebuild
        }
        // Budget exhausted: last_veff from final trial (or last outer build).
        if (!last_veff.empty())
            store_k_payload(rebuild_all_H(last_veff));
        fold_dispersion_k(res, P_k);
        res.e_fermi = use_smear ? last_mu : 0.0;
        res.e_entropy = use_smear ? (-degauss_ha * last_entropy) : 0.0;
        res.n_elec_occ = use_smear ? last_n_elec : n_elec;
        return res;
    }

    for (int iter = 0; iter < cfg.max_iter; ++iter) {
        const double e_total = build_k_energy(P_k);
        res.iters = iter + 1;
        res.e_total = e_total;
        const double e_dftu = res.e_dftu;

        // k-resolved commutator residual X^dag (H P S - S P H) X (local k).
        std::vector<CplxMatrix> err(static_cast<std::size_t>(nk));
        double err_norm = 0.0;
        for (int ik = layout.rank; ik < nk; ik += layout.nprocs) {
            const CplxMatrix e =
                internal::k_diis_error(H_k[static_cast<std::size_t>(ik)],
                                       P_k[static_cast<std::size_t>(ik)],
                                       S_k[static_cast<std::size_t>(ik)],
                                       orth[static_cast<std::size_t>(ik)].X);
            for (const Cplx v : e)
                err_norm = std::max(err_norm, std::abs(v));
            err[static_cast<std::size_t>(ik)] = e;
        }
#ifdef ATV_HAS_MPI
        if (layout.active) {
            double global_err = err_norm;
            MPI_Allreduce(&err_norm, &global_err, 1, MPI_DOUBLE, MPI_MAX,
                          MPI_COMM_WORLD);
            err_norm = global_err;
        }
#endif

        if (std::getenv("ATV_GPW_TRACE") && layout.rank == 0) {
            if (use_smear)
                std::fprintf(stderr,
                             "[gpw k %2d] F=%.8f  T=%.4f loc=%.4f es=%.4f "
                             "xc=%.4f U=%.4f  dE=%.2e err=%.2e mu=%.6f Ne=%.6f "
                             "(nk=%d nlocal=%d nprocs=%d)\n",
                             iter, e_total, res.e_kinetic, res.e_loc_sr,
                             res.e_es, res.e_xc, e_dftu,
                             std::abs(e_total - e_prev), err_norm, last_mu,
                             last_n_elec, nk, n_kpts_local, layout.nprocs);
            else
                std::fprintf(stderr,
                             "[gpw k %2d] E=%.8f  T=%.4f loc=%.4f es=%.4f "
                             "xc=%.4f U=%.4f  dE=%.2e err=%.2e (nk=%d "
                             "nlocal=%d nprocs=%d)\n",
                             iter, e_total, res.e_kinetic, res.e_loc_sr,
                             res.e_es, res.e_xc, e_dftu,
                             std::abs(e_total - e_prev), err_norm, nk,
                             n_kpts_local, layout.nprocs);
        }

        const double err_gate = do_dftu ? 5e-3 : kDiisErrTolK;
        if (iter > 0 && std::abs(e_total - e_prev) < cfg.tol
            && err_norm < err_gate) {
            res.converged = true;
            store_k_payload(rebuild_all_H(last_veff));
            fold_dispersion_k(res, P_k);
            return res;
        }
        e_prev = e_total;

        if (use_diis) // extrapolated KS matrices drive the next density
            H_k = diis.step(H_k, err);

        // Diagonalize owned k; for metallic smear, gather the full spectrum
        // and rebuild P_k with fractional occupations at a shared μ.
        std::vector<std::vector<double>> evals_k(static_cast<std::size_t>(nk));
        std::vector<CplxMatrix> C_k_local(static_cast<std::size_t>(nk));
        for (int ik = layout.rank; ik < nk; ik += layout.nprocs) {
            auto [eps, C] =
                internal::diagonalize_k(H_k[static_cast<std::size_t>(ik)],
                                        orth[static_cast<std::size_t>(ik)]);
            evals_k[static_cast<std::size_t>(ik)] = std::move(eps);
            C_k_local[static_cast<std::size_t>(ik)] = std::move(C);
        }
#ifdef ATV_HAS_MPI
        if (layout.active && use_smear) {
            // Pack eigenvalues: variable m per k — pad to nb (upper bound).
            std::vector<double> pack(static_cast<std::size_t>(nk * nb), 0.0);
            std::vector<int> nband(static_cast<std::size_t>(nk), 0);
            for (int ik = layout.rank; ik < nk; ik += layout.nprocs) {
                const auto& ev = evals_k[static_cast<std::size_t>(ik)];
                nband[static_cast<std::size_t>(ik)] =
                    static_cast<int>(ev.size());
                for (std::size_t b = 0; b < ev.size(); ++b)
                    pack[static_cast<std::size_t>(ik * nb) + b] = ev[b];
            }
            MPI_Allreduce(MPI_IN_PLACE, pack.data(), nk * nb, MPI_DOUBLE,
                          MPI_SUM, MPI_COMM_WORLD);
            MPI_Allreduce(MPI_IN_PLACE, nband.data(), nk, MPI_INT, MPI_SUM,
                          MPI_COMM_WORLD);
            for (int ik = 0; ik < nk; ++ik) {
                const int m = nband[static_cast<std::size_t>(ik)];
                evals_k[static_cast<std::size_t>(ik)].resize(
                    static_cast<std::size_t>(m));
                for (int b = 0; b < m; ++b)
                    evals_k[static_cast<std::size_t>(ik)]
                           [static_cast<std::size_t>(b)] =
                               pack[static_cast<std::size_t>(ik * nb + b)];
            }
        }
#endif
        auto mix_density = [&](int ik, CplxMatrix P_new) {
            if (use_diis) {
                if (do_dftu) {
                    const double a = std::min(std::max(cfg.mix, 0.15), 0.6);
                    P_k[static_cast<std::size_t>(ik)] =
                        (1.0 - a) * P_k[static_cast<std::size_t>(ik)]
                        + a * P_new;
                } else {
                    P_k[static_cast<std::size_t>(ik)] = std::move(P_new);
                }
            } else {
                P_k[static_cast<std::size_t>(ik)] =
                    (1.0 - cfg.mix) * P_k[static_cast<std::size_t>(ik)]
                    + cfg.mix * P_new;
            }
        };
        if (use_smear) {
            auto occ_k =
                fill_metallic(evals_k, last_mu, last_n_elec, last_entropy);
            for (int ik = layout.rank; ik < nk; ik += layout.nprocs) {
                mix_density(ik, density_from_occ(
                                    C_k_local[static_cast<std::size_t>(ik)],
                                    occ_k[static_cast<std::size_t>(ik)]));
            }
        } else {
            for (int ik = layout.rank; ik < nk; ik += layout.nprocs) {
                mix_density(ik, density_from_fixed(
                                    C_k_local[static_cast<std::size_t>(ik)],
                                    cfg.n_occ));
            }
        }
    }
    // Budget exhausted: report the last state (converged stays false).
    if (!last_veff.empty())
        store_k_payload(rebuild_all_H(last_veff));
    fold_dispersion_k(res, P_k);
    res.e_fermi = use_smear ? last_mu : 0.0;
    res.e_entropy = use_smear ? (-degauss_ha * last_entropy) : 0.0;
    res.n_elec_occ = use_smear ? last_n_elec : n_elec;
    return res;
}

/// k-point collinear UKS SCF (Monkhorst-Pack / explicit list). Same Bloch
/// machinery as `run_gpw_scf_kpoints`, with dual spin densities Pa_k/Pb_k,
/// spin-resolved XC, and separate H_α^k / H_β^k + DIIS. Fixed integer
/// fillings n_alpha / n_beta by default; optional metallic smearing (FD/MP)
/// with a shared \(\mu\) over spin-orbitals (spin_degen=1).
static GpwResult run_gpw_scf_kpoints_uks(
    const std::vector<GpwAtom>& atoms, const GpwDft& cfg,
    const std::vector<GridBasisFunction>& basis,
    const std::vector<double>& radii, const int mimg[3],
    const UniformGrid& grid, const std::vector<double>& n_ion,
    const std::vector<double>& v_locsr, double e_self, double e_d3,
    const std::vector<internal::KPoint>& kpts,
    const std::vector<linalg::CplxMatrix>* guess_Pa_k = nullptr,
    const std::vector<linalg::CplxMatrix>* guess_Pb_k = nullptr) {
    using internal::CplxMatrix;
    using Cplx = std::complex<double>;
    const int nb = static_cast<int>(basis.size());
    const int nk = static_cast<int>(kpts.size());
    const int n_a = cfg.n_alpha;
    const int n_b = cfg.n_beta;

    // Dual-spin multi-k density guess (XL-BOMD): both or neither; shapes
    // match AO basis. Partial guesses would cold-start one channel silently.
    auto validate_spin_k_guess = [&](const std::vector<linalg::CplxMatrix>* g,
                                     const char* tag) {
        if (g == nullptr)
            return;
        if (static_cast<int>(g->size()) != nk)
            throw std::invalid_argument(std::string("GPW k-UKS: ") + tag
                                        + " size (" + std::to_string(g->size())
                                        + ") != n_kpts (" + std::to_string(nk)
                                        + ")");
        for (int ik = 0; ik < nk; ++ik) {
            const auto& Pk = (*g)[static_cast<std::size_t>(ik)];
            if (static_cast<int>(Pk.shape(0)) != nb
                || static_cast<int>(Pk.shape(1)) != nb)
                throw std::invalid_argument(
                    std::string("GPW k-UKS: ") + tag + "[" + std::to_string(ik)
                    + "] shape does not match AO basis (nb="
                    + std::to_string(nb) + ")");
        }
    };
    if ((guess_Pa_k == nullptr) != (guess_Pb_k == nullptr))
        throw std::invalid_argument(
            "GPW k-UKS: density guess requires both guess_P_k (alpha) and "
            "guess_P_k_beta (no partial spin guess)");
    validate_spin_k_guess(guess_Pa_k, "guess_P_k");
    validate_spin_k_guess(guess_Pb_k, "guess_P_k_beta");
    if (cfg.xc != XcKind::PADE && cfg.xc != XcKind::PBE
        && cfg.xc != XcKind::PBEsol && cfg.xc != XcKind::RevPBE
        && cfg.xc != XcKind::RPBE && cfg.xc != XcKind::BLYP
        && cfg.xc != XcKind::OptB88VdW)
        throw std::runtime_error("GPW k-UKS: XcKind must be PADE, PBE, PBEsol, "
                                 "RevPBE, RPBE, BLYP, or OptB88VdW");

    const GpwMpiLayout layout = gpw_mpi_layout();
    int n_kpts_local = 0;
    for (int ik = layout.rank; ik < nk; ik += layout.nprocs)
        ++n_kpts_local;

    std::vector<CplxMatrix> S_k, T_k;
    internal::build_overlap_kinetic_k(basis, radii, cfg.cell, mimg,
                                      kBuildScreen, kpts, S_k, T_k);
    std::vector<CplxMatrix> Vnl_k(static_cast<std::size_t>(nk),
                                  CplxMatrix(xt::zeros<Cplx>({nb, nb})));
    internal::add_nonlocal_kb_k(Vnl_k, basis, atoms, radii, cfg.cell, mimg,
                                kBuildScreen, kpts);

    std::vector<internal::KOrth> orth;
    orth.reserve(static_cast<std::size_t>(nk));
    const int n_need = std::max(n_a, n_b);
    int m_min = nb;
    for (int ik = 0; ik < nk; ++ik) {
        orth.emplace_back(S_k[static_cast<std::size_t>(ik)],
                          cfg.lindep_threshold);
        const int m = static_cast<int>(orth.back().X.shape(1));
        m_min = std::min(m_min, m);
        if (n_need > m)
            throw std::runtime_error(
                "GPW k-UKS: max(n_alpha,n_beta)=" + std::to_string(n_need)
                + " exceeds the " + std::to_string(m)
                + " independent basis directions at k #" + std::to_string(ik)
                + " (no-fallback rule)");
    }

    // Metallic smearing (FD / MP) — shared μ over α+β spin-orbitals at all k.
    // Occupation math is `cpu::pw::internal` with spin_degen=1; units: Ha.
    const bool use_smear = (cfg.smearing != Smearing::none);
    const double n_elec = static_cast<double>(n_a) + static_cast<double>(n_b);
    const double degauss_ha = use_smear ? cfg.degauss * 0.5 : 0.0; // Ry → Ha
    using pw::internal::SmearKind;
    SmearKind smear_kind = SmearKind::fermi_dirac;
    if (cfg.smearing == Smearing::methfessel_paxton)
        smear_kind = SmearKind::methfessel_paxton;
    const int mp_order = cfg.mp_order;
    if (use_smear && m_min < n_need + std::max(cfg.extra_bands, 0)
        && std::getenv("ATV_GPW_TRACE") && layout.rank == 0)
        std::fprintf(stderr,
                     "[gpw k-UKS] smear: m_min=%d n_need=%d extra_bands=%d "
                     "(full AO spectrum used)\n",
                     m_min, n_need, cfg.extra_bands);

    // Spin density from spinor occupations f_b ∈ [0, 1] (factor 1, not 2).
    auto density_from_occ = [&](const CplxMatrix& C,
                                const std::vector<double>& occ) {
        const int m = static_cast<int>(C.shape(1));
        CplxMatrix P = xt::zeros<Cplx>({nb, nb});
        const int nfill = std::min(m, static_cast<int>(occ.size()));
        for (int o = 0; o < nfill; ++o) {
            const double fo = occ[static_cast<std::size_t>(o)];
            if (!(fo > 1e-14))
                continue;
            for (int i = 0; i < nb; ++i) {
                const Cplx ci = fo * C(i, o);
                for (int j = 0; j < nb; ++j)
                    P(i, j) += ci * std::conj(C(j, o));
            }
        }
        return P;
    };
    auto density_spin = [&](const CplxMatrix& C, int n_occ) {
        std::vector<double> occ(static_cast<std::size_t>(n_occ), 1.0);
        return density_from_occ(C, occ);
    };

    std::vector<double> weights(static_cast<std::size_t>(nk));
    for (int ik = 0; ik < nk; ++ik)
        weights[static_cast<std::size_t>(ik)] =
            kpts[static_cast<std::size_t>(ik)].weight;

    // CP2K REL_CUTOFF multigrid on the multi-k UKS path (gated
    // ngrids>1 && rel_cutoff>0): same geometry-fixed product list as the RKS
    // multi-k multigrid, applied once per spin. Per spin σ the Bloch-phased
    // real-space density stack
    //   D^σ(t) = Σ_k w_k Re[P^{σ,k} e^{−2πi κ·t}]
    // weighs the per-level product collocation; ρ^σ_l prolongs to the fine
    // grid (ρ = ρα + ρβ feeds Poisson; (ρα, ρβ) feed spin XC); each V_eff^σ
    // restricts back and integrates into V^σ_t; phase-summed
    // V^{σ,k} = Σ_t e^{+2πi κ·t} V^σ_t (Hermitian, exact transpose of the
    // density assembly — SCF stays variational per spin).
    const bool use_mg = cfg.ngrids > 1 && cfg.rel_cutoff > 0.0;
    internal::GpwWorkspace ws;
    ws.grid = grid;
    MultiGrid& mg = ws.mg;
    ProductSet& products = ws.products;
    if (use_mg) {
        mg = make_multigrid(cfg.cell, cfg.cutoff_ry, cfg.ngrids,
                            cfg.rel_cutoff);
        if (mg.levels[0].n[0] != grid.n[0] || mg.levels[0].n[1] != grid.n[1]
            || mg.levels[0].n[2] != grid.n[2])
            throw std::logic_error(
                "GPW k-UKS: multigrid finest level n[] != SCF grid n[]");
        products = build_products(mg, basis, cfg.cell, cfg.rel_cutoff,
                                  cfg.eps_default, /*periodic=*/true);
    }
    const std::size_t n_tcells = products.cells.size();
    std::vector<std::vector<Cplx>> tcell_phase(static_cast<std::size_t>(nk));
    if (use_mg) {
        constexpr double kTwoPi = 6.283185307179586476925286766559;
        for (int ik = 0; ik < nk; ++ik) {
            auto& row = tcell_phase[static_cast<std::size_t>(ik)];
            row.resize(n_tcells);
            for (std::size_t c = 0; c < n_tcells; ++c) {
                const auto& t = products.cells[c];
                const double arg =
                    -kTwoPi
                    * (kpts[static_cast<std::size_t>(ik)].frac[0] * t[0]
                       + kpts[static_cast<std::size_t>(ik)].frac[1] * t[1]
                       + kpts[static_cast<std::size_t>(ik)].frac[2] * t[2]);
                row[c] = Cplx(std::cos(arg), std::sin(arg));
            }
        }
    }
    auto mg_density_stack = [&](const std::vector<CplxMatrix>& Pk) {
        std::vector<std::vector<double>> D(
            n_tcells,
            std::vector<double>(static_cast<std::size_t>(nb) * nb, 0.0));
        for (int ik = layout.rank; ik < nk; ik += layout.nprocs) {
            const double wk = weights[static_cast<std::size_t>(ik)];
            const auto& P = Pk[static_cast<std::size_t>(ik)];
            for (std::size_t c = 0; c < n_tcells; ++c) {
                const Cplx ph = tcell_phase[static_cast<std::size_t>(ik)][c];
                double* Dc = D[c].data();
                const Cplx* Pd = P.data();
                const std::size_t nn = static_cast<std::size_t>(nb) * nb;
                for (std::size_t e = 0; e < nn; ++e)
                    Dc[e] += wk
                             * (Pd[e].real() * ph.real()
                                - Pd[e].imag() * ph.imag());
            }
        }
#ifdef ATV_HAS_MPI
        if (layout.active)
            for (auto& Dc : D)
                MPI_Allreduce(MPI_IN_PLACE, Dc.data(),
                              static_cast<int>(Dc.size()), MPI_DOUBLE, MPI_SUM,
                              MPI_COMM_WORLD);
#endif
        return D;
    };
    auto mg_density = [&](const std::vector<CplxMatrix>& Pk) {
        const auto D = mg_density_stack(Pk);
        std::vector<double> rho(static_cast<std::size_t>(grid.npts()), 0.0);
        for (std::size_t l = 0; l < mg.levels.size(); ++l) {
            std::vector<double> rho_l = collocate_products_cells(
                mg.levels[l], products.by_level[l], products.prims,
                products.contribs, nb, D, cfg.eps_default);
            if (l == 0) {
                rho = std::move(rho_l);
            } else {
                const std::vector<double> up =
                    prolong_fft(mg.levels[l], mg.levels[0], rho_l,
                                /*spectral*/ true);
                for (std::size_t g = 0; g < rho.size(); ++g)
                    rho[g] += up[g];
            }
        }
        return rho;
    };
    auto mg_vt_stack = [&](const std::vector<double>& veff) {
        std::vector<std::vector<double>> Vt(
            n_tcells,
            std::vector<double>(static_cast<std::size_t>(nb) * nb, 0.0));
        for (std::size_t l = 0; l < mg.levels.size(); ++l) {
            std::vector<double> V_l;
            if (l == 0) {
                V_l = veff;
            } else {
                V_l = restrict_fft(mg.levels[0], mg.levels[l], veff,
                                   /*spectral*/ true);
            }
            const auto Vt_l = integrate_products_cells(
                mg.levels[l], products.by_level[l], products.prims,
                products.contribs, nb, V_l, n_tcells, cfg.eps_default);
            for (std::size_t c = 0; c < n_tcells; ++c)
                for (std::size_t e = 0; e < Vt[c].size(); ++e)
                    Vt[c][e] += Vt_l[c][e];
        }
        return Vt;
    };
    auto mg_phase_V = [&](const std::vector<std::vector<double>>& Vt, int ik) {
        CplxMatrix V = xt::zeros<Cplx>({nb, nb});
        const std::size_t nn = static_cast<std::size_t>(nb) * nb;
        for (std::size_t c = 0; c < n_tcells; ++c) {
            const Cplx ph =
                std::conj(tcell_phase[static_cast<std::size_t>(ik)][c]);
            const double* Vc = Vt[c].data();
            Cplx* out = V.data();
            for (std::size_t e = 0; e < nn; ++e)
                out[e] += ph * Vc[e];
        }
        return V;
    };
    // Dense χ^k path only when multigrid is off — products replace it.
    if (!use_mg)
        ws.kbg = internal::build_kpoint_basis_grid(grid, basis, kpts,
                                                   cfg.eps_default);
    const internal::KpointBasisGrid& kbg = ws.kbg;
    // k-summed spin density ρ^σ on the fine grid from {P^σ_k}.
    auto collocate_rho_spin_k = [&](const std::vector<CplxMatrix>& Pk) {
        if (use_mg)
            return mg_density(Pk);
        std::vector<double> rho(static_cast<std::size_t>(grid.npts()), 0.0);
        for (int ik = layout.rank; ik < nk; ik += layout.nprocs)
            internal::collocate_density_k(kbg, ik,
                                          Pk[static_cast<std::size_t>(ik)],
                                          weights[static_cast<std::size_t>(ik)],
                                          rho);
#ifdef ATV_HAS_MPI
        if (layout.active) {
            MPI_Allreduce(MPI_IN_PLACE, rho.data(),
                          static_cast<int>(rho.size()), MPI_DOUBLE, MPI_SUM,
                          MPI_COMM_WORLD);
        }
#endif
        return rho;
    };

    // Shared μ over α+β (spin_degen=1). Returns {occ_a_k, occ_b_k}.
    auto fill_metallic_uks =
        [&](const std::vector<std::vector<double>>& evals_a,
            const std::vector<std::vector<double>>& evals_b, double& mu_ha,
            double& n_occ_sum, double& entropy_sum) {
            std::vector<AccReal> all_evals;
            std::vector<AccReal> all_weights;
            for (int ik = 0; ik < nk; ++ik) {
                const double wk = weights[static_cast<std::size_t>(ik)];
                for (double e : evals_a[static_cast<std::size_t>(ik)]) {
                    all_evals.push_back(static_cast<AccReal>(e));
                    all_weights.push_back(static_cast<AccReal>(wk));
                }
                for (double e : evals_b[static_cast<std::size_t>(ik)]) {
                    all_evals.push_back(static_cast<AccReal>(e));
                    all_weights.push_back(static_cast<AccReal>(wk));
                }
            }
            mu_ha = static_cast<double>(pw::internal::find_fermi_level(
                all_evals, all_weights, static_cast<AccReal>(n_elec),
                static_cast<AccReal>(degauss_ha), smear_kind, mp_order,
                AccReal(1)));
            n_occ_sum = 0.0;
            entropy_sum = 0.0;
            std::vector<std::vector<double>> occ_a(
                static_cast<std::size_t>(nk));
            std::vector<std::vector<double>> occ_b(
                static_cast<std::size_t>(nk));
            for (int ik = 0; ik < nk; ++ik) {
                const double wk = weights[static_cast<std::size_t>(ik)];
                auto fill_spin = [&](const std::vector<double>& ev,
                                     std::vector<double>& occ_out) {
                    occ_out.resize(ev.size());
                    std::vector<AccReal> ev_a(ev.begin(), ev.end());
                    std::vector<AccReal> w_one(ev.size(),
                                               static_cast<AccReal>(wk));
                    auto fill = pw::internal::fill_occupations(
                        ev_a, w_one, static_cast<AccReal>(mu_ha),
                        static_cast<AccReal>(degauss_ha), smear_kind, mp_order,
                        AccReal(1));
                    n_occ_sum += static_cast<double>(fill.n_occ);
                    entropy_sum += static_cast<double>(fill.entropy);
                    for (std::size_t b = 0; b < ev.size(); ++b)
                        occ_out[b] =
                            static_cast<double>(pw::internal::smear_f_spinor(
                                static_cast<AccReal>(ev[b]),
                                static_cast<AccReal>(mu_ha),
                                static_cast<AccReal>(degauss_ha), smear_kind,
                                mp_order));
                };
                fill_spin(evals_a[static_cast<std::size_t>(ik)],
                          occ_a[static_cast<std::size_t>(ik)]);
                fill_spin(evals_b[static_cast<std::size_t>(ik)],
                          occ_b[static_cast<std::size_t>(ik)]);
            }
            return std::make_pair(std::move(occ_a), std::move(occ_b));
        };

    std::vector<CplxMatrix> Pa_k(static_cast<std::size_t>(nk));
    std::vector<CplxMatrix> Pb_k(static_cast<std::size_t>(nk));
    double last_mu = 0.0;
    double last_n_elec = 0.0;
    double last_entropy = 0.0;
    for (int ik = layout.rank; ik < nk; ik += layout.nprocs) {
        const CplxMatrix H0k = T_k[static_cast<std::size_t>(ik)]
                               + Vnl_k[static_cast<std::size_t>(ik)];
        auto [eps0, C0k] =
            internal::diagonalize_k(H0k, orth[static_cast<std::size_t>(ik)]);
        (void)eps0;
        // Cold start: fixed integer fill (smear path re-fills after first diag).
        // XL multi-k UKS warm start: dual-spin auxiliary densities when given.
        if (guess_Pa_k != nullptr) {
            Pa_k[static_cast<std::size_t>(ik)] =
                CplxMatrix((*guess_Pa_k)[static_cast<std::size_t>(ik)]);
            Pb_k[static_cast<std::size_t>(ik)] =
                CplxMatrix((*guess_Pb_k)[static_cast<std::size_t>(ik)]);
        } else {
            Pa_k[static_cast<std::size_t>(ik)] = density_spin(C0k, n_a);
            Pb_k[static_cast<std::size_t>(ik)] = density_spin(C0k, n_b);
        }
    }

    auto ctrace = [&](const CplxMatrix& P, const CplxMatrix& M) {
        double s = 0.0;
        for (int i = 0; i < nb; ++i)
            for (int j = 0; j < nb; ++j)
                s += (P(i, j) * M(j, i)).real();
        return s;
    };

    GpwResult res;
    res.n_kpts = nk;
    res.n_kpts_local = n_kpts_local;
    // Force path reuses geometry-fixed χ^k (single-grid only; MG has no kbg).
    if (!use_mg)
        res.scf_kbg = std::make_shared<internal::KpointBasisGrid>(ws.kbg);
    const bool use_diis = cfg.diis_depth > 0;
    internal::KDiis diis_a(use_diis ? cfg.diis_depth : 1, weights);
    internal::KDiis diis_b(use_diis ? cfg.diis_depth : 1, weights);
    constexpr double kDiisErrTolK = 1e-5;
    double e_prev = 0.0;

    // Dudarev multi-k setup (empty plus_u → no-op).
    std::vector<int> plus_u_z;
    std::vector<internal::PlusUHa> plus_u_slots;
    pack_plus_u(cfg.plus_u, plus_u_z, plus_u_slots);
    const bool do_dftu = !plus_u_slots.empty();
    const std::vector<internal::AoShellTag> ao_tags = global_ao_tags(atoms);
    std::vector<int> z_of_atom(atoms.size());
    for (std::size_t i = 0; i < atoms.size(); ++i)
        z_of_atom[i] = atoms[i].z;
    linalg::RealMatrix Va_dftu, Vb_dftu;
    bool have_V_dftu = false;

    auto rebuild_all_H = [&](const std::vector<double>& veff_spin) {
        // veff_spin layout: first npts α, next npts β.
        const long ng = grid.npts();
        std::vector<double> va(static_cast<std::size_t>(ng));
        std::vector<double> vb(static_cast<std::size_t>(ng));
        for (long g = 0; g < ng; ++g) {
            va[static_cast<std::size_t>(g)] =
                veff_spin[static_cast<std::size_t>(g)];
            vb[static_cast<std::size_t>(g)] =
                veff_spin[static_cast<std::size_t>(g + ng)];
        }
        // Multigrid: one V_t stack per spin (k-independent), phase-summed.
        std::vector<std::vector<double>> Vt_a, Vt_b;
        if (use_mg) {
            Vt_a = mg_vt_stack(va);
            Vt_b = mg_vt_stack(vb);
        }
        std::vector<CplxMatrix> Ha(static_cast<std::size_t>(nk));
        std::vector<CplxMatrix> Hb(static_cast<std::size_t>(nk));
        for (int ik = 0; ik < nk; ++ik) {
            const CplxMatrix Va =
                use_mg ? mg_phase_V(Vt_a, ik)
                       : internal::integrate_potential_k(kbg, ik, va);
            const CplxMatrix Vb =
                use_mg ? mg_phase_V(Vt_b, ik)
                       : internal::integrate_potential_k(kbg, ik, vb);
            const CplxMatrix H0 = T_k[static_cast<std::size_t>(ik)]
                                  + Vnl_k[static_cast<std::size_t>(ik)];
            Ha[static_cast<std::size_t>(ik)] = H0 + Va;
            Hb[static_cast<std::size_t>(ik)] = H0 + Vb;
            if (do_dftu && have_V_dftu) {
                internal::add_dftu_potential_cplx(
                    Ha[static_cast<std::size_t>(ik)], Va_dftu,
                    S_k[static_cast<std::size_t>(ik)]);
                internal::add_dftu_potential_cplx(
                    Hb[static_cast<std::size_t>(ik)], Vb_dftu,
                    S_k[static_cast<std::size_t>(ik)]);
            }
        }
        return std::make_pair(std::move(Ha), std::move(Hb));
    };

    auto store_k_payload = [&](const std::vector<CplxMatrix>& Ha_full,
                               const std::vector<CplxMatrix>& Hb_full) {
        res.k_frac.clear();
        res.k_weights.clear();
        res.eps_k.clear();
        res.eps_k_beta.clear();
        res.C_k.clear();
        res.C_k.reserve(static_cast<std::size_t>(nk));
        res.C_k_beta.clear();
        res.C_k_beta.reserve(static_cast<std::size_t>(nk));
        res.C_occ_k.clear();
        res.C_occ_k.resize(static_cast<std::size_t>(nk));
        res.C_occ_k_beta.clear();
        res.C_occ_k_beta.resize(static_cast<std::size_t>(nk));
        res.f_k.clear();
        res.f_k_beta.clear();
        res.P_k.clear();
        res.P_k.resize(static_cast<std::size_t>(nk));
        res.P_k_beta.clear();
        res.P_k_beta.resize(static_cast<std::size_t>(nk));
        std::vector<std::vector<double>> evals_a_store(
            static_cast<std::size_t>(nk));
        std::vector<std::vector<double>> evals_b_store(
            static_cast<std::size_t>(nk));
        std::vector<CplxMatrix> Ca_store(static_cast<std::size_t>(nk));
        std::vector<CplxMatrix> Cb_store(static_cast<std::size_t>(nk));
        for (int ik = 0; ik < nk; ++ik) {
            const auto& kp = kpts[static_cast<std::size_t>(ik)];
            res.k_frac.push_back({kp.frac[0], kp.frac[1], kp.frac[2]});
            res.k_weights.push_back(kp.weight);
            auto [epsa, Ca] =
                internal::diagonalize_k(Ha_full[static_cast<std::size_t>(ik)],
                                        orth[static_cast<std::size_t>(ik)]);
            auto [epsb, Cb] =
                internal::diagonalize_k(Hb_full[static_cast<std::size_t>(ik)],
                                        orth[static_cast<std::size_t>(ik)]);
            evals_a_store[static_cast<std::size_t>(ik)] = epsa;
            evals_b_store[static_cast<std::size_t>(ik)] = epsb;
            res.eps_k.push_back(std::move(epsa));
            res.eps_k_beta.push_back(std::move(epsb));
            Ca_store[static_cast<std::size_t>(ik)] = Ca;
            Cb_store[static_cast<std::size_t>(ik)] = Cb;
            // Full Bloch C_k / C_k_beta for multi-k UMP2 and free-energy forces.
            res.C_k.push_back(linalg::CplxMatrix(Ca));
            res.C_k_beta.push_back(linalg::CplxMatrix(Cb));
        }
        std::vector<std::vector<double>> occ_a_store, occ_b_store;
        if (use_smear) {
            double mu_tmp = last_mu, ne_tmp = last_n_elec, s_tmp = last_entropy;
            auto filled = fill_metallic_uks(evals_a_store, evals_b_store,
                                            mu_tmp, ne_tmp, s_tmp);
            occ_a_store = std::move(filled.first);
            occ_b_store = std::move(filled.second);
            last_mu = mu_tmp;
            last_n_elec = ne_tmp;
            last_entropy = s_tmp;
            res.f_k = occ_a_store;
            res.f_k_beta = occ_b_store;
        }
        for (int ik = 0; ik < nk; ++ik) {
            const CplxMatrix& Ca = Ca_store[static_cast<std::size_t>(ik)];
            const CplxMatrix& Cb = Cb_store[static_cast<std::size_t>(ik)];
            std::vector<Cplx> occ_a(static_cast<std::size_t>(nb)
                                    * static_cast<std::size_t>(n_a));
            for (int mu = 0; mu < nb; ++mu)
                for (int o = 0; o < n_a; ++o)
                    occ_a[static_cast<std::size_t>(mu) * n_a
                          + static_cast<std::size_t>(o)] = Ca(mu, o);
            res.C_occ_k[static_cast<std::size_t>(ik)] = std::move(occ_a);
            std::vector<Cplx> occ_b(static_cast<std::size_t>(nb)
                                    * static_cast<std::size_t>(n_b));
            for (int mu = 0; mu < nb; ++mu)
                for (int o = 0; o < n_b; ++o)
                    occ_b[static_cast<std::size_t>(mu) * n_b
                          + static_cast<std::size_t>(o)] = Cb(mu, o);
            res.C_occ_k_beta[static_cast<std::size_t>(ik)] = std::move(occ_b);
            if (use_smear) {
                res.P_k[static_cast<std::size_t>(ik)] = density_from_occ(
                    Ca, occ_a_store[static_cast<std::size_t>(ik)]);
                res.P_k_beta[static_cast<std::size_t>(ik)] = density_from_occ(
                    Cb, occ_b_store[static_cast<std::size_t>(ik)]);
            } else {
                // XL-BOMD dual-spin payload: P_k^σ from final MOs (occ = 1).
                res.P_k[static_cast<std::size_t>(ik)] = density_spin(Ca, n_a);
                res.P_k_beta[static_cast<std::size_t>(ik)] =
                    density_spin(Cb, n_b);
            }
        }
    };

    // Fold nonlocal vdW (DRSLL/LMKLL/rVV10), free-atom e_d3, or
    // Hirshfeld-scaled TS/MBD from k-summed total density ρ = ρα + ρβ
    // (post-SCF). Mirrors RKS `fold_dispersion_k` (multigrid product density
    // via collocate_rho_spin_k) and Γ UKS `fold_dispersion`.
    const bool periodic_cell_uks = !cfg.nonperiodic;
    auto fold_dispersion_k_uks = [&](GpwResult& out) {
        // Final nonlocal E_c^nl on the (self-consistent) k-summed total
        // density. V_nl was already applied every SCF iteration inside
        // build_uks_k_energy / the DIIS twin; this mirrors the multi-k RKS
        // post-SCF evaluation (gpw-vdw-kpoints) and Γ UKS e_disp fold.
        if (cfg.vdw_df || cfg.vdw_df2 || cfg.rvv10) {
            const std::vector<double> ra = collocate_rho_spin_k(Pa_k);
            const std::vector<double> rb = collocate_rho_spin_k(Pb_k);
            std::vector<double> rho(ra.size());
            for (std::size_t g = 0; g < rho.size(); ++g)
                rho[g] = ra[g] + rb[g];
            out.e_disp = nonlocal_vdw_on_grid(grid, rho, cfg.vdw_df,
                                              cfg.vdw_df2, cfg.rvv10,
                                              /*potential=*/false)
                             .energy;
            out.e_total += out.e_disp;
            return;
        }
        if (cfg.hirshfeld && (cfg.ts || cfg.mbd)) {
            const std::vector<double> ra = collocate_rho_spin_k(Pa_k);
            const std::vector<double> rb = collocate_rho_spin_k(Pb_k);
            std::vector<double> rho(ra.size());
            for (std::size_t g = 0; g < rho.size(); ++g)
                rho[g] = ra[g] + rb[g];
            std::vector<int> Z(atoms.size());
            std::vector<std::array<double, 3>> pos(Z.size());
            for (std::size_t i = 0; i < Z.size(); ++i) {
                Z[i] = atoms[i].z;
                pos[i] = {atoms[i].pos[0], atoms[i].pos[1], atoms[i].pos[2]};
            }
            out.volume_ratios = hirshfeld_volume_ratios(
                grid, rho, Z, pos, cfg.cell, periodic_cell_uks);
            if (cfg.mbd) {
                const bool mbd_per = cfg.mbd_k_mesh[0] > 0
                                     || cfg.mbd_k_mesh[1] > 0
                                     || cfg.mbd_k_mesh[2] > 0;
                if (mbd_per && cfg.nonperiodic)
                    throw std::runtime_error(
                        "GPW MBD: mbd_k_mesh requires a periodic cell "
                        "(!nonperiodic)");
                out.e_disp = mbd_rsscs_pbe(Z, pos, cfg.cell, mbd_per, 0.83, 6.0,
                                           out.volume_ratios,
                                           mbd_per ? cfg.mbd_k_mesh : nullptr)
                                 .energy;
            } else {
                constexpr double kAng2Bohr = 1.0 / 0.52917720859;
                constexpr double kRcutBohr = 50.0 * kAng2Bohr;
                out.e_disp = ts_pbe(Z, pos, cfg.cell, periodic_cell_uks,
                                    kRcutBohr, 0.94, 20.0, out.volume_ratios)
                                 .energy;
            }
            out.e_total += out.e_disp;
            return;
        }
        out.e_disp = e_d3;
        out.e_total += e_d3;
    };

    std::vector<CplxMatrix> Ha_k(static_cast<std::size_t>(nk));
    std::vector<CplxMatrix> Hb_k(static_cast<std::size_t>(nk));
    // Packed α||β on the workspace bag (same layout as before).
    std::vector<double>& last_veff_ab = ws.last_veff;

    // One dual-spin KS build at (Pa_k, Pb_k): collocate ρ^σ → Poisson → spin
    // XC → V_eff^σ → local H^σ_k + energy terms. Shared by the OT minimizer
    // and the fixed-density (max_iter == 0) eval below; the DIIS loop keeps
    // its historical inline copy (byte-identical assembly). Multigrid and
    // single-grid paths share this chokepoint via collocate_rho_spin_k /
    // mg_vt_stack + mg_phase_V.
    auto build_uks_k_energy = [&](const std::vector<CplxMatrix>& Pak,
                                  const std::vector<CplxMatrix>& Pbk)
        -> double {
        const std::vector<double> rho_a = collocate_rho_spin_k(Pak);
        const std::vector<double> rho_b = collocate_rho_spin_k(Pbk);
        const internal::KsGridUks ks = internal::ks_grid_build_uks(
            grid, rho_a, rho_b, n_ion, v_locsr, cfg.xc, /*nonperiodic=*/false,
            cfg.vdw_df, cfg.vdw_df2, cfg.rvv10);
        const double e_locgrid = ks.e_locgrid;
        const std::vector<double>& veff_a = ks.veff_a;
        const std::vector<double>& veff_b = ks.veff_b;
        last_veff_ab.resize(veff_a.size() + veff_b.size());
        for (std::size_t g = 0; g < veff_a.size(); ++g) {
            last_veff_ab[g] = veff_a[g];
            last_veff_ab[g + veff_a.size()] = veff_b[g];
        }
        // Force assembly reuses dual V_eff + V_es (same functional as the SCF).
        res.veff_a = veff_a;
        res.veff_b = veff_b;
        res.v_es = ks.v_es;
        std::vector<std::vector<double>> Vt_a, Vt_b;
        if (use_mg) {
            Vt_a = mg_vt_stack(veff_a);
            Vt_b = mg_vt_stack(veff_b);
        }
        double e_kin = 0.0, e_nl = 0.0;
        for (int ik = layout.rank; ik < nk; ik += layout.nprocs) {
            const double wk = weights[static_cast<std::size_t>(ik)];
            const CplxMatrix Va =
                use_mg ? mg_phase_V(Vt_a, ik)
                       : internal::integrate_potential_k(kbg, ik, veff_a);
            const CplxMatrix Vb =
                use_mg ? mg_phase_V(Vt_b, ik)
                       : internal::integrate_potential_k(kbg, ik, veff_b);
            const CplxMatrix H0 = T_k[static_cast<std::size_t>(ik)]
                                  + Vnl_k[static_cast<std::size_t>(ik)];
            Ha_k[static_cast<std::size_t>(ik)] = H0 + Va;
            Hb_k[static_cast<std::size_t>(ik)] = H0 + Vb;
            const CplxMatrix Ptot = Pak[static_cast<std::size_t>(ik)]
                                    + Pbk[static_cast<std::size_t>(ik)];
            e_kin += wk * ctrace(Ptot, T_k[static_cast<std::size_t>(ik)]);
            e_nl += wk * ctrace(Ptot, Vnl_k[static_cast<std::size_t>(ik)]);
        }
#ifdef ATV_HAS_MPI
        if (layout.active) {
            MPI_Allreduce(MPI_IN_PLACE, &e_kin, 1, MPI_DOUBLE, MPI_SUM,
                          MPI_COMM_WORLD);
            MPI_Allreduce(MPI_IN_PLACE, &e_nl, 1, MPI_DOUBLE, MPI_SUM,
                          MPI_COMM_WORLD);
        }
#endif
        double e_dftu = 0.0;
        if (do_dftu) {
            e_dftu += internal::apply_dft_plus_u_k(
                Ha_k, Pak, S_k, weights, layout.rank, layout.nprocs,
                layout.active, ao_tags, z_of_atom, plus_u_slots, plus_u_z,
                &Va_dftu);
            e_dftu += internal::apply_dft_plus_u_k(
                Hb_k, Pbk, S_k, weights, layout.rank, layout.nprocs,
                layout.active, ao_tags, z_of_atom, plus_u_slots, plus_u_z,
                &Vb_dftu);
            have_V_dftu = true;
        }
        res.e_dftu = e_dftu;
        res.e_kinetic = e_kin;
        res.e_nonlocal = 0.0;
        res.e_loc_sr = e_nl + e_locgrid;
        res.e_es = ks.e_hartree - e_self;
        res.e_xc = ks.e_xc;
        const double e_ks =
            res.e_kinetic + res.e_loc_sr + res.e_es + res.e_xc + e_dftu;
        const double e_ent = use_smear ? (-degauss_ha * last_entropy) : 0.0;
        res.e_entropy = e_ent;
        res.e_fermi = use_smear ? last_mu : 0.0;
        res.n_elec_occ = use_smear ? last_n_elec : n_elec;
        return e_ks + e_ent;
    };

    // ── Multi-k UKS fixed-density KS eval (linearized-XL warm step / the
    //    dual-spin Bloch CPMD electronic step) ───────────────────────────────
    // max_iter == 0: one dual-spin collocate→Poisson→spin-XC→integrate at the
    // supplied {P_k^α, P_k^β}; no diagonalization, no mixing, no orbital
    // update. Populates H_ks_k (α) / H_ks_k_beta (β) / S_k — the dual-spin
    // sibling of the multi-k RKS branch in run_gpw_scf_kpoints.
    if (cfg.max_iter == 0) {
        if (guess_Pa_k == nullptr || guess_Pb_k == nullptr)
            throw std::invalid_argument(
                "GPW k-UKS: max_iter==0 (fixed-density) requires the dual "
                "guess_P_k / guess_P_k_beta densities");
        if (cfg.use_ot)
            throw std::invalid_argument(
                "GPW k-UKS: fixed-density eval is incompatible with use_ot");
        if (use_smear)
            throw std::invalid_argument(
                "GPW k-UKS: fixed-density eval with metallic smearing is not "
                "productized (fractional occupations need a diagonalization)");
        // Every rank holds the full dual {P_k^σ} (init loop fills local k).
        for (int ik = 0; ik < nk; ++ik) {
            Pa_k[static_cast<std::size_t>(ik)] =
                CplxMatrix((*guess_Pa_k)[static_cast<std::size_t>(ik)]);
            Pb_k[static_cast<std::size_t>(ik)] =
                CplxMatrix((*guess_Pb_k)[static_cast<std::size_t>(ik)]);
        }
        res.e_total = build_uks_k_energy(Pa_k, Pb_k);
        // Full H^σ_k on every rank from the shared dual V_eff.
        auto [Ha_full, Hb_full] = rebuild_all_H(last_veff_ab);
        res.H_ks_k.resize(static_cast<std::size_t>(nk));
        res.H_ks_k_beta.resize(static_cast<std::size_t>(nk));
        res.S_k.resize(static_cast<std::size_t>(nk));
        res.P_k.resize(static_cast<std::size_t>(nk));
        res.P_k_beta.resize(static_cast<std::size_t>(nk));
        res.k_frac.clear();
        res.k_weights.clear();
        res.k_frac.reserve(static_cast<std::size_t>(nk));
        res.k_weights.reserve(static_cast<std::size_t>(nk));
        for (int ik = 0; ik < nk; ++ik) {
            const auto& kp = kpts[static_cast<std::size_t>(ik)];
            res.k_frac.push_back({kp.frac[0], kp.frac[1], kp.frac[2]});
            res.k_weights.push_back(kp.weight);
            res.H_ks_k[static_cast<std::size_t>(ik)] =
                Ha_full[static_cast<std::size_t>(ik)];
            res.H_ks_k_beta[static_cast<std::size_t>(ik)] =
                Hb_full[static_cast<std::size_t>(ik)];
            res.S_k[static_cast<std::size_t>(ik)] =
                S_k[static_cast<std::size_t>(ik)];
            res.P_k[static_cast<std::size_t>(ik)] =
                Pa_k[static_cast<std::size_t>(ik)];
            res.P_k_beta[static_cast<std::size_t>(ik)] =
                Pb_k[static_cast<std::size_t>(ik)];
        }
        // Integer-occupation ⟨S²⟩ seed (same convention as the SCF exits).
        {
            const double sz = 0.5 * static_cast<double>(n_a - n_b);
            res.s2 = sz * (sz + 1.0) + static_cast<double>(n_b);
        }
        res.converged = true;
        res.iters = 0;
        fold_dispersion_k_uks(res);
        return res;
    }

    // ── Multi-k UKS OT (integer / metallic FD/MP / +DFT+U; dual spin) ────
    // Independent complex OT manifolds per (k, σ). Integer path: occ=1
    // open-shell gradient, manifold size n_α / n_β. Metallic smear: full AO
    // spectrum, shared μ over α+β (spin_degen=1), f_σ ∈ [0,1], Mermin free
    // energy F = E_KS − σ S. Joint PR+ CG + Armijo on rebuilt F.
    // Integer + PLUS_U: CP2K-style OUTER_SCF — rebuild H^σ (incl. V_U^σ),
    // then fixed-H OT band minimize per (k, σ). Metallic smear with or
    // without PLUS_U: joint free-energy PR+ Armijo (build_uks_k_energy
    // folds V_U / E_U when plus_u is set). Same fixed point as multi-k UKS
    // DIIS (+smear / +U / +U+smear).
    if (cfg.use_ot) {
        constexpr double kOtGradTolK = 1e-5;
        // Active manifold: integer keeps n_a / n_b; smear uses full spectrum.
        const int n_act_a = use_smear ? m_min : n_a;
        const int n_act_b = use_smear ? m_min : n_b;
        if (use_smear && m_min < 1)
            throw std::runtime_error(
                "GPW k-UKS OT: n_active < 1 under metallic smearing");

        std::vector<CplxMatrix> Cocc_a_k(static_cast<std::size_t>(nk));
        std::vector<CplxMatrix> Cocc_b_k(static_cast<std::size_t>(nk));
        // Per-k spinor occupations f ∈ [0,1]; integer path all 1 on occupied.
        std::vector<std::vector<double>> occ_a_k(
            static_cast<std::size_t>(nk),
            std::vector<double>(static_cast<std::size_t>(std::max(n_act_a, 0)),
                                1.0));
        std::vector<std::vector<double>> occ_b_k(
            static_cast<std::size_t>(nk),
            std::vector<double>(static_cast<std::size_t>(std::max(n_act_b, 0)),
                                1.0));
        if (!use_smear) {
            for (auto& o : occ_a_k)
                o.assign(static_cast<std::size_t>(std::max(n_act_a, 0)), 1.0);
            for (auto& o : occ_b_k)
                o.assign(static_cast<std::size_t>(std::max(n_act_b, 0)), 1.0);
        }

        // Dual-spin natural-orbital warm start (SGCP OT corrector): purify
        // each predicted P_k^σ to natural orbitals — eigenvalues of
        // X† P^σ X ascend, so the largest occupations sit in the LAST
        // columns (same convention as the multi-k RKS warm start in the
        // run_gpw_scf_kpoints init loop and the Γ-UKS guess purification).
        // Integer-occupation path only: under metallic smearing the OT
        // manifold is the full AO spectrum and the RKS precedent discards
        // the guess (cold H0 expand); the evaluator exposes no smearing,
        // so the SGCP corrector never reaches it.
        auto warm_from_guess = [&](const CplxMatrix& Pg, int n_occ_s, int ik) {
            if (n_occ_s < 1)
                return CplxMatrix(xt::zeros<Cplx>({nb, 0}));
            auto [n_occ_vals, Cnat] = internal::diagonalize_k(
                Pg, orth[static_cast<std::size_t>(ik)]);
            (void)n_occ_vals;
            const int m = static_cast<int>(Cnat.shape(1));
            if (m < n_occ_s)
                throw std::runtime_error(
                    "GPW k-UKS OT: natural-orbital warm start needs at "
                    "least n_occ kept orthogonal directions (m="
                    + std::to_string(m) + ", n_occ=" + std::to_string(n_occ_s)
                    + ")");
            CplxMatrix Cw = xt::zeros<Cplx>({nb, n_occ_s});
            for (int i = 0; i < n_occ_s; ++i) {
                const int src = m - n_occ_s + i;
                for (int mu = 0; mu < nb; ++mu)
                    Cw(mu, i) = Cnat(mu, src);
            }
            return linalg::lowdin_orthonormalize_k(
                Cw, S_k[static_cast<std::size_t>(ik)]);
        };
        const bool warm_ot_uks = (guess_Pa_k != nullptr) && !use_smear;

        for (int ik = layout.rank; ik < nk; ik += layout.nprocs) {
            if (warm_ot_uks) {
                Pa_k[static_cast<std::size_t>(ik)] =
                    CplxMatrix((*guess_Pa_k)[static_cast<std::size_t>(ik)]);
                Pb_k[static_cast<std::size_t>(ik)] =
                    CplxMatrix((*guess_Pb_k)[static_cast<std::size_t>(ik)]);
                Cocc_a_k[static_cast<std::size_t>(ik)] = warm_from_guess(
                    Pa_k[static_cast<std::size_t>(ik)], n_act_a, ik);
                Cocc_b_k[static_cast<std::size_t>(ik)] = warm_from_guess(
                    Pb_k[static_cast<std::size_t>(ik)], n_act_b, ik);
                continue;
            }
            const CplxMatrix H0k = T_k[static_cast<std::size_t>(ik)]
                                   + Vnl_k[static_cast<std::size_t>(ik)];
            auto [eps0, C0k] = internal::diagonalize_k(
                H0k, orth[static_cast<std::size_t>(ik)]);
            (void)eps0;
            const auto& S = S_k[static_cast<std::size_t>(ik)];
            if (n_act_a > 0) {
                const int mcols =
                    std::min(n_act_a, static_cast<int>(C0k.shape(1)));
                CplxMatrix Ca = xt::zeros<Cplx>({nb, n_act_a});
                for (int mu = 0; mu < nb; ++mu)
                    for (int o = 0; o < mcols; ++o)
                        Ca(mu, o) = C0k(mu, o);
                Cocc_a_k[static_cast<std::size_t>(ik)] =
                    linalg::lowdin_orthonormalize_k(Ca, S);
            } else {
                Cocc_a_k[static_cast<std::size_t>(ik)] =
                    xt::zeros<Cplx>({nb, 0});
            }
            if (n_act_b > 0) {
                const int mcols =
                    std::min(n_act_b, static_cast<int>(C0k.shape(1)));
                CplxMatrix Cb = xt::zeros<Cplx>({nb, n_act_b});
                for (int mu = 0; mu < nb; ++mu)
                    for (int o = 0; o < mcols; ++o)
                        Cb(mu, o) = C0k(mu, o);
                Cocc_b_k[static_cast<std::size_t>(ik)] =
                    linalg::lowdin_orthonormalize_k(Cb, S);
            } else {
                Cocc_b_k[static_cast<std::size_t>(ik)] =
                    xt::zeros<Cplx>({nb, 0});
            }
            // Cold density: integer fill of seed n_a / n_b (smear re-fills).
            Pa_k[static_cast<std::size_t>(ik)] = density_spin(C0k, n_a);
            Pb_k[static_cast<std::size_t>(ik)] = density_spin(C0k, n_b);
        }

        auto dens_from_cocc = [&](const CplxMatrix& C, int n_int,
                                  const std::vector<double>& f) {
            if (use_smear)
                return density_from_occ(C, f);
            return density_spin(C, n_int);
        };

        auto frob_k = [](const CplxMatrix& a, const CplxMatrix& b) {
            double s = 0.0;
            const int n0 = static_cast<int>(a.shape(0));
            const int n1 = static_cast<int>(a.shape(1));
            for (int i = 0; i < n0; ++i)
                for (int j = 0; j < n1; ++j)
                    s += (std::conj(a(i, j)) * b(i, j)).real();
            return s;
        };

        // Subspace-canonicalize C† H C → (ε, U), C ← C U; fill metallic f.
        auto refresh_occupations_uks = [&]() {
            if (!use_smear)
                return;
            std::vector<std::vector<double>> evals_a(
                static_cast<std::size_t>(nk));
            std::vector<std::vector<double>> evals_b(
                static_cast<std::size_t>(nk));
            auto canon = [&](CplxMatrix& C, const CplxMatrix& H,
                             const CplxMatrix& S, int n_act,
                             std::vector<double>& evals_out) {
                if (n_act < 1) {
                    evals_out.clear();
                    return;
                }
                CplxMatrix h = xt::zeros<Cplx>({n_act, n_act});
                for (int i = 0; i < n_act; ++i)
                    for (int j = 0; j < n_act; ++j) {
                        Cplx s(0, 0);
                        for (int mu = 0; mu < nb; ++mu) {
                            Cplx hcv(0, 0);
                            for (int nu = 0; nu < nb; ++nu)
                                hcv += H(mu, nu) * C(nu, j);
                            s += std::conj(C(mu, i)) * hcv;
                        }
                        h(i, j) = s;
                    }
                std::vector<double> eps;
                CplxMatrix U;
                linalg::eigh_hermitian(h, eps, U);
                CplxMatrix Cnew = xt::zeros<Cplx>({nb, n_act});
                for (int mu = 0; mu < nb; ++mu)
                    for (int o = 0; o < n_act; ++o) {
                        Cplx s(0, 0);
                        for (int p = 0; p < n_act; ++p)
                            s += C(mu, p) * U(p, o);
                        Cnew(mu, o) = s;
                    }
                C = linalg::lowdin_orthonormalize_k(Cnew, S);
                evals_out = std::move(eps);
            };
            for (int ik = layout.rank; ik < nk; ik += layout.nprocs) {
                canon(Cocc_a_k[static_cast<std::size_t>(ik)],
                      Ha_k[static_cast<std::size_t>(ik)],
                      S_k[static_cast<std::size_t>(ik)], n_act_a,
                      evals_a[static_cast<std::size_t>(ik)]);
                canon(Cocc_b_k[static_cast<std::size_t>(ik)],
                      Hb_k[static_cast<std::size_t>(ik)],
                      S_k[static_cast<std::size_t>(ik)], n_act_b,
                      evals_b[static_cast<std::size_t>(ik)]);
            }
#ifdef ATV_HAS_MPI
            if (layout.active) {
                auto allgather_evals = [&](std::vector<std::vector<double>>&
                                               evals,
                                           int n_act) {
                    if (n_act < 1)
                        return;
                    std::vector<double> pack(
                        static_cast<std::size_t>(nk * n_act), 0.0);
                    std::vector<int> nband(static_cast<std::size_t>(nk), 0);
                    for (int ik = layout.rank; ik < nk; ik += layout.nprocs) {
                        const auto& ev = evals[static_cast<std::size_t>(ik)];
                        nband[static_cast<std::size_t>(ik)] =
                            static_cast<int>(ev.size());
                        for (std::size_t b = 0; b < ev.size(); ++b)
                            pack[static_cast<std::size_t>(ik * n_act) + b] =
                                ev[b];
                    }
                    MPI_Allreduce(MPI_IN_PLACE, pack.data(), nk * n_act,
                                  MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
                    MPI_Allreduce(MPI_IN_PLACE, nband.data(), nk, MPI_INT,
                                  MPI_SUM, MPI_COMM_WORLD);
                    for (int ik = 0; ik < nk; ++ik) {
                        const int m = nband[static_cast<std::size_t>(ik)];
                        evals[static_cast<std::size_t>(ik)].resize(
                            static_cast<std::size_t>(m));
                        for (int b = 0; b < m; ++b)
                            evals[static_cast<std::size_t>(
                                ik)][static_cast<std::size_t>(b)] =
                                pack[static_cast<std::size_t>(ik * n_act + b)];
                    }
                };
                allgather_evals(evals_a, n_act_a);
                allgather_evals(evals_b, n_act_b);
            }
#endif
            auto filled = fill_metallic_uks(evals_a, evals_b, last_mu,
                                            last_n_elec, last_entropy);
            for (int ik = 0; ik < nk; ++ik) {
                occ_a_k[static_cast<std::size_t>(ik)] =
                    std::move(filled.first[static_cast<std::size_t>(ik)]);
                occ_b_k[static_cast<std::size_t>(ik)] =
                    std::move(filled.second[static_cast<std::size_t>(ik)]);
                occ_a_k[static_cast<std::size_t>(ik)].resize(
                    static_cast<std::size_t>(std::max(n_act_a, 0)), 0.0);
                occ_b_k[static_cast<std::size_t>(ik)].resize(
                    static_cast<std::size_t>(std::max(n_act_b, 0)), 0.0);
            }
        };

        // ── OUTER_SCF + fixed-H OT for multi-k UKS OT+DFT+U (integer) ────
        // Metallic U+smear rides the free-energy PR+ path below (same as RKS
        // OT+U+smear): OUTER_SCF freezes occupations and stalls under FD.
        if (do_dftu && !use_smear) {
            constexpr double kInnerTol = 1e-7;
            constexpr int kInnerMax = 80;
            for (int outer = 0; outer < cfg.max_iter; ++outer) {
                for (int ik = layout.rank; ik < nk; ik += layout.nprocs) {
                    Pa_k[static_cast<std::size_t>(ik)] = density_spin(
                        Cocc_a_k[static_cast<std::size_t>(ik)], n_a);
                    Pb_k[static_cast<std::size_t>(ik)] = density_spin(
                        Cocc_b_k[static_cast<std::size_t>(ik)], n_b);
                }
                const double e_total = build_uks_k_energy(Pa_k, Pb_k);
                res.iters = outer + 1;
                res.e_total = e_total;

                if (std::getenv("ATV_GPW_TRACE") && layout.rank == 0)
                    std::fprintf(
                        stderr,
                        "[gpw k-UKS OT+U outer %2d] E=%.8f  U=%.4f  dE=%.2e "
                        "(nk=%d Na=%d Nb=%d)\n",
                        outer, e_total, res.e_dftu, std::abs(e_total - e_prev),
                        nk, n_a, n_b);

                if (outer > 0 && std::abs(e_total - e_prev) < cfg.tol) {
                    res.converged = true;
                    auto [Ha_full, Hb_full] = rebuild_all_H(last_veff_ab);
                    store_k_payload(Ha_full, Hb_full);
                    {
                        const double sz = 0.5 * static_cast<double>(n_a - n_b);
                        res.s2 = sz * (sz + 1.0) + static_cast<double>(n_b);
                    }
                    fold_dispersion_k_uks(res);
                    res.n_elec_occ = n_elec;
                    return res;
                }
                e_prev = e_total;

                // Inner: fixed-H OT → occupied eigenspace of current H_k^σ.
                for (int ik = layout.rank; ik < nk; ik += layout.nprocs) {
                    const auto& S = S_k[static_cast<std::size_t>(ik)];
                    const auto& T = T_k[static_cast<std::size_t>(ik)];
                    if (n_a > 0) {
                        const auto ot = internal::ot_minimize_fixed_k(
                            Ha_k[static_cast<std::size_t>(ik)], S, T,
                            Cocc_a_k[static_cast<std::size_t>(ik)], kInnerTol,
                            kInnerMax);
                        Cocc_a_k[static_cast<std::size_t>(ik)] = ot.C;
                        Pa_k[static_cast<std::size_t>(ik)] =
                            density_spin(ot.C, n_a);
                    }
                    if (n_b > 0) {
                        const auto ot = internal::ot_minimize_fixed_k(
                            Hb_k[static_cast<std::size_t>(ik)], S, T,
                            Cocc_b_k[static_cast<std::size_t>(ik)], kInnerTol,
                            kInnerMax);
                        Cocc_b_k[static_cast<std::size_t>(ik)] = ot.C;
                        Pb_k[static_cast<std::size_t>(ik)] =
                            density_spin(ot.C, n_b);
                    }
                }
            }
            if (!last_veff_ab.empty()) {
                auto [Ha_full, Hb_full] = rebuild_all_H(last_veff_ab);
                store_k_payload(Ha_full, Hb_full);
            }
            {
                const double sz = 0.5 * static_cast<double>(n_a - n_b);
                res.s2 = sz * (sz + 1.0) + static_cast<double>(n_b);
            }
            fold_dispersion_k_uks(res);
            res.n_elec_occ = n_elec;
            return res;
        }

        // ── One PR+ Armijo step per outer H rebuild (smear and/or no U) ──
        // Active OT manifold: integer path keeps n_α / n_β; metallic smear
        // uses the full AO spectrum. PLUS_U + smear rides this path (V_U via
        // build_uks_k_energy) — product: OT+U+smear ≡ DIIS+U+smear.
        std::vector<CplxMatrix> d_a_k(static_cast<std::size_t>(nk));
        std::vector<CplxMatrix> d_b_k(static_cast<std::size_t>(nk));
        std::vector<CplxMatrix> pz_a_prev_k(static_cast<std::size_t>(nk));
        std::vector<CplxMatrix> pz_b_prev_k(static_cast<std::size_t>(nk));
        double denom_prev = 1.0;
        bool have_prev = false;
        double e_prev_ot = 0.0;
        // After an accepted Armijo trial, Ha/Hb already match dens(C,f).
        bool h_matches_manifold = false;

        for (int iter = 0; iter < cfg.max_iter; ++iter) {
            // Integer: one ρ→H (refresh_occupations_uks is a no-op without
            // smear). Metallic: H → fill f → rebuild P → H; skip the first
            // when H already matches dens(C,f_old) from the previous accept.
            for (int ik = layout.rank; ik < nk; ik += layout.nprocs) {
                Pa_k[static_cast<std::size_t>(ik)] =
                    dens_from_cocc(Cocc_a_k[static_cast<std::size_t>(ik)], n_a,
                                   occ_a_k[static_cast<std::size_t>(ik)]);
                Pb_k[static_cast<std::size_t>(ik)] =
                    dens_from_cocc(Cocc_b_k[static_cast<std::size_t>(ik)], n_b,
                                   occ_b_k[static_cast<std::size_t>(ik)]);
            }
            double e_total = 0.0;
            if (use_smear) {
                if (!h_matches_manifold)
                    (void)build_uks_k_energy(Pa_k, Pb_k);
                refresh_occupations_uks();
                for (int ik = layout.rank; ik < nk; ik += layout.nprocs) {
                    Pa_k[static_cast<std::size_t>(ik)] = dens_from_cocc(
                        Cocc_a_k[static_cast<std::size_t>(ik)], n_a,
                        occ_a_k[static_cast<std::size_t>(ik)]);
                    Pb_k[static_cast<std::size_t>(ik)] = dens_from_cocc(
                        Cocc_b_k[static_cast<std::size_t>(ik)], n_b,
                        occ_b_k[static_cast<std::size_t>(ik)]);
                }
                e_total = build_uks_k_energy(Pa_k, Pb_k);
            } else if (!h_matches_manifold) {
                e_total = build_uks_k_energy(Pa_k, Pb_k);
            } else {
                e_total = res.e_total; // carried from accepted Armijo trial
            }
            h_matches_manifold = true;
            res.iters = iter + 1;
            res.e_total = e_total;

            double gnorm = 0.0;
            std::vector<CplxMatrix> g_a_k(static_cast<std::size_t>(nk));
            std::vector<CplxMatrix> g_b_k(static_cast<std::size_t>(nk));
            std::vector<CplxMatrix> pz_a_k(static_cast<std::size_t>(nk));
            std::vector<CplxMatrix> pz_b_k(static_cast<std::size_t>(nk));
            for (int ik = layout.rank; ik < nk; ik += layout.nprocs) {
                const auto& S = S_k[static_cast<std::size_t>(ik)];
                if (n_act_a > 0) {
                    const auto& Ha = Ha_k[static_cast<std::size_t>(ik)];
                    const auto& Ca = Cocc_a_k[static_cast<std::size_t>(ik)];
                    const CplxMatrix g =
                        use_smear ? linalg::band_gradient_k(
                                        Ha, S, Ca,
                                        occ_a_k[static_cast<std::size_t>(ik)])
                                  : linalg::band_gradient_k(Ha, S, Ca, 1.0);
                    for (const Cplx v : g)
                        gnorm = std::max(gnorm, std::abs(v));
                    const internal::OtPrecondSingleInverseK pre(Ha, S, Ca);
                    pz_a_k[static_cast<std::size_t>(ik)] =
                        linalg::s_tangent_k(Ca, S, pre.apply(g));
                    g_a_k[static_cast<std::size_t>(ik)] = g;
                } else {
                    g_a_k[static_cast<std::size_t>(ik)] =
                        xt::zeros<Cplx>({nb, 0});
                    pz_a_k[static_cast<std::size_t>(ik)] =
                        xt::zeros<Cplx>({nb, 0});
                }
                if (n_act_b > 0) {
                    const auto& Hb = Hb_k[static_cast<std::size_t>(ik)];
                    const auto& Cb = Cocc_b_k[static_cast<std::size_t>(ik)];
                    const CplxMatrix g =
                        use_smear ? linalg::band_gradient_k(
                                        Hb, S, Cb,
                                        occ_b_k[static_cast<std::size_t>(ik)])
                                  : linalg::band_gradient_k(Hb, S, Cb, 1.0);
                    for (const Cplx v : g)
                        gnorm = std::max(gnorm, std::abs(v));
                    const internal::OtPrecondSingleInverseK pre(Hb, S, Cb);
                    pz_b_k[static_cast<std::size_t>(ik)] =
                        linalg::s_tangent_k(Cb, S, pre.apply(g));
                    g_b_k[static_cast<std::size_t>(ik)] = g;
                } else {
                    g_b_k[static_cast<std::size_t>(ik)] =
                        xt::zeros<Cplx>({nb, 0});
                    pz_b_k[static_cast<std::size_t>(ik)] =
                        xt::zeros<Cplx>({nb, 0});
                }
            }
#ifdef ATV_HAS_MPI
            if (layout.active) {
                double gglob = gnorm;
                MPI_Allreduce(&gnorm, &gglob, 1, MPI_DOUBLE, MPI_MAX,
                              MPI_COMM_WORLD);
                gnorm = gglob;
            }
#endif
            if (std::getenv("ATV_GPW_TRACE") && layout.rank == 0) {
                if (use_smear)
                    std::fprintf(stderr,
                                 "[gpw k-UKS OT %2d] F=%.8f  T=%.4f loc=%.4f "
                                 "es=%.4f xc=%.4f  dE=%.2e gnorm=%.2e mu=%.6f "
                                 "Ne=%.6f (nk=%d n_act=%d/%d)\n",
                                 iter, e_total, res.e_kinetic, res.e_loc_sr,
                                 res.e_es, res.e_xc,
                                 std::abs(e_total - e_prev_ot), gnorm, last_mu,
                                 last_n_elec, nk, n_act_a, n_act_b);
                else
                    std::fprintf(stderr,
                                 "[gpw k-UKS OT %2d] E=%.8f  T=%.4f loc=%.4f "
                                 "es=%.4f xc=%.4f  dE=%.2e gnorm=%.2e "
                                 "(nk=%d Na=%d Nb=%d)\n",
                                 iter, e_total, res.e_kinetic, res.e_loc_sr,
                                 res.e_es, res.e_xc,
                                 std::abs(e_total - e_prev_ot), gnorm, nk, n_a,
                                 n_b);
            }

            // Metallic / PLUS_U softens g — accept energy plateau with soft
            // gtol (same policy as multi-k RKS OT+U+smear).
            const double gtol = (use_smear || do_dftu)
                                    ? std::max(kOtGradTolK, 5e-4)
                                    : kOtGradTolK;
            auto finish_converged = [&]() {
                res.converged = true;
                auto [Ha_full, Hb_full] = rebuild_all_H(last_veff_ab);
                store_k_payload(Ha_full, Hb_full);
                {
                    const double sz = 0.5 * static_cast<double>(n_a - n_b);
                    res.s2 = sz * (sz + 1.0) + static_cast<double>(n_b);
                }
                fold_dispersion_k_uks(res);
                res.e_fermi = use_smear ? last_mu : 0.0;
                res.e_entropy = use_smear ? (-degauss_ha * last_entropy) : 0.0;
                res.n_elec_occ = use_smear ? last_n_elec : n_elec;
            };
            if (iter > 0 && gnorm < gtol
                && std::abs(e_total - e_prev_ot) < cfg.tol) {
                finish_converged();
                return res;
            }
            // Energy plateau fallback for metals / OT+U+smear.
            if ((use_smear || do_dftu) && iter > 15
                && std::abs(e_total - e_prev_ot) < cfg.tol) {
                finish_converged();
                return res;
            }
            e_prev_ot = e_total;

            // Joint Polak–Ribière+ over all (k, σ).
            double num = 0.0, den = 0.0;
            for (int ik = layout.rank; ik < nk; ik += layout.nprocs) {
                const double wk = weights[static_cast<std::size_t>(ik)];
                if (n_act_a > 0) {
                    const auto& g = g_a_k[static_cast<std::size_t>(ik)];
                    const auto& pz = pz_a_k[static_cast<std::size_t>(ik)];
                    den += wk * frob_k(g, pz);
                    if (have_prev)
                        num +=
                            wk
                            * frob_k(
                                g,
                                pz - pz_a_prev_k[static_cast<std::size_t>(ik)]);
                }
                if (n_act_b > 0) {
                    const auto& g = g_b_k[static_cast<std::size_t>(ik)];
                    const auto& pz = pz_b_k[static_cast<std::size_t>(ik)];
                    den += wk * frob_k(g, pz);
                    if (have_prev)
                        num +=
                            wk
                            * frob_k(
                                g,
                                pz - pz_b_prev_k[static_cast<std::size_t>(ik)]);
                }
            }
#ifdef ATV_HAS_MPI
            if (layout.active) {
                double pack[2] = {num, den};
                MPI_Allreduce(MPI_IN_PLACE, pack, 2, MPI_DOUBLE, MPI_SUM,
                              MPI_COMM_WORLD);
                num = pack[0];
                den = pack[1];
            }
#endif
            const double beta = (!have_prev || !(denom_prev > 0.0))
                                    ? 0.0
                                    : std::max(0.0, num / denom_prev);
            denom_prev = (den > 1e-300) ? den : 1.0;

            double slope_dot = 0.0;
            for (int ik = layout.rank; ik < nk; ik += layout.nprocs) {
                const auto& S = S_k[static_cast<std::size_t>(ik)];
                const double wk = weights[static_cast<std::size_t>(ik)];
                if (n_act_a > 0) {
                    const auto& Ca = Cocc_a_k[static_cast<std::size_t>(ik)];
                    const auto& g = g_a_k[static_cast<std::size_t>(ik)];
                    const auto& pz = pz_a_k[static_cast<std::size_t>(ik)];
                    CplxMatrix d;
                    if (!have_prev)
                        d = -pz;
                    else
                        d = linalg::s_tangent_k(
                            Ca, S,
                            -pz + beta * d_a_k[static_cast<std::size_t>(ik)]);
                    double sl = frob_k(g, d);
                    if (sl >= 0.0) {
                        d = -pz;
                        sl = frob_k(g, d);
                    }
                    d_a_k[static_cast<std::size_t>(ik)] = d;
                    pz_a_prev_k[static_cast<std::size_t>(ik)] = pz;
                    slope_dot += wk * sl;
                } else {
                    d_a_k[static_cast<std::size_t>(ik)] =
                        xt::zeros<Cplx>({nb, 0});
                    pz_a_prev_k[static_cast<std::size_t>(ik)] =
                        xt::zeros<Cplx>({nb, 0});
                }
                if (n_act_b > 0) {
                    const auto& Cb = Cocc_b_k[static_cast<std::size_t>(ik)];
                    const auto& g = g_b_k[static_cast<std::size_t>(ik)];
                    const auto& pz = pz_b_k[static_cast<std::size_t>(ik)];
                    CplxMatrix d;
                    if (!have_prev)
                        d = -pz;
                    else
                        d = linalg::s_tangent_k(
                            Cb, S,
                            -pz + beta * d_b_k[static_cast<std::size_t>(ik)]);
                    double sl = frob_k(g, d);
                    if (sl >= 0.0) {
                        d = -pz;
                        sl = frob_k(g, d);
                    }
                    d_b_k[static_cast<std::size_t>(ik)] = d;
                    pz_b_prev_k[static_cast<std::size_t>(ik)] = pz;
                    slope_dot += wk * sl;
                } else {
                    d_b_k[static_cast<std::size_t>(ik)] =
                        xt::zeros<Cplx>({nb, 0});
                    pz_b_prev_k[static_cast<std::size_t>(ik)] =
                        xt::zeros<Cplx>({nb, 0});
                }
            }
#ifdef ATV_HAS_MPI
            if (layout.active)
                MPI_Allreduce(MPI_IN_PLACE, &slope_dot, 1, MPI_DOUBLE, MPI_SUM,
                              MPI_COMM_WORLD);
#endif
            have_prev = true;

            // Frozen-H band parabola weighted by f_σ (integer: f=1).
            auto band_spin = [&](const CplxMatrix& H, const CplxMatrix& Cc,
                                 const std::vector<double>& f, int n_int) {
                const int no = static_cast<int>(Cc.shape(1));
                double s = 0.0;
                for (int o = 0; o < no; ++o) {
                    const double fo =
                        use_smear ? ((o < static_cast<int>(f.size()))
                                         ? f[static_cast<std::size_t>(o)]
                                         : 0.0)
                                  : ((o < n_int) ? 1.0 : 0.0);
                    if (!(fo > 1e-14))
                        continue;
                    double eorb = 0.0;
                    for (int mu = 0; mu < nb; ++mu) {
                        Cplx hcv(0, 0);
                        for (int nu = 0; nu < nb; ++nu)
                            hcv += H(mu, nu) * Cc(nu, o);
                        eorb += (std::conj(Cc(mu, o)) * hcv).real();
                    }
                    s += fo * eorb;
                }
                return s;
            };
            double Eb0 = 0.0, Ebt = 0.0;
            for (int ik = layout.rank; ik < nk; ik += layout.nprocs) {
                const double wk = weights[static_cast<std::size_t>(ik)];
                const auto& S = S_k[static_cast<std::size_t>(ik)];
                if (n_act_a > 0) {
                    const auto& Ha = Ha_k[static_cast<std::size_t>(ik)];
                    const auto& Ca = Cocc_a_k[static_cast<std::size_t>(ik)];
                    const auto& da = d_a_k[static_cast<std::size_t>(ik)];
                    const auto& fa = occ_a_k[static_cast<std::size_t>(ik)];
                    Eb0 += wk * band_spin(Ha, Ca, fa, n_a);
                    Ebt +=
                        wk
                        * band_spin(Ha, internal::ot_parameterize_k(Ca, da, S),
                                    fa, n_a);
                }
                if (n_act_b > 0) {
                    const auto& Hb = Hb_k[static_cast<std::size_t>(ik)];
                    const auto& Cb = Cocc_b_k[static_cast<std::size_t>(ik)];
                    const auto& db = d_b_k[static_cast<std::size_t>(ik)];
                    const auto& fb = occ_b_k[static_cast<std::size_t>(ik)];
                    Eb0 += wk * band_spin(Hb, Cb, fb, n_b);
                    Ebt +=
                        wk
                        * band_spin(Hb, internal::ot_parameterize_k(Cb, db, S),
                                    fb, n_b);
                }
            }
#ifdef ATV_HAS_MPI
            if (layout.active) {
                double pack[2] = {Eb0, Ebt};
                MPI_Allreduce(MPI_IN_PLACE, pack, 2, MPI_DOUBLE, MPI_SUM,
                              MPI_COMM_WORLD);
                Eb0 = pack[0];
                Ebt = pack[1];
            }
#endif
            const double curv = 2.0 * (Ebt - Eb0 - slope_dot);
            double alpha = (curv > 1e-300) ? -slope_dot / curv : 1.0;
            if (!(alpha > 0.0))
                alpha = 1.0;
            alpha = std::min(alpha, 2.0);
            constexpr double c1 = 1e-4;

            auto trial_energy = [&](double a) {
                std::vector<CplxMatrix> Pat(static_cast<std::size_t>(nk));
                std::vector<CplxMatrix> Pbt(static_cast<std::size_t>(nk));
                for (int ik = layout.rank; ik < nk; ik += layout.nprocs) {
                    const auto& S = S_k[static_cast<std::size_t>(ik)];
                    if (n_act_a > 0) {
                        CplxMatrix Cnew = internal::ot_parameterize_k(
                            Cocc_a_k[static_cast<std::size_t>(ik)],
                            a * d_a_k[static_cast<std::size_t>(ik)], S);
                        Cnew = linalg::lowdin_orthonormalize_k(Cnew, S);
                        Pat[static_cast<std::size_t>(ik)] = dens_from_cocc(
                            Cnew, n_a, occ_a_k[static_cast<std::size_t>(ik)]);
                    } else {
                        Pat[static_cast<std::size_t>(ik)] =
                            xt::zeros<Cplx>({nb, nb});
                    }
                    if (n_act_b > 0) {
                        CplxMatrix Cnew = internal::ot_parameterize_k(
                            Cocc_b_k[static_cast<std::size_t>(ik)],
                            a * d_b_k[static_cast<std::size_t>(ik)], S);
                        Cnew = linalg::lowdin_orthonormalize_k(Cnew, S);
                        Pbt[static_cast<std::size_t>(ik)] = dens_from_cocc(
                            Cnew, n_b, occ_b_k[static_cast<std::size_t>(ik)]);
                    } else {
                        Pbt[static_cast<std::size_t>(ik)] =
                            xt::zeros<Cplx>({nb, nb});
                    }
                }
                return build_uks_k_energy(Pat, Pbt);
            };

            for (int bt = 0; bt < 50; ++bt) {
                const double et = trial_energy(alpha);
                if (et <= e_total + c1 * alpha * slope_dot || bt == 49) {
                    e_total = et;
                    res.e_total = et;
                    break;
                }
                const double denom = et - e_total - slope_dot * alpha;
                const double a_q =
                    (denom > 1e-300) ? -slope_dot * alpha * alpha / (2.0 * denom)
                                     : 0.5 * alpha;
                alpha = std::min(std::max(a_q, 0.1 * alpha), 0.5 * alpha);
            }
            for (int ik = layout.rank; ik < nk; ik += layout.nprocs) {
                const auto& S = S_k[static_cast<std::size_t>(ik)];
                if (n_act_a > 0) {
                    auto& Ca = Cocc_a_k[static_cast<std::size_t>(ik)];
                    Ca = linalg::lowdin_orthonormalize_k(
                        internal::ot_parameterize_k(
                            Ca, alpha * d_a_k[static_cast<std::size_t>(ik)], S),
                        S);
                    Pa_k[static_cast<std::size_t>(ik)] = dens_from_cocc(
                        Ca, n_a, occ_a_k[static_cast<std::size_t>(ik)]);
                }
                if (n_act_b > 0) {
                    auto& Cb = Cocc_b_k[static_cast<std::size_t>(ik)];
                    Cb = linalg::lowdin_orthonormalize_k(
                        internal::ot_parameterize_k(
                            Cb, alpha * d_b_k[static_cast<std::size_t>(ik)], S),
                        S);
                    Pb_k[static_cast<std::size_t>(ik)] = dens_from_cocc(
                        Cb, n_b, occ_b_k[static_cast<std::size_t>(ik)]);
                }
            }
            h_matches_manifold = true; // next outer skips integer KS rebuild
        }
        // Budget exhausted: last_veff_ab from final trial.
        if (!last_veff_ab.empty()) {
            auto [Ha_full, Hb_full] = rebuild_all_H(last_veff_ab);
            store_k_payload(Ha_full, Hb_full);
        }
        {
            const double sz = 0.5 * static_cast<double>(n_a - n_b);
            res.s2 = sz * (sz + 1.0) + static_cast<double>(n_b);
        }
        fold_dispersion_k_uks(res);
        res.e_fermi = use_smear ? last_mu : 0.0;
        res.e_entropy = use_smear ? (-degauss_ha * last_entropy) : 0.0;
        res.n_elec_occ = use_smear ? last_n_elec : n_elec;
        return res;
    }

    for (int iter = 0; iter < cfg.max_iter; ++iter) {
        // DIIS path: same shared fine-grid KS build as build_uks_k_energy.
        const std::vector<double> rho_a = collocate_rho_spin_k(Pa_k);
        const std::vector<double> rho_b = collocate_rho_spin_k(Pb_k);
        const internal::KsGridUks ks = internal::ks_grid_build_uks(
            grid, rho_a, rho_b, n_ion, v_locsr, cfg.xc, /*nonperiodic=*/false,
            cfg.vdw_df, cfg.vdw_df2, cfg.rvv10);
        const double e_locgrid = ks.e_locgrid;
        const std::vector<double>& veff_a = ks.veff_a;
        const std::vector<double>& veff_b = ks.veff_b;
        last_veff_ab.resize(veff_a.size() + veff_b.size());
        for (std::size_t g = 0; g < veff_a.size(); ++g) {
            last_veff_ab[g] = veff_a[g];
            last_veff_ab[g + veff_a.size()] = veff_b[g];
        }
        // Force assembly reuses dual V_eff + V_es (same functional as the SCF).
        res.veff_a = veff_a;
        res.veff_b = veff_b;
        res.v_es = ks.v_es;

        std::vector<std::vector<double>> Vt_a, Vt_b;
        if (use_mg) {
            Vt_a = mg_vt_stack(veff_a);
            Vt_b = mg_vt_stack(veff_b);
        }
        double e_kin = 0.0, e_nl = 0.0;
        for (int ik = layout.rank; ik < nk; ik += layout.nprocs) {
            const double wk = weights[static_cast<std::size_t>(ik)];
            const CplxMatrix Va =
                use_mg ? mg_phase_V(Vt_a, ik)
                       : internal::integrate_potential_k(kbg, ik, veff_a);
            const CplxMatrix Vb =
                use_mg ? mg_phase_V(Vt_b, ik)
                       : internal::integrate_potential_k(kbg, ik, veff_b);
            const CplxMatrix H0 = T_k[static_cast<std::size_t>(ik)]
                                  + Vnl_k[static_cast<std::size_t>(ik)];
            Ha_k[static_cast<std::size_t>(ik)] = H0 + Va;
            Hb_k[static_cast<std::size_t>(ik)] = H0 + Vb;
            const CplxMatrix Ptot = Pa_k[static_cast<std::size_t>(ik)]
                                    + Pb_k[static_cast<std::size_t>(ik)];
            e_kin += wk * ctrace(Ptot, T_k[static_cast<std::size_t>(ik)]);
            e_nl += wk * ctrace(Ptot, Vnl_k[static_cast<std::size_t>(ik)]);
        }
        double e_dftu = 0.0;
        if (do_dftu) {
            e_dftu += internal::apply_dft_plus_u_k(
                Ha_k, Pa_k, S_k, weights, layout.rank, layout.nprocs,
                layout.active, ao_tags, z_of_atom, plus_u_slots, plus_u_z,
                &Va_dftu);
            e_dftu += internal::apply_dft_plus_u_k(
                Hb_k, Pb_k, S_k, weights, layout.rank, layout.nprocs,
                layout.active, ao_tags, z_of_atom, plus_u_slots, plus_u_z,
                &Vb_dftu);
            have_V_dftu = true;
        }
        res.e_dftu = e_dftu;
#ifdef ATV_HAS_MPI
        if (layout.active) {
            MPI_Allreduce(MPI_IN_PLACE, &e_kin, 1, MPI_DOUBLE, MPI_SUM,
                          MPI_COMM_WORLD);
            MPI_Allreduce(MPI_IN_PLACE, &e_nl, 1, MPI_DOUBLE, MPI_SUM,
                          MPI_COMM_WORLD);
        }
#endif
        res.e_kinetic = e_kin;
        res.e_nonlocal = 0.0;
        res.e_loc_sr = e_nl + e_locgrid;
        res.e_es = ks.e_hartree - e_self;
        res.e_xc = ks.e_xc;
        // Mermin free energy when smearing: F = E_KS − σ S (+ DFT+U if on).
        const double e_ks =
            res.e_kinetic + res.e_loc_sr + res.e_es + res.e_xc + e_dftu;
        const double e_ent = use_smear ? (-degauss_ha * last_entropy) : 0.0;
        const double e_total = e_ks + e_ent;
        res.iters = iter + 1;
        res.e_total = e_total;
        res.e_entropy = e_ent;
        res.e_fermi = use_smear ? last_mu : 0.0;
        res.n_elec_occ = use_smear ? last_n_elec : n_elec;

        std::vector<CplxMatrix> err_a(static_cast<std::size_t>(nk));
        std::vector<CplxMatrix> err_b(static_cast<std::size_t>(nk));
        double err_norm = 0.0;
        for (int ik = layout.rank; ik < nk; ik += layout.nprocs) {
            const CplxMatrix ea =
                internal::k_diis_error(Ha_k[static_cast<std::size_t>(ik)],
                                       Pa_k[static_cast<std::size_t>(ik)],
                                       S_k[static_cast<std::size_t>(ik)],
                                       orth[static_cast<std::size_t>(ik)].X);
            const CplxMatrix eb =
                internal::k_diis_error(Hb_k[static_cast<std::size_t>(ik)],
                                       Pb_k[static_cast<std::size_t>(ik)],
                                       S_k[static_cast<std::size_t>(ik)],
                                       orth[static_cast<std::size_t>(ik)].X);
            for (const Cplx v : ea)
                err_norm = std::max(err_norm, std::abs(v));
            for (const Cplx v : eb)
                err_norm = std::max(err_norm, std::abs(v));
            err_a[static_cast<std::size_t>(ik)] = ea;
            err_b[static_cast<std::size_t>(ik)] = eb;
        }
#ifdef ATV_HAS_MPI
        if (layout.active) {
            double global_err = err_norm;
            MPI_Allreduce(&err_norm, &global_err, 1, MPI_DOUBLE, MPI_MAX,
                          MPI_COMM_WORLD);
            err_norm = global_err;
        }
#endif

        if (std::getenv("ATV_GPW_TRACE") && layout.rank == 0) {
            if (use_smear)
                std::fprintf(stderr,
                             "[gpw k-UKS %2d] F=%.8f  T=%.4f loc=%.4f es=%.4f "
                             "xc=%.4f U=%.4f  dE=%.2e err=%.2e mu=%.6f Ne=%.6f "
                             "(nk=%d Na_seed=%d Nb_seed=%d)\n",
                             iter, e_total, res.e_kinetic, res.e_loc_sr,
                             res.e_es, res.e_xc, e_dftu,
                             std::abs(e_total - e_prev), err_norm, last_mu,
                             last_n_elec, nk, n_a, n_b);
            else
                std::fprintf(
                    stderr,
                    "[gpw k-UKS %2d] E=%.8f  T=%.4f loc=%.4f es=%.4f "
                    "xc=%.4f U=%.4f  dE=%.2e err=%.2e (nk=%d Na=%d Nb=%d)\n",
                    iter, e_total, res.e_kinetic, res.e_loc_sr, res.e_es,
                    res.e_xc, e_dftu, std::abs(e_total - e_prev), err_norm, nk,
                    n_a, n_b);
        }

        const double err_gate = do_dftu ? 5e-3 : kDiisErrTolK;
        if (iter > 0 && std::abs(e_total - e_prev) < cfg.tol
            && err_norm < err_gate) {
            res.converged = true;
            auto [Ha_full, Hb_full] = rebuild_all_H(last_veff_ab);
            store_k_payload(Ha_full, Hb_full);
            // Fixed occupations ⇒ integer M = Nα−Nβ; smear ⇒ free M (seed only).
            {
                const double sz = 0.5 * static_cast<double>(n_a - n_b);
                res.s2 = sz * (sz + 1.0) + static_cast<double>(n_b);
            }
            fold_dispersion_k_uks(res);
            res.e_fermi = use_smear ? last_mu : 0.0;
            res.e_entropy = use_smear ? (-degauss_ha * last_entropy) : 0.0;
            res.n_elec_occ = use_smear ? last_n_elec : n_elec;
            return res;
        }
        e_prev = e_total;

        if (use_diis) {
            Ha_k = diis_a.step(Ha_k, err_a);
            Hb_k = diis_b.step(Hb_k, err_b);
        }

        // Diagonalize owned k; for metallic smear, gather full α+β spectrum
        // and rebuild Pa/Pb with fractional occupations at a shared μ.
        std::vector<std::vector<double>> evals_a(static_cast<std::size_t>(nk));
        std::vector<std::vector<double>> evals_b(static_cast<std::size_t>(nk));
        std::vector<CplxMatrix> Ca_local(static_cast<std::size_t>(nk));
        std::vector<CplxMatrix> Cb_local(static_cast<std::size_t>(nk));
        for (int ik = layout.rank; ik < nk; ik += layout.nprocs) {
            auto [epsa, Ca] =
                internal::diagonalize_k(Ha_k[static_cast<std::size_t>(ik)],
                                        orth[static_cast<std::size_t>(ik)]);
            auto [epsb, Cb] =
                internal::diagonalize_k(Hb_k[static_cast<std::size_t>(ik)],
                                        orth[static_cast<std::size_t>(ik)]);
            evals_a[static_cast<std::size_t>(ik)] = std::move(epsa);
            evals_b[static_cast<std::size_t>(ik)] = std::move(epsb);
            Ca_local[static_cast<std::size_t>(ik)] = std::move(Ca);
            Cb_local[static_cast<std::size_t>(ik)] = std::move(Cb);
        }
#ifdef ATV_HAS_MPI
        if (layout.active && use_smear) {
            auto allreduce_evals =
                [&](std::vector<std::vector<double>>& evals) {
                    std::vector<double> pack(static_cast<std::size_t>(nk * nb),
                                             0.0);
                    std::vector<int> nband(static_cast<std::size_t>(nk), 0);
                    for (int ik = layout.rank; ik < nk; ik += layout.nprocs) {
                        const auto& ev = evals[static_cast<std::size_t>(ik)];
                        nband[static_cast<std::size_t>(ik)] =
                            static_cast<int>(ev.size());
                        for (std::size_t b = 0; b < ev.size(); ++b)
                            pack[static_cast<std::size_t>(ik * nb) + b] = ev[b];
                    }
                    MPI_Allreduce(MPI_IN_PLACE, pack.data(), nk * nb,
                                  MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
                    MPI_Allreduce(MPI_IN_PLACE, nband.data(), nk, MPI_INT,
                                  MPI_SUM, MPI_COMM_WORLD);
                    for (int ik = 0; ik < nk; ++ik) {
                        const int m = nband[static_cast<std::size_t>(ik)];
                        evals[static_cast<std::size_t>(ik)].resize(
                            static_cast<std::size_t>(m));
                        for (int b = 0; b < m; ++b)
                            evals[static_cast<std::size_t>(
                                ik)][static_cast<std::size_t>(b)] =
                                pack[static_cast<std::size_t>(ik * nb + b)];
                    }
                };
            allreduce_evals(evals_a);
            allreduce_evals(evals_b);
        }
#endif
        auto mix_density = [&](int ik, CplxMatrix Pa_new, CplxMatrix Pb_new) {
            if (use_diis) {
                if (do_dftu) {
                    const double a = std::min(std::max(cfg.mix, 0.15), 0.6);
                    Pa_k[static_cast<std::size_t>(ik)] =
                        (1.0 - a) * Pa_k[static_cast<std::size_t>(ik)]
                        + a * Pa_new;
                    Pb_k[static_cast<std::size_t>(ik)] =
                        (1.0 - a) * Pb_k[static_cast<std::size_t>(ik)]
                        + a * Pb_new;
                } else {
                    Pa_k[static_cast<std::size_t>(ik)] = std::move(Pa_new);
                    Pb_k[static_cast<std::size_t>(ik)] = std::move(Pb_new);
                }
            } else {
                Pa_k[static_cast<std::size_t>(ik)] =
                    (1.0 - cfg.mix) * Pa_k[static_cast<std::size_t>(ik)]
                    + cfg.mix * Pa_new;
                Pb_k[static_cast<std::size_t>(ik)] =
                    (1.0 - cfg.mix) * Pb_k[static_cast<std::size_t>(ik)]
                    + cfg.mix * Pb_new;
            }
        };
        if (use_smear) {
            auto [occ_a, occ_b] = fill_metallic_uks(evals_a, evals_b, last_mu,
                                                    last_n_elec, last_entropy);
            for (int ik = layout.rank; ik < nk; ik += layout.nprocs) {
                mix_density(
                    ik,
                    density_from_occ(Ca_local[static_cast<std::size_t>(ik)],
                                     occ_a[static_cast<std::size_t>(ik)]),
                    density_from_occ(Cb_local[static_cast<std::size_t>(ik)],
                                     occ_b[static_cast<std::size_t>(ik)]));
            }
        } else {
            for (int ik = layout.rank; ik < nk; ik += layout.nprocs) {
                mix_density(
                    ik,
                    density_spin(Ca_local[static_cast<std::size_t>(ik)], n_a),
                    density_spin(Cb_local[static_cast<std::size_t>(ik)], n_b));
            }
        }
    }
    // Budget exhausted: report the last state (converged stays false).
    if (!last_veff_ab.empty()) {
        auto [Ha_full, Hb_full] = rebuild_all_H(last_veff_ab);
        store_k_payload(Ha_full, Hb_full);
    }
    {
        const double sz = 0.5 * static_cast<double>(n_a - n_b);
        res.s2 = sz * (sz + 1.0) + static_cast<double>(n_b);
    }
    fold_dispersion_k_uks(res);
    res.e_fermi = use_smear ? last_mu : 0.0;
    res.e_entropy = use_smear ? (-degauss_ha * last_entropy) : 0.0;
    res.n_elec_occ = use_smear ? last_n_elec : n_elec;
    return res;
}

/// The full GPW SCF implementation, owned and reached through
/// `GpwDft::run`.
static GpwResult
run_gpw_scf_impl(const std::vector<GpwAtom>& atoms, const GpwDft& cfg,
                 const RealMatrix* guess_C, const RealMatrix* guess_P,
                 const RealMatrix* guess_P_beta,
                 const std::vector<linalg::CplxMatrix>* guess_P_k,
                 const std::vector<linalg::CplxMatrix>* guess_P_k_beta) {
#ifdef ATV_GPW_FFTW_THREADS
    // One-time global FFTW threading init; fftw_plan_with_nthreads is a global
    // setting, so every FFT plan created downstream (Poisson, GGA grad/div, MT)
    // runs multithreaded. The density-grid FFTs dominate GPW's wall time.
    static const int kFftwThreadsOk = fftw_init_threads();
    if (kFftwThreadsOk)
        fftw_plan_with_nthreads(omp_get_max_threads());
#endif
    const bool periodic = !cfg.nonperiodic;
    const double eps_default = cfg.eps_default;
    // Flattened basis, Gaussian-tail radii, and the Γ pair-image extent —
    // constructed through the same type `GpwDft::overlap` / `overlap_k` use, so
    // the standalone overlaps sum exactly the images the SCF's S does.
    const OneElectronSetup setup(atoms, cfg.cell, periodic, eps_default);
    const std::vector<GridBasisFunction>& basis = setup.basis;
    const std::vector<double>& radii = setup.radii;
    const int* const mimg = setup.mimg;
    const int nb = setup.nb();
    const std::vector<internal::AoShellTag> ao_tags = global_ao_tags(atoms);
    if (static_cast<int>(ao_tags.size()) != nb)
        throw std::runtime_error(
            "GPW: AO tag count does not match basis size (internal)");
    std::vector<int> z_of_atom(atoms.size());
    for (std::size_t i = 0; i < atoms.size(); ++i)
        z_of_atom[i] = atoms[i].z;
    std::vector<int> plus_u_z;
    std::vector<internal::PlusUHa> plus_u_slots;
    pack_plus_u(cfg.plus_u, plus_u_z, plus_u_slots);
    const bool do_dftu = !plus_u_slots.empty();
    // Solver-private workspace for this run (Γ path owns bags; k runners
    // build their own). Grid is filled here so Γ aliases and k call sites
    // share one make_uniform_grid.
    internal::GpwWorkspace ws;
    ws.grid = make_uniform_grid(cfg.cell, cfg.cutoff_ry);
    const UniformGrid& grid = ws.grid;

    // k-point RKS branch selector.
    // (Configuration conflicts are rejected fail-fast in GpwDft::build.)
    const bool use_kpath = uses_kpath(cfg);

    // Dispersion — free-atom TS/MBD/XDM/D3/D4 are geometry-only (pre-SCF).
    // Hirshfeld-scaled TS/MBD and density-dependent vdw_df / vdw_df2 / rvv10
    // are deferred to post-SCF (`fold_dispersion` / k-path density fold) for
    // RKS and UKS (total density ρ=ρα+ρβ). `d3`, `d3bj`, `d4`, `ts`, `mbd`,
    // `xdm`, `vdw_df`, `vdw_df2`, and `rvv10` are pairwise mutually exclusive.
    double e_d3 = 0.0;
    const int n_disp_flags =
        (cfg.d3 ? 1 : 0) + (cfg.d3bj ? 1 : 0) + (cfg.d4 ? 1 : 0)
        + (cfg.ts ? 1 : 0) + (cfg.xdm ? 1 : 0) + (cfg.mbd ? 1 : 0)
        + (cfg.vdw_df ? 1 : 0) + (cfg.vdw_df2 ? 1 : 0) + (cfg.rvv10 ? 1 : 0);
    if (n_disp_flags > 1)
        throw std::runtime_error(
            "GPW: d3, d3bj, d4, ts, xdm, mbd, vdw_df, vdw_df2, and rvv10 are "
            "mutually exclusive — set only one");
    if (cfg.hirshfeld) {
        if (!(cfg.ts || cfg.mbd))
            throw std::runtime_error(
                "GPW: hirshfeld requires ts or mbd (no standalone Hirshfeld)");
        if (cfg.d3 || cfg.d3bj || cfg.d4 || cfg.xdm || cfg.vdw_df || cfg.vdw_df2
            || cfg.rvv10)
            throw std::runtime_error(
                "GPW: hirshfeld applies only to ts/mbd, not "
                "d3/d3bj/d4/xdm/vdw_df/vdw_df2/rvv10");
        // RKS and UKS (Γ + multi-k): volumes from total valence density.
    }
    if (cfg.vdw_df || cfg.vdw_df2 || cfg.rvv10) {
        const char* tag = cfg.rvv10     ? "rvv10"
                          : cfg.vdw_df2 ? "vdw_df2"
                                        : "vdw_df";
        if (!periodic)
            throw std::runtime_error(
                std::string("GPW ") + tag
                + ": periodic cell required (RPS nonlocal kernel)");
        if (cfg.nonperiodic)
            throw std::runtime_error(
                std::string("GPW ") + tag
                + ": incompatible with nonperiodic (MT) Poisson");
        // Energy is evaluated post-SCF on the final density (below).
    } else if (cfg.d3 || cfg.d3bj || cfg.d4 || cfg.ts || cfg.xdm || cfg.mbd) {
        std::vector<int> Z(atoms.size());
        std::vector<std::array<double, 3>> pos(Z.size());
        for (std::size_t i = 0; i < Z.size(); ++i) {
            Z[i] = atoms[i].z;
            pos[i] = {atoms[i].pos[0], atoms[i].pos[1], atoms[i].pos[2]};
        }
        if (cfg.mbd) {
            // MBD@rsSCS: free-atom isolated when mbd_k_mesh is all-zero;
            // periodic Ewald + k-mesh when any component is set.
            // Deferred when hirshfeld=true (post-SCF volume scaling).
            if (!cfg.hirshfeld) {
                const bool mbd_per = cfg.mbd_k_mesh[0] > 0
                                     || cfg.mbd_k_mesh[1] > 0
                                     || cfg.mbd_k_mesh[2] > 0;
                if (mbd_per && cfg.nonperiodic)
                    throw std::runtime_error(
                        "GPW MBD: mbd_k_mesh requires a periodic cell "
                        "(!nonperiodic)");
                e_d3 = mbd_rsscs_pbe(Z, pos, cfg.cell, mbd_per, 0.83, 6.0, {},
                                     mbd_per ? cfg.mbd_k_mesh : nullptr)
                           .energy;
            }
        } else if (cfg.xdm) {
            constexpr double kAng2Bohr = 1.0 / 0.52917720859;
            constexpr double kRcutBohr = 50.0 * kAng2Bohr;
            e_d3 = xdm_pbe(Z, pos, cfg.cell, periodic, kRcutBohr).energy;
        } else if (cfg.ts) {
            // Free-atom TS (volume ratio = 1). Deferred when hirshfeld=true.
            if (!cfg.hirshfeld) {
                constexpr double kAng2Bohr = 1.0 / 0.52917720859;
                constexpr double kRcutBohr = 50.0 * kAng2Bohr;
                e_d3 = ts_pbe(Z, pos, cfg.cell, periodic, kRcutBohr).energy;
            }
        } else if (cfg.d4) {
            const double s9 = cfg.d4_atm ? 1.0 : 0.0;
            e_d3 = cpu::scf::dftd4_energy(Z, pos, cfg.cell, periodic,
                                          /*s6=*/1.0, /*s8=*/0.95948085,
                                          /*a1=*/0.38574991, /*a2=*/4.80688534,
                                          /*ga=*/3.0, /*gc=*/2.0, /*wf=*/6.0,
                                          /*charge=*/0.0, /*cn_cut=*/30.0,
                                          /*disp2_cut=*/60.0, s9);
        } else {
            // In-tree D3 tables only (basis_data/dftd3_tables.h).
            for (int z : Z)
                if (!D3Params::supports_element(z))
                    throw std::runtime_error(
                        "GPW D3: element Z=" + std::to_string(z)
                        + " is not in the in-tree D3 table (no fallback)");
            // Both damping variants use the one configured 2 × R_CUTOFF pair
            // cutoff (cfg.d3_rcut_bohr, CP2K doubled convention) so energy,
            // force and stress all differentiate the same functional.
            if (cfg.d3)
                e_d3 = dftd3_pbe(Z, pos, cfg.cell, periodic, D3Params{},
                                 cfg.d3_rcut_bohr, cfg.d3_atm)
                           .energy;
            else
                e_d3 = dftd3_pbe_bj(Z, pos, cfg.cell, periodic, D3Params{},
                                    cfg.d3_rcut_bohr, 1.0, 0.7875, 0.4289,
                                    4.4407, /*cp2k_cn=*/true, cfg.d3_atm)
                           .energy;
        }
    }

    // Geometry-fixed matrices.
    RealMatrix S = xt::zeros<double>({nb, nb});
    RealMatrix T = xt::zeros<double>({nb, nb});
    RealMatrix Vpp = xt::zeros<double>({nb, nb}); // V_loc^sr + V_nl
    // The geometry-fixed one-electron / GTH matrices dominate the GPW SCF wall
    // time at scale (profiled: ~75-80% for hundreds of atoms; the eigensolve is
    // <2%). Each (i,j) is an independent primitive contraction; the assembly
    // itself lives in OneElectronSetup::gamma (shared with GpwDft::overlap).
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
    // Local short-range PP: by default collocated on the grid (local_pp_grid +
    // V_eff, O(grid)); local_pp_on_grid=false selects the per-pair analytic
    // gth_local_sr (O(nb²·natoms)), kept for grid-vs-analytic energy validation.
    // gpw_forces_analytic reads the SAME flag — the gradient must be taken of the
    // functional the SCF actually converged, or the two disagree by the grid
    // discretization error of V_loc^SR (see internal/local_pp.h).
    const bool grid_loc = cfg.local_pp_on_grid;
    // The Γ real S/T/Vpp lattice sums serve the Γ paths only; the k-point
    // branch builds its own phased (complex) matrices per k.
    if (!use_kpath)
        setup.gamma(atoms, cfg.cell, periodic, grid_loc, S, &T, &Vpp);
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

    // ── k-point RKS / collinear UKS branch ─────────────────────────────────
    if (use_kpath) {
        // Real Γ density guess has no meaning on the complex k path — refuse
        // rather than silently cold-start (would mask a broken multi-k XL).
        if (guess_P != nullptr || guess_P_beta != nullptr)
            throw std::invalid_argument(
                "GpwDft::run: multi-k SCF takes complex guess_P_k (one "
                "density per k); real guess_P / guess_P_beta are Γ-only");
        const std::vector<internal::KPoint> kpts = sampled_kpoints(cfg);
        if (cfg.is_uks()) {
            // Dual-spin multi-k XL: both channels or neither (partial
            // validated inside run_gpw_scf_kpoints_uks).
            return run_gpw_scf_kpoints_uks(atoms, cfg, basis, radii, mimg, grid,
                                           n_ion, v_locsr, e_self, e_d3, kpts,
                                           guess_P_k, guess_P_k_beta);
        }
        if (guess_P_k_beta != nullptr)
            throw std::invalid_argument(
                "GpwDft::run: guess_P_k_beta is multi-k UKS-only "
                "(closed-shell multi-k RKS takes total density on guess_P_k)");
        return run_gpw_scf_kpoints(atoms, cfg, basis, radii, mimg, grid, n_ion,
                                   v_locsr, e_self, e_d3, kpts, guess_P_k);
    }
    if (guess_P_k != nullptr || guess_P_k_beta != nullptr)
        throw std::invalid_argument(
            "GpwDft::run: guess_P_k / guess_P_k_beta are multi-k only "
            "(Γ path takes real guess_P / guess_P_beta)");

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
            const int n_need =
                cfg.is_uks() ? std::max(cfg.n_alpha, cfg.n_beta) : cfg.n_occ;
            if (n_need > m)
                throw std::runtime_error(
                    "GPW: occupation=" + std::to_string(n_need)
                    + " exceeds the " + std::to_string(m)
                    + " linearly independent basis directions that survive "
                      "lindep_threshold — the basis cannot represent the "
                      "occupied "
                      "manifold (no-fallback rule)");
        }
        return *orth_cache;
    };

    const bool uks = cfg.is_uks();

    // cold start falls back to diagonalizing the core Hamiltonian H0.
    RealMatrix C0;
    if (guess_C != nullptr && static_cast<int>(guess_C->shape(0)) == nb
        && static_cast<int>(guess_C->shape(1)) == nb) {
        C0 = *guess_C;
    } else {
        C0 = std::get<1>(linalg::diagonalize_fock(H0, orth().X));
    }
    RealMatrix P;
    if (uks) {
        P = linalg::build_density_matrix(C0, cfg.n_alpha, nb, 1.0)
            + linalg::build_density_matrix(C0, cfg.n_beta, nb, 1.0);
    } else {
        P = linalg::build_density_matrix(C0, cfg.n_occ, nb, 2.0);
    }
    RealMatrix C_prev = C0; // level-shift needs the previous occupied space
    RealMatrix Ca_prev = C0;
    RealMatrix Cb_prev = C0;
    RealMatrix Pa = uks ? linalg::build_density_matrix(C0, cfg.n_alpha, nb, 1.0)
                        : RealMatrix{};
    RealMatrix Pb = uks ? linalg::build_density_matrix(C0, cfg.n_beta, nb, 1.0)
                        : RealMatrix{};
    // AFM/FM cold-start bias: scale per-atom spin density on the diagonal of
    // Pa/Pb using atom_spin_guess (units-free). Does not alter Tr(P) much when
    // guesses cancel; breaks spin symmetry for AFM NiO.
    if (uks && !cfg.atom_spin_guess.empty()) {
        if (static_cast<int>(cfg.atom_spin_guess.size())
            != static_cast<int>(atoms.size()))
            throw std::invalid_argument(
                "GpwDft: atom_spin_guess size must be 0 or n_atoms");
        constexpr double kBias = 0.15; // fraction of local diagonal shift
        for (int i = 0; i < nb; ++i) {
            const int ia = ao_tags[static_cast<std::size_t>(i)].atom;
            const double g = cfg.atom_spin_guess[static_cast<std::size_t>(ia)];
            if (g == 0.0)
                continue;
            const double d = kBias * g * 0.5 * (Pa(i, i) + Pb(i, i));
            Pa(i, i) += d;
            Pb(i, i) -= d;
            if (Pa(i, i) < 0.0)
                Pa(i, i) = 0.0;
            if (Pb(i, i) < 0.0)
                Pb(i, i) = 0.0;
        }
    }
    // XL-BOMD / SGCP ASPC warm density start (UKS partial-guess guards live
    // in GpwDft::run): the propagated / predicted density replaces the
    // C0-derived initial P (RKS) or Pa/Pb (UKS) for the first Fock build.
    // C_prev (level shift) still comes from C0 on the DIIS path.
    if (guess_P != nullptr) {
        if (static_cast<int>(guess_P->shape(0)) != nb
            || static_cast<int>(guess_P->shape(1)) != nb)
            throw std::invalid_argument("GpwDft::run: guess_P shape ("
                                        + std::to_string(guess_P->shape(0))
                                        + "x"
                                        + std::to_string(guess_P->shape(1))
                                        + ") does not match the AO basis (nb="
                                        + std::to_string(nb) + ")");
        if (uks) {
            if (guess_P_beta == nullptr)
                throw std::invalid_argument(
                    "GpwDft::run: UKS density guess requires both guess_P "
                    "(alpha) and guess_P_beta");
            if (static_cast<int>(guess_P_beta->shape(0)) != nb
                || static_cast<int>(guess_P_beta->shape(1)) != nb)
                throw std::invalid_argument(
                    "GpwDft::run: guess_P_beta shape ("
                    + std::to_string(guess_P_beta->shape(0)) + "x"
                    + std::to_string(guess_P_beta->shape(1))
                    + ") does not match the AO basis (nb=" + std::to_string(nb)
                    + ")");
            Pa = *guess_P;
            Pb = *guess_P_beta;
            P = Pa + Pb;
        } else {
            P = *guess_P;
        }
        // OT: purify ASPC/XL density to natural orbitals so the OT manifold
        // starts on the predicted density (not a silent ignore of guess_P,
        // and not the previous-step C alone). Eigenvalues of Xᵀ P X ascend;
        // largest occupations occupy the last columns.
        // RKS: total P → C0. UKS: dual Pα/Pβ → Ca_prev / Cb_prev (UKS OT
        // seeds from those; second-gen-cpmd-uks-ot).
        if (cfg.use_ot && !uks) {
            if (cfg.n_occ < 1)
                throw std::invalid_argument("GpwDft::run: RKS OT density "
                                            "warm-start requires n_occ >= 1");
            auto [n_occ_vals, Cnat] = linalg::diagonalize_fock(P, orth().X);
            (void)n_occ_vals;
            C0 = xt::zeros<double>(
                {static_cast<std::size_t>(nb), static_cast<std::size_t>(nb)});
            for (int i = 0; i < cfg.n_occ; ++i) {
                const int src = nb - cfg.n_occ + i;
                for (int u = 0; u < nb; ++u)
                    C0(u, i) = Cnat(u, src);
            }
            C_prev = C0;
        } else if (cfg.use_ot && uks) {
            if (cfg.n_alpha < 1 && cfg.n_beta < 1)
                throw std::invalid_argument(
                    "GpwDft::run: UKS OT density warm-start requires "
                    "n_alpha+n_beta >= 1");
            auto [na_vals, Cnat_a] = linalg::diagonalize_fock(Pa, orth().X);
            auto [nb_vals, Cnat_b] = linalg::diagonalize_fock(Pb, orth().X);
            (void)na_vals;
            (void)nb_vals;
            Ca_prev = xt::zeros<double>(
                {static_cast<std::size_t>(nb), static_cast<std::size_t>(nb)});
            Cb_prev = xt::zeros<double>(
                {static_cast<std::size_t>(nb), static_cast<std::size_t>(nb)});
            for (int i = 0; i < cfg.n_alpha; ++i) {
                const int src = nb - cfg.n_alpha + i;
                for (int u = 0; u < nb; ++u)
                    Ca_prev(u, i) = Cnat_a(u, src);
            }
            for (int i = 0; i < cfg.n_beta; ++i) {
                const int src = nb - cfg.n_beta + i;
                for (int u = 0; u < nb; ++u)
                    Cb_prev(u, i) = Cnat_b(u, src);
            }
            // Keep C0 alpha-aligned for any code that still peeks C0[:,0:na].
            C0 = Ca_prev;
            C_prev = Ca_prev;
        }
    }
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
    if (!use_prod)
        ws.cbg =
            build_compact_basis_grid(grid, basis, periodic, eps_default);
    const CompactBasisGrid& cbg = ws.cbg;
    smark("cbg_build");

    // CP2K-style REL_CUTOFF multigrid (gated on ngrids>1 && rel_cutoff>0). Each
    // primitive-pair product of the density is assigned (by its combined exponent)
    // to the coarsest grid level that resolves it; diffuse products collocate on a
    // coarse grid (far fewer points). The per-level densities are prolonged to the
    // finest grid for the (nonlinear) Hartree+XC, and the fine potential is
    // restricted back to each level to integrate the KS matrix. ngrids==1 leaves
    // the single-grid path above untouched (bit-identical). The product list is
    // geometry-fixed (excludes P); P is applied per SCF iteration.
    MultiGrid& mg = ws.mg;
    ProductSet& products = ws.products;
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

    // Total density on the finest grid from a flattened P. Single-grid:
    // direct compact (cbg) collocation. Multigrid: collocate each level's
    // products, then spectral (band-limited) prolongation to the fine grid —
    // shape-exact and mass-conserving (no D_l correction). Shared by the SCF
    // KS build, fold_dispersion, and the final nonlocal-vdW evaluation.
    auto collocate_rho_fine = [&](const std::vector<double>& Pvec) {
        if (!use_prod)
            return collocate_density(cbg, nb, Pvec);
        std::vector<double> rho(static_cast<std::size_t>(mg.levels[0].npts()),
                                0.0);
        for (std::size_t l = 0; l < mg.levels.size(); ++l) {
            std::vector<double> rho_l = collocate_products(
                mg.levels[l], products.by_level[l], products.prims,
                products.contribs, nb, Pvec, eps_default);
            if (l == 0) {
                rho = std::move(rho_l); // finest level is the reference grid
            } else {
                const std::vector<double> up = prolong_fft(
                    mg.levels[l], mg.levels[0], rho_l, /*spectral*/ true);
                for (std::size_t g = 0; g < rho.size(); ++g)
                    rho[g] += up[g];
            }
        }
        return rho;
    };

    // Fold free-atom e_d3, or Hirshfeld-scaled TS/MBD from the final density.
    auto fold_dispersion = [&](GpwResult& out, const RealMatrix& Pmat) {
        if (cfg.hirshfeld && (cfg.ts || cfg.mbd)) {
            const std::vector<double> Pvec = mat_to_vec(Pmat, nb);
            std::vector<double> rho = collocate_rho_fine(Pvec);
            std::vector<int> Z(atoms.size());
            std::vector<std::array<double, 3>> pos(Z.size());
            for (std::size_t i = 0; i < Z.size(); ++i) {
                Z[i] = atoms[i].z;
                pos[i] = {atoms[i].pos[0], atoms[i].pos[1], atoms[i].pos[2]};
            }
            out.volume_ratios =
                hirshfeld_volume_ratios(grid, rho, Z, pos, cfg.cell, periodic);
            if (cfg.mbd) {
                const bool mbd_per = cfg.mbd_k_mesh[0] > 0
                                     || cfg.mbd_k_mesh[1] > 0
                                     || cfg.mbd_k_mesh[2] > 0;
                if (mbd_per && cfg.nonperiodic)
                    throw std::runtime_error(
                        "GPW MBD: mbd_k_mesh requires a periodic cell "
                        "(!nonperiodic)");
                out.e_disp = mbd_rsscs_pbe(Z, pos, cfg.cell, mbd_per, 0.83, 6.0,
                                           out.volume_ratios,
                                           mbd_per ? cfg.mbd_k_mesh : nullptr)
                                 .energy;
            } else {
                constexpr double kAng2Bohr = 1.0 / 0.52917720859;
                constexpr double kRcutBohr = 50.0 * kAng2Bohr;
                out.e_disp = ts_pbe(Z, pos, cfg.cell, periodic, kRcutBohr, 0.94,
                                    20.0, out.volume_ratios)
                                 .energy;
            }
            out.e_total += out.e_disp;
            return;
        }
        out.e_disp = e_d3;
        out.e_total += e_d3;
    };

    GpwResult res;
    // Force path reuses geometry-fixed Γ cbg (single-grid only; MG has no cbg).
    if (!use_prod) {
        res.scf_cbg = ws.cbg;
        res.scf_cbg_valid = true;
    }
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
    // Matched to multi-k kOtGradTolK (1e-5): Γ OT often plateaus ~2.7e-6, so
    // the old 1e-6 floor caused thrashing against a non-zero residual floor.
    // DFT+U softens the residual (same 5e-3 class as DIIS+U): V_U can leave a
    // small projected-gradient floor while the total energy is fully stable.
    constexpr double kOtGradTol = 1e-5;
    constexpr double kOtGradTolDftu = 1e-2;
    const double ot_grad_tol = do_dftu ? kOtGradTolDftu : kOtGradTol;

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
        // Density on the finest grid (products + prolongation under
        // multigrid, cbg gather otherwise).
        std::vector<double> rho = collocate_rho_fine(Pvec);
        auto t1p = ProfClk::now();
        if (use_mg && std::getenv("ATV_GPW_MG_TRACE")) {
            double qe = 0.0;
            for (double v : rho)
                qe += v;
            qe *= grid.weight();
            std::fprintf(stderr,
                         "[gpw mg] int rho_fine = %.10f (Tr(P S) check)\n", qe);
        }
        const internal::KsGridRks ks = internal::ks_grid_build_rks(
            grid, rho, n_ion, v_locsr, cfg.xc, cfg.nonperiodic, cfg.vdw_df,
            cfg.vdw_df2, cfg.rvv10);
        auto t3p = ProfClk::now();
        auto t2p = t1p;
        (void)t2p;
        if (cfg.vdw_df || cfg.vdw_df2 || cfg.rvv10)
            e_d3 = ks.e_nl_vdw; // fold_dispersion reports SC E_c^nl
        // Force path reuses this V_eff / V_es (same functional as the SCF).
        res.veff = ks.veff;
        res.v_es = ks.v_es;
        const std::vector<double>& veff = res.veff;
        const double e_locgrid = ks.e_locgrid;
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
                                       /*spectral*/ true);
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
        res.e_es = ks.e_hartree - e_self;
        res.e_xc = ks.e_xc;
        return {H, res.e_kinetic + res.e_loc_sr + res.e_es + res.e_xc};
    };

    // ── Fixed-density KS evaluation (Car–Parrinello electronic step) ─────
    // max_iter == 0: one collocate→Poisson→XC→integrate at the supplied
    // density; no diagonalization, no mixing, no orbital update. Populates
    // H_ks (+ H_ks_beta for UKS) and S for the electronic EOM. Γ RKS only
    // here — multi-k fixed-density lives in run_gpw_scf_kpoints; UKS
    // fixed-density is inside the UKS block below (reuses build_uks_H).
    if (cfg.max_iter == 0 && !uks) {
        if (use_kpath)
            throw std::invalid_argument(
                "GpwDft: multi-k fixed-density must be reached via the "
                "k-path runner (internal error: Γ RKS block saw use_kpath)");
        if (guess_P == nullptr)
            throw std::invalid_argument(
                "GpwDft: max_iter==0 (fixed-density eval) requires guess_P "
                "(the Car–Parrinello density)");
        if (cfg.use_ot)
            throw std::invalid_argument(
                "GpwDft: fixed-density eval is incompatible with use_ot");
        // P was already replaced by guess_P above when non-null.
        auto [H, e_total] = build_H_and_energy(P);
        res.e_total = e_total;
        res.P = P;
        res.H_ks = std::move(H);
        res.S = S;
        res.converged = true;
        res.iters = 0;
        fold_dispersion(res, P);
        return res;
    }

    // ── UKS (spin-unrestricted, Γ-point) ────────────────────────────────────
    // Separate Pa/Pb, spin-resolved XC (PADE LSD or pure GGA spin),
    // dual KS matrices. Multigrid product collocation (ngrids>1 && rel_cutoff>0)
    // uses the same product set / prolong_fft / restrict_fft path as RKS;
    // ngrids==1 keeps the CompactBasisGrid single-grid path bit-identical.
    // OT (use_ot) is product for Γ-UKS; DIIS/diagonalization remains default.
    if (uks) {
        if (cfg.xc != XcKind::PADE && cfg.xc != XcKind::PBE
            && cfg.xc != XcKind::PBEsol && cfg.xc != XcKind::RevPBE
            && cfg.xc != XcKind::RPBE && cfg.xc != XcKind::BLYP
            && cfg.xc != XcKind::OptB88VdW)
            throw std::runtime_error(
                "GPW UKS: XcKind must be PADE, PBE, PBEsol, RevPBE, RPBE, "
                "BLYP, or OptB88VdW");

        // Collocate one spin density: multigrid products or single-grid cbg.
        auto collocate_P = [&](const RealMatrix& Pin) {
            return collocate_rho_fine(mat_to_vec(Pin, nb));
        };
        // Integrate one spin potential: multigrid restrict+products or cbg.
        auto integrate_V = [&](const std::vector<double>& veff) {
            if (use_prod) {
                std::vector<double> V_ks(static_cast<std::size_t>(nb) * nb,
                                         0.0);
                for (std::size_t l = 0; l < mg.levels.size(); ++l) {
                    std::vector<double> V_l;
                    if (l == 0) {
                        V_l = veff;
                    } else {
                        V_l = restrict_fft(mg.levels[0], mg.levels[l], veff,
                                           /*spectral*/ true);
                    }
                    const std::vector<double> Vl_ks = integrate_products(
                        mg.levels[l], products.by_level[l], products.prims,
                        products.contribs, nb, V_l, eps_default);
                    for (std::size_t kk = 0; kk < V_ks.size(); ++kk)
                        V_ks[kk] += Vl_ks[kk];
                }
                return vec_to_mat(V_ks, nb);
            }
            return vec_to_mat(integrate_potential(cbg, nb, veff), nb);
        };

        // Build dual KS + total energy (and fill res energy channels) from Pa/Pb.
        // ATV_GPW_PROF: Γ-UKS [gpw prof uks] matches RKS [gpw prof] shape.
        const bool uks_prof = std::getenv("ATV_GPW_PROF") != nullptr;
        using UksProfClk = std::chrono::steady_clock;
        auto uks_pms = [](UksProfClk::time_point a, UksProfClk::time_point b) {
            return std::chrono::duration<double, std::milli>(b - a).count();
        };
        int uks_energy_builds = 0;
        auto build_uks_H = [&](const RealMatrix& Pa_in, const RealMatrix& Pb_in)
            -> std::tuple<RealMatrix, RealMatrix, double> {
            ++uks_energy_builds;
            auto t0p = UksProfClk::now();
            std::vector<double> rho_a = collocate_P(Pa_in);
            std::vector<double> rho_b = collocate_P(Pb_in);
            auto t1p = UksProfClk::now();
            if (use_prod && use_mg
                && (mg.levels[0].n[0] != grid.n[0]
                    || mg.levels[0].n[1] != grid.n[1]
                    || mg.levels[0].n[2] != grid.n[2]))
                throw std::runtime_error(
                    "GPW UKS: multigrid finest level n[] != SCF grid n[]");
            if (use_prod && std::getenv("ATV_GPW_MG_TRACE")) {
                double qe = 0.0;
                for (std::size_t g = 0; g < rho_a.size(); ++g)
                    qe += rho_a[g] + rho_b[g];
                qe *= grid.weight();
                std::fprintf(stderr,
                             "[gpw uks mg] int rho_fine = %.10f "
                             "(expect ~%d valence e)\n",
                             qe, cfg.n_alpha + cfg.n_beta);
            }
            const internal::KsGridUks ks = internal::ks_grid_build_uks(
                grid, rho_a, rho_b, n_ion, v_locsr, cfg.xc, cfg.nonperiodic,
                cfg.vdw_df, cfg.vdw_df2, cfg.rvv10);
            auto t3p = UksProfClk::now();
            auto t2p = t1p;
            (void)t2p;
            if (cfg.vdw_df || cfg.vdw_df2 || cfg.rvv10)
                e_d3 = ks.e_nl_vdw;
            // Force path reuses dual V_eff + V_es.
            res.veff_a = ks.veff_a;
            res.veff_b = ks.veff_b;
            res.v_es = ks.v_es;

            RealMatrix Ha = H0 + integrate_V(res.veff_a);
            RealMatrix Hb = H0 + integrate_V(res.veff_b);
            auto t4p = UksProfClk::now();
            if (uks_prof)
                std::fprintf(stderr,
                             "[gpw prof uks] collocate=%.0f poisson=%.0f "
                             "xc=%.0f integrate=%.0f ms "
                             "(ngrid=%ld build=%d)\n",
                             uks_pms(t0p, t1p), uks_pms(t1p, t2p),
                             uks_pms(t2p, t3p), uks_pms(t3p, t4p),
                             grid.npts(), uks_energy_builds);

            double e_dftu = 0.0;
            if (do_dftu) {
                e_dftu += internal::apply_dft_plus_u(
                    Ha, Pa_in, S, ao_tags, z_of_atom, plus_u_slots, plus_u_z);
                e_dftu += internal::apply_dft_plus_u(
                    Hb, Pb_in, S, ao_tags, z_of_atom, plus_u_slots, plus_u_z);
            }
            res.e_dftu = e_dftu;

            const RealMatrix Ptot = Pa_in + Pb_in;
            res.e_kinetic = trace_product(Ptot, T, nb);
            res.e_nonlocal = 0.0;
            res.e_loc_sr = trace_product(Ptot, Vpp, nb) + ks.e_locgrid;
            res.e_es = ks.e_hartree - e_self;
            res.e_xc = ks.e_xc;
            const double e_total =
                res.e_kinetic + res.e_loc_sr + res.e_es + res.e_xc + e_dftu;
            return {std::move(Ha), std::move(Hb), e_total};
        };

        // ── Fixed-density UKS (Car–Parrinello electronic step) ────────────
        // max_iter == 0: one dual-KS build at the supplied (Pa, Pb); no
        // diagonalization / mixing / orbital update. Populates H_ks (α),
        // H_ks_beta (β), and S for the dual electronic EOMs.
        if (cfg.max_iter == 0) {
            if (guess_P == nullptr || guess_P_beta == nullptr)
                throw std::invalid_argument(
                    "GpwDft: max_iter==0 UKS fixed-density eval requires "
                    "guess_P (alpha) and guess_P_beta");
            if (cfg.use_ot)
                throw std::invalid_argument(
                    "GpwDft: fixed-density eval is incompatible with use_ot");
            // Pa/Pb already loaded from guesses above.
            auto [Ha, Hb, e_total] = build_uks_H(Pa, Pb);
            res.e_total = e_total;
            res.P = Pa;
            res.P_beta = Pb;
            res.H_ks = std::move(Ha);
            res.H_ks_beta = std::move(Hb);
            res.S = S;
            res.converged = true;
            res.iters = 0;
            fold_dispersion(res, Pa + Pb);
            return res;
        }

        // ── UKS OT (orbital transformation) ────────────────────────────────
        // Independent S-orthonormal occupied manifolds per spin; joint CG
        // step with Armijo line search on the rebuilt total energy. Gradient
        // uses open-shell occupation (occ=1). Same fixed point as DIIS UKS.
        // Dudarev DFT+U is included: build_uks_H applies PLUS_U to Ha/Hb and
        // folds e_dftu into the total energy used by Armijo (product path).
        if (cfg.use_ot) {
            const int na = cfg.n_alpha;
            const int nb_spin = cfg.n_beta;
            // Seed from Ca_prev/Cb_prev so dual-spin natural-orbital warm-start
            // (ASPC density guess) can differ per spin; cold path sets both
            // equal to C0 (core-guess) so this is bit-equivalent to reading C0.
            RealMatrix Cocc_a =
                na > 0 ? linalg::lowdin_orthonormalize(
                             RealMatrix(xt::view(Ca_prev, xt::all(),
                                                 xt::range(0, na))),
                             S)
                       : RealMatrix(xt::zeros<double>({nb, 0}));
            RealMatrix Cocc_b =
                nb_spin > 0 ? linalg::lowdin_orthonormalize(
                                  RealMatrix(xt::view(Cb_prev, xt::all(),
                                                      xt::range(0, nb_spin))),
                                  S)
                            : RealMatrix(xt::zeros<double>({nb, 0}));

            auto frob = [](const RealMatrix& a, const RealMatrix& b) {
                return xt::sum(a * b)();
            };
            RealMatrix d_a, d_b, pz_a_prev, pz_b_prev;
            double denom_prev = 1.0;
            bool have_prev = false;
            // Carried KS state (Γ RKS OT economics): accepted Armijo trial's
            // (Ha,Hb,E) becomes E(0) of the next outer — ~1 KS build / OT step.
            RealMatrix Ha, Hb;
            double e_total = 0.0;
            bool have_carried = false;

            for (int iter = 0; iter < cfg.max_iter; ++iter) {
                if (!have_carried) {
                    Pa = na > 0
                             ? linalg::build_density_matrix(Cocc_a, na, nb, 1.0)
                             : RealMatrix(xt::zeros<double>({nb, nb}));
                    Pb = nb_spin > 0
                             ? linalg::build_density_matrix(Cocc_b, nb_spin, nb,
                                                            1.0)
                             : RealMatrix(xt::zeros<double>({nb, nb}));
                    auto he0 = build_uks_H(Pa, Pb);
                    Ha = std::move(std::get<0>(he0));
                    Hb = std::move(std::get<1>(he0));
                    e_total = std::get<2>(he0);
                    have_carried = true;
                }
                res.iters = iter + 1;
                res.e_total = e_total;

                RealMatrix g_a = na > 0
                                     ? linalg::band_gradient(Ha, S, Cocc_a, 1.0)
                                     : RealMatrix(xt::zeros<double>({nb, 0}));
                RealMatrix g_b = nb_spin > 0
                                     ? linalg::band_gradient(Hb, S, Cocc_b, 1.0)
                                     : RealMatrix(xt::zeros<double>({nb, 0}));
                double gnorm = 0.0;
                for (double v : g_a)
                    gnorm = std::max(gnorm, std::abs(v));
                for (double v : g_b)
                    gnorm = std::max(gnorm, std::abs(v));

                if (std::getenv("ATV_GPW_TRACE"))
                    std::fprintf(stderr,
                                 "[gpw UKS OT %2d] E=%.8f  T=%.4f loc=%.4f "
                                 "es=%.4f xc=%.4f U=%.4f  dE=%.2e gnorm=%.2e\n",
                                 iter, e_total, res.e_kinetic, res.e_loc_sr,
                                 res.e_es, res.e_xc, res.e_dftu,
                                 std::abs(e_total - e_prev), gnorm);

                if (iter > 0 && gnorm < ot_grad_tol
                    && std::abs(e_total - e_prev) < cfg.tol) {
                    res.converged = true;
                    store_converged_state_uks(res, Ha, Hb, orth().X, S, na,
                                              nb_spin, nb);
                    res.mulliken_spin = internal::mulliken_spin_populations(
                        res.P, res.P_beta, S, ao_tags,
                        static_cast<int>(atoms.size()));
                    fold_dispersion(res, res.P + res.P_beta);
                    return res;
                }
                // DFT+U: accept an energy plateau after a few OT steps even if
                // the projected gradient sits just above the soft floor.
                if (do_dftu && iter >= 5
                    && std::abs(e_total - e_prev) < cfg.tol) {
                    res.converged = true;
                    store_converged_state_uks(res, Ha, Hb, orth().X, S, na,
                                              nb_spin, nb);
                    res.mulliken_spin = internal::mulliken_spin_populations(
                        res.P, res.P_beta, S, ao_tags,
                        static_cast<int>(atoms.size()));
                    fold_dispersion(res, res.P + res.P_beta);
                    return res;
                }
                e_prev = e_total;

                // Per-spin FULL_SINGLE_INVERSE preconditioner + PR+ CG.
                RealMatrix pz_a, pz_b;
                if (na > 0) {
                    const internal::OtPrecondSingleInverse pre_a(Ha, S, Cocc_a);
                    pz_a = linalg::s_tangent(Cocc_a, S, pre_a.apply(g_a));
                } else {
                    pz_a = RealMatrix(xt::zeros<double>({nb, 0}));
                }
                if (nb_spin > 0) {
                    const internal::OtPrecondSingleInverse pre_b(Hb, S, Cocc_b);
                    pz_b = linalg::s_tangent(Cocc_b, S, pre_b.apply(g_b));
                } else {
                    pz_b = RealMatrix(xt::zeros<double>({nb, 0}));
                }

                if (!have_prev) {
                    if (na > 0)
                        d_a = RealMatrix(-pz_a);
                    if (nb_spin > 0)
                        d_b = RealMatrix(-pz_b);
                } else {
                    double num = 0.0;
                    if (na > 0)
                        num += frob(g_a, RealMatrix(pz_a - pz_a_prev));
                    if (nb_spin > 0)
                        num += frob(g_b, RealMatrix(pz_b - pz_b_prev));
                    const double beta = std::max(0.0, num / denom_prev);
                    if (na > 0)
                        d_a = linalg::s_tangent(Cocc_a, S,
                                                RealMatrix(-pz_a + beta * d_a));
                    if (nb_spin > 0)
                        d_b = linalg::s_tangent(Cocc_b, S,
                                                RealMatrix(-pz_b + beta * d_b));
                }
                pz_a_prev = pz_a;
                pz_b_prev = pz_b;
                denom_prev = 0.0;
                if (na > 0)
                    denom_prev += frob(g_a, pz_a);
                if (nb_spin > 0)
                    denom_prev += frob(g_b, pz_b);
                if (!(denom_prev > 1e-300))
                    denom_prev = 1.0;
                have_prev = true;

                double slope = 0.0;
                if (na > 0)
                    slope += frob(g_a, d_a);
                if (nb_spin > 0)
                    slope += frob(g_b, d_b);
                if (slope >= 0.0) {
                    if (na > 0)
                        d_a = linalg::s_tangent(Cocc_a, S, -pz_a);
                    if (nb_spin > 0)
                        d_b = linalg::s_tangent(Cocc_b, S, -pz_b);
                    slope = 0.0;
                    if (na > 0)
                        slope += frob(g_a, d_a);
                    if (nb_spin > 0)
                        slope += frob(g_b, d_b);
                }
                // Frozen-H band parabola for initial alpha (occ=1 → Tr(CᵀHC)).
                auto band = [&](const RealMatrix& Ca_b,
                                const RealMatrix& Cb_b) {
                    double e = 0.0;
                    if (na > 0)
                        e += frob(Ca_b, xt::linalg::dot(Ha, Ca_b));
                    if (nb_spin > 0)
                        e += frob(Cb_b, xt::linalg::dot(Hb, Cb_b));
                    return e;
                };
                const double Eb0 = band(Cocc_a, Cocc_b);
                const RealMatrix Ca_t1 =
                    na > 0 ? internal::ot_parameterize(Cocc_a, d_a, S) : Cocc_a;
                const RealMatrix Cb_t1 =
                    nb_spin > 0 ? internal::ot_parameterize(Cocc_b, d_b, S)
                                : Cocc_b;
                const double Ebt = band(Ca_t1, Cb_t1);
                const double curv = 2.0 * (Ebt - Eb0 - slope);
                double alpha = (curv > 1e-300) ? -slope / curv : 1.0;
                if (!(alpha > 0.0))
                    alpha = 1.0;
                alpha = std::min(alpha, 2.0);
                constexpr double c1 = 1e-4;
                RealMatrix Cnew_a =
                    na > 0 ? internal::ot_parameterize(Cocc_a, alpha * d_a, S)
                           : Cocc_a;
                RealMatrix Cnew_b =
                    nb_spin > 0
                        ? internal::ot_parameterize(Cocc_b, alpha * d_b, S)
                        : Cocc_b;
                if (na > 0)
                    Cnew_a = linalg::lowdin_orthonormalize(Cnew_a, S);
                if (nb_spin > 0)
                    Cnew_b = linalg::lowdin_orthonormalize(Cnew_b, S);
                // Each trial builds full (Ha,Hb,E); accepted pair is carried.
                // Retry alpha = true-energy quadratic through (0,E0,slope) and
                // (α,E_t), clamped to [0.1,0.5]·α (Γ RKS OT parity).
                for (int bt = 0; bt < 50; ++bt) {
                    const RealMatrix Pa_t =
                        na > 0
                            ? linalg::build_density_matrix(Cnew_a, na, nb, 1.0)
                            : RealMatrix(xt::zeros<double>({nb, nb}));
                    const RealMatrix Pb_t =
                        nb_spin > 0 ? linalg::build_density_matrix(
                                          Cnew_b, nb_spin, nb, 1.0)
                                    : RealMatrix(xt::zeros<double>({nb, nb}));
                    auto trial = build_uks_H(Pa_t, Pb_t);
                    const double et = std::get<2>(trial);
                    if (et <= e_total + c1 * alpha * slope || bt == 49) {
                        Ha = std::move(std::get<0>(trial));
                        Hb = std::move(std::get<1>(trial));
                        e_total = et;
                        break;
                    }
                    const double denom = et - e_total - slope * alpha;
                    const double a_q = (denom > 1e-300)
                                           ? -slope * alpha * alpha / (2.0 * denom)
                                           : 0.5 * alpha;
                    alpha = std::min(std::max(a_q, 0.1 * alpha), 0.5 * alpha);
                    if (na > 0) {
                        Cnew_a =
                            internal::ot_parameterize(Cocc_a, alpha * d_a, S);
                        Cnew_a = linalg::lowdin_orthonormalize(Cnew_a, S);
                    }
                    if (nb_spin > 0) {
                        Cnew_b =
                            internal::ot_parameterize(Cocc_b, alpha * d_b, S);
                        Cnew_b = linalg::lowdin_orthonormalize(Cnew_b, S);
                    }
                }
                if (na > 0)
                    Cocc_a = Cnew_a;
                if (nb_spin > 0)
                    Cocc_b = Cnew_b;
                // Densities for next gradient are implicit in carried H
                // (built from accepted C); keep Pa/Pb in sync for exits.
                Pa = na > 0 ? linalg::build_density_matrix(Cocc_a, na, nb, 1.0)
                            : RealMatrix(xt::zeros<double>({nb, nb}));
                Pb = nb_spin > 0
                         ? linalg::build_density_matrix(Cocc_b, nb_spin, nb, 1.0)
                         : RealMatrix(xt::zeros<double>({nb, nb}));
            }
            // Budget exhausted: reuse carried (Ha,Hb,E) — no extra KS build.
            res.e_total = e_total;
            store_converged_state_uks(res, Ha, Hb, orth().X, S, na, nb_spin, nb);
            res.mulliken_spin = internal::mulliken_spin_populations(
                res.P, res.P_beta, S, ao_tags, static_cast<int>(atoms.size()));
            fold_dispersion(res, res.P + res.P_beta);
            return res;
        }

        internal::GpwDiis diis_a(use_diis ? cfg.diis_depth : 1);
        internal::GpwDiis diis_b(use_diis ? cfg.diis_depth : 1);

        for (int iter = 0; iter < cfg.max_iter; ++iter) {
            // Dual KS via shared builder (single-grid cbg or multigrid products).
            auto [Ha, Hb, e_total] = build_uks_H(Pa, Pb);
            res.iters = iter + 1;
            res.e_total = e_total;

            // Commutator residuals for both spins (DIIS-style convergence).
            const RealMatrix err_a =
                internal::gpw_diis_error(Ha, Pa, S, orth().X);
            const RealMatrix err_b =
                internal::gpw_diis_error(Hb, Pb, S, orth().X);
            double err_norm = 0.0;
            for (const double v : err_a)
                err_norm = std::max(err_norm, std::abs(v));
            for (const double v : err_b)
                err_norm = std::max(err_norm, std::abs(v));

            if (std::getenv("ATV_GPW_TRACE"))
                std::fprintf(stderr,
                             "[gpw UKS %2d] E=%.8f  T=%.4f loc=%.4f es=%.4f "
                             "xc=%.4f  dE=%.2e err=%.2e\n",
                             iter, e_total, res.e_kinetic, res.e_loc_sr,
                             res.e_es, res.e_xc, std::abs(e_total - e_prev),
                             err_norm);

            // DFT+U softens the DIIS residual; require energy Δ and either the
            // standard commutator tol or a looser residual when PLUS_U is on.
            const double err_gate = do_dftu ? 5e-3 : kDiisErrTol;
            if (iter > 0 && std::abs(e_total - e_prev) < cfg.tol
                && err_norm < err_gate) {
                res.converged = true;
                store_converged_state_uks(res, Ha, Hb, orth().X, S, cfg.n_alpha,
                                          cfg.n_beta, nb);
                res.mulliken_spin = internal::mulliken_spin_populations(
                    Pa, Pb, S, ao_tags, static_cast<int>(atoms.size()));
                // Hirshfeld uses total density ρ=ρα+ρβ (Pα+Pβ).
                fold_dispersion(res, Pa + Pb);
                return res;
            }
            e_prev = e_total;

            RealMatrix Ha_use = Ha;
            RealMatrix Hb_use = Hb;
            if (use_diis) {
                Ha_use = diis_a.step(Ha, err_a);
                Hb_use = diis_b.step(Hb, err_b);
            }
            auto [eiga, Ca] = diagonalize_fock_shifted(
                Ha_use, orth().X, S, Ca_prev, cfg.n_alpha, cfg.level_shift);
            auto [eigb, Cb] = diagonalize_fock_shifted(
                Hb_use, orth().X, S, Cb_prev, cfg.n_beta, cfg.level_shift);
            (void)eiga;
            (void)eigb;
            Ca_prev = Ca;
            Cb_prev = Cb;
            RealMatrix Pa_new =
                linalg::build_density_matrix(Ca, cfg.n_alpha, nb, 1.0);
            RealMatrix Pb_new =
                linalg::build_density_matrix(Cb, cfg.n_beta, nb, 1.0);
            if (use_diis) {
                // With DFT+U, pure DIIS can oscillate — damp with linear mix.
                if (do_dftu) {
                    const double a = std::min(std::max(cfg.mix, 0.15), 0.6);
                    Pa = (1.0 - a) * Pa + a * Pa_new;
                    Pb = (1.0 - a) * Pb + a * Pb_new;
                } else {
                    Pa = std::move(Pa_new);
                    Pb = std::move(Pb_new);
                }
            } else {
                Pa = (1.0 - cfg.mix) * Pa + cfg.mix * Pa_new;
                Pb = (1.0 - cfg.mix) * Pb + cfg.mix * Pb_new;
            }
        }
        // Budget exhausted: export full orbital state at the last mixed
        // densities (partial-SCF / second-gen CP UKS corrector consumers).
        // Matches RKS: store_converged_state* fills P/C/eps; leave
        // converged=false so the residual budget is honest.
        {
            const auto rebuilt = build_uks_H(Pa, Pb);
            const RealMatrix& Ha_final = std::get<0>(rebuilt);
            const RealMatrix& Hb_final = std::get<1>(rebuilt);
            res.e_total = std::get<2>(rebuilt);
            store_converged_state_uks(res, Ha_final, Hb_final, orth().X, S,
                                      cfg.n_alpha, cfg.n_beta, nb);
        }
        res.mulliken_spin = internal::mulliken_spin_populations(
            res.P, res.P_beta, S, ao_tags, static_cast<int>(atoms.size()));
        fold_dispersion(res, res.P + res.P_beta);
        return res;
    }

    // ── OT (orbital transformation) path ────────────────────────────────────
    // Diagonalization-free SCF: each outer step takes one preconditioned-CG
    // step on the OT manifold (gradient O(nb²·n_occ), no O(nb³) eigensolve).
    // Converges to the same gapped ground state as the diagonalization path.
    // The KS build of the accepted line-search trial is carried into the next
    // iteration, so one OT step costs ~1 KS build (frozen-H band parabola
    // seeds alpha; true-energy quadratic interpolation handles retries).
    if (cfg.use_ot) {
        RealMatrix Cocc = linalg::lowdin_orthonormalize(
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
        // KS state (H, E) carried across iterations: the accepted line-search
        // trial already built H at the new density, so E(0) is never
        // re-evaluated at the top of an iteration (CP2K-style OT economics —
        // ~1 KS build per OT step). Only iter 0 builds from scratch.
        RealMatrix H;
        double e_total = 0.0;
        for (int iter = 0; iter < cfg.max_iter; ++iter) {
            auto tA = Clk::now();
            if (iter == 0) {
                P = linalg::build_density_matrix(Cocc, cfg.n_occ, nb,
                                                 2.0); // 2 Cocc Coccᵀ
                auto he0 = build_H_and_energy(P);
                H = std::move(he0.first);
                e_total = he0.second;
            }
            auto tB = Clk::now();
            res.iters = iter + 1;
            res.e_total = e_total;
            int ls_rebuilds = 0;

            const RealMatrix g = linalg::band_gradient(H, S, Cocc);
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
                res.H_ks = H;
                res.S = S;
                break;
            }
            e_prev = e_total;

            // FULL_SINGLE_INVERSE preconditioner: H-aware, rebuilt each step
            // (CP2K's default for gapped systems; FULL_KINETIC T+S was too weak
            // and OT crawled). M = H + a S with a below the spectrum -> SPD.
            const internal::OtPrecondSingleInverse precond(H, S, Cocc);
            const RealMatrix pz = linalg::s_tangent(Cocc, S, precond.apply(g));
            auto tC = Clk::now();
            if (!have_prev) {
                d = -pz;
            } else {
                const double beta =
                    std::max(0.0, frob(g, pz - pz_prev) / denom_prev);
                d = linalg::s_tangent(Cocc, S, -pz + beta * d);
            }
            pz_prev = pz;
            denom_prev = frob(g, pz);
            have_prev = true;

            // Monotonic line search on the SCF TOTAL energy E[rho(C)]. The
            // frozen-H band energy 2 Tr(CᵀHC) is only the local quadratic model;
            // minimizing *it* overshoots the self-consistent E and let OT wander
            // (descend then drift up). Armijo acceptance on the rebuilt-H total
            // energy enforces monotonic descent — every accepted step lowers E.
            double slope = frob(g, d);
            if (slope
                >= 0.0) { // not a descent direction -> restart to (precond) SD
                d = linalg::s_tangent(Cocc, S, -pz);
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
            // Löwdin-clean every Armijo trial (multi-k parity): without it
            // trial energies see drifted metric and accept-only clean leaves
            // a residual floor that thrashing cannot leave.
            RealMatrix Cnew = internal::ot_parameterize(Cocc, alpha * d, S);
            Cnew = linalg::lowdin_orthonormalize(Cnew, S);
            // Each trial's KS build yields (H_t, E_t); the accepted trial's
            // pair becomes the next iteration's carried state, so acceptance
            // costs exactly one KS build. On an Armijo failure the retry
            // alpha is the minimizer of the quadratic through (0, E0, slope)
            // and (alpha, E_t) on the TRUE SCF energy, safeguarded to
            // [0.1, 0.5]·alpha (Nocedal–Wright); plain halving only when the
            // interpolation degenerates (non-convex model).
            for (int bt = 0; bt < 50; ++bt) {
                ++ls_rebuilds;
                auto trial = build_H_and_energy(
                    linalg::build_density_matrix(Cnew, cfg.n_occ, nb, 2.0));
                if (trial.second <= e_total + c1 * alpha * slope || bt == 49) {
                    H = std::move(trial.first);
                    e_total = trial.second;
                    break;
                }
                const double denom = trial.second - e_total - slope * alpha;
                const double a_q = (denom > 1e-300)
                                       ? -slope * alpha * alpha / (2.0 * denom)
                                       : 0.5 * alpha;
                alpha = std::min(std::max(a_q, 0.1 * alpha), 0.5 * alpha);
                Cnew = internal::ot_parameterize(Cocc, alpha * d, S);
                Cnew = linalg::lowdin_orthonormalize(Cnew, S);
            }
            Cocc = Cnew; // already Löwdin-clean
            if (prof) {
                auto tD = Clk::now();
                std::fprintf(stderr,
                             "[OT prof %2d] H_grad=%.0fms precond=%.0fms "
                             "linesearch=%.0fms (%d rebuilds) | total=%.0fms\n",
                             iter, ms(tA, tB), ms(tB, tC), ms(tC, tD),
                             ls_rebuilds, ms(tA, tD));
            }
        }
        // Always expose final occupied orbitals, density, and KS matrix so
        // partial-SCF consumers (SGCP OT corrector, force-at-density) work
        // even when the solve plateaued short of the tolerance.
        if (static_cast<int>(res.C.shape(0)) != nb) {
            res.C = xt::zeros<double>({nb, nb});
            for (int u = 0; u < nb; ++u)
                for (int i = 0; i < cfg.n_occ; ++i)
                    res.C(u, i) = Cocc(u, i);
        }
        {
            const RealMatrix P_final =
                linalg::build_density_matrix(Cocc, cfg.n_occ, nb, 2.0);
            res.P = P_final;
            // Budget-exhausted path: the accepted line-search trial already
            // built H at this final density — reuse the carried state instead
            // of a fresh KS build (converged path set H_ks on exit).
            if (res.H_ks.size() == 0) {
                res.H_ks = std::move(H);
                res.e_total = e_total;
                // Virtuals/eps for gpw_forces_from_scf consumers.
                if (res.eps.empty())
                    store_converged_state(res, res.H_ks, orth().X, cfg.n_occ,
                                          nb);
            }
            res.S = S;
            fold_dispersion(res, P_final);
        }
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
                // H_ks/S for force-at-density consumers (second-gen CP).
                res.H_ks = H;
                res.S = S;
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
                res.H_ks = H;
                res.S = S;
                break;
            }
            e_prev = e_total;
            P = (1.0 - cfg.mix) * P + cfg.mix * P_new;
        }
    }
    // Partial-SCF consumers (second-generation CP corrector): export the last
    // density / orbitals even when the residual budget was exhausted short of
    // tol. Without this, res.P/C stay empty and forces cannot be assembled at
    // the post-corrector density.
    if (!res.converged && P.size() > 0) {
        const auto [H_final, e_final] = build_H_and_energy(P);
        res.e_total = e_final;
        store_converged_state(res, H_final, orth().X, cfg.n_occ, nb);
        res.H_ks = H_final;
        res.S = S;
        // res.converged remains false — honest partial SCF.
    }
    // Final nonlocal E_c^{nl} on the (self-consistent) total density. V_nl was
    // already applied every SCF iteration; this sets e_disp for fold_dispersion.
    if (cfg.vdw_df || cfg.vdw_df2 || cfg.rvv10) {
        const RealMatrix& Pfinal = (res.P.size() > 0) ? res.P : P;
        const std::vector<double> Pvec = mat_to_vec(Pfinal, nb);
        const std::vector<double> rho = collocate_rho_fine(Pvec);
        e_d3 = nonlocal_vdw_on_grid(grid, rho, cfg.vdw_df, cfg.vdw_df2,
                                    cfg.rvv10, /*potential=*/false)
                   .energy;
    }
    fold_dispersion(res, (res.P.size() > 0) ? res.P : P);
    return res;
}

// ── GpwDft — the public solver face over run_gpw_scf_impl ────────

void GpwDft::build(const std::vector<GpwAtom>& reference) {
    if (reference.empty())
        throw std::invalid_argument(
            "GpwDft::build: empty atom set (nothing to solve)");
    if (is_uks()) {
        if (n_alpha < 0 || n_beta < 0 || n_alpha + n_beta < 1)
            throw std::invalid_argument(
                "GpwDft::build: UKS needs n_alpha,n_beta >= 0 with "
                "n_alpha+n_beta >= 1 (got n_alpha="
                + std::to_string(n_alpha) + " n_beta=" + std::to_string(n_beta)
                + ")");
        if (xc != XcKind::PADE && xc != XcKind::PBE && xc != XcKind::PBEsol
            && xc != XcKind::RevPBE && xc != XcKind::RPBE && xc != XcKind::BLYP
            && xc != XcKind::OptB88VdW)
            throw std::invalid_argument(
                "GpwDft::build: UKS supports XcKind::PADE, PBE, PBEsol, "
                "RevPBE, RPBE, BLYP, or OptB88VdW");
        // Nonlocal vdW (vdw_df / vdw_df2 / rvv10) is wired for Γ UKS and
        // multi-k UKS: V_nl on the total density lands in both spin
        // potentials each SCF iteration (build_uks_H / build_uks_k_energy)
        // and E_c^nl is folded onto e_disp at every UKS exit
        // (fold_dispersion / fold_dispersion_k_uks). Closed-shell multi-k
        // RKS nonlocal is product via gpw-vdw-kpoints.
    } else if (n_occ < 1) {
        throw std::invalid_argument(
            "GpwDft::build: n_occ must be >= 1 for closed-shell RKS (got "
            + std::to_string(n_occ) + "); or set n_alpha/n_beta for UKS");
    }
    if (smearing != Smearing::none) {
        // Multi-k RKS (DIIS and OT, with or without PLUS_U) + multi-k
        // collinear UKS (DIIS and OT, with or without PLUS_U) productized.
        if (!(degauss > 0.0) || !std::isfinite(degauss))
            throw std::invalid_argument(
                "GpwDft::build: degauss must be finite and > 0 [Ry] when "
                "smearing is enabled (got "
                + std::to_string(degauss) + ")");
        if (smearing == Smearing::methfessel_paxton && mp_order < 1)
            throw std::invalid_argument(
                "GpwDft::build: mp_order must be >= 1 for Methfessel-Paxton");
        if (extra_bands < 0)
            throw std::invalid_argument(
                "GpwDft::build: extra_bands must be >= 0");
    }
    if (!nonperiodic) {
        for (int d = 0; d < 3; ++d)
            if (!(cell[d] > 0.0))
                throw std::invalid_argument("GpwDft::build: periodic cell edge "
                                            + std::to_string(d)
                                            + " must be > 0 Bohr (got "
                                            + std::to_string(cell[d]) + ")");
    } else {
        // Martyna-Tuckerman cluster boundaries assume the charge density is
        // *contained* in the cell: the collocation does not wrap in this mode,
        // so a Gaussian centred near a face is clipped, the energy stops being
        // translation invariant, and the ionic forces acquire a huge spurious
        // boundary term (Sum_A F_A != 0). Energy still looks plausible, which
        // makes this a silent-wrong-answer trap: it cost one full worker shift
        // on gpw-fd-phonon-si, where an atom sat exactly at (0,0,0).
        //
        // Require every nucleus to keep one margin from each face. The margin
        // is a fixed fraction of the edge rather than a basis-tail estimate so
        // the check cannot itself depend on the basis being loaded.
        constexpr double kMtMarginFraction = 0.15;
        for (int d = 0; d < 3; ++d) {
            if (!(cell[d] > 0.0))
                throw std::invalid_argument(
                    "GpwDft::build: nonperiodic (MT) cell edge "
                    + std::to_string(d) + " must be > 0 Bohr (got "
                    + std::to_string(cell[d]) + ")");
            const double margin = kMtMarginFraction * cell[d];
            for (std::size_t a = 0; a < reference.size(); ++a) {
                const double p = reference[a].pos[d];
                if (p < margin || p > cell[d] - margin)
                    throw std::invalid_argument(
                        "GpwDft::build: nonperiodic (Martyna-Tuckerman) "
                        "boundaries require the system to sit inside the cell, "
                        "but atom "
                        + std::to_string(a) + " has axis " + std::to_string(d)
                        + " coordinate " + std::to_string(p)
                        + " Bohr, outside the allowed ["
                        + std::to_string(margin) + ", "
                        + std::to_string(cell[d] - margin)
                        + "] Bohr window (cell edge " + std::to_string(cell[d])
                        + " Bohr). Centre the molecule in the cell (or enlarge "
                          "the cell); an isolated system's absolute position "
                          "is "
                          "physically irrelevant, but the cluster boundary "
                          "condition is not translation invariant and off-"
                          "centre placement silently corrupts the forces.");
            }
        }
    }
    for (int d = 0; d < 3; ++d)
        if (k_mesh[d] < 1)
            throw std::invalid_argument(
                "GpwDft::build: k_mesh subdivision must be >= 1 (axis "
                + std::to_string(d) + " got " + std::to_string(k_mesh[d])
                + ")");
    if (k_points.size() != k_point_weights.size())
        throw std::invalid_argument(
            "GpwDft::build: k_points and k_point_weights must have the same "
            "length (got "
            + std::to_string(k_points.size()) + " vs "
            + std::to_string(k_point_weights.size()) + ")");
    // Match run_gpw_scf_impl: metallic smear always takes the k-point path
    // (including effective Γ via k_mesh = {1,1,1}).
    const bool kpath = !k_points.empty() || k_mesh[0] != 1 || k_mesh[1] != 1
                       || k_mesh[2] != 1 || smearing != Smearing::none;
    if (!kpath && !is_uks()) {
        // The Γ-point closed-shell RKS SCF has no +U fold — reject instead of
        // silently ignoring an active plus_u entry (no-fallback rule).
        for (const auto& pu : plus_u)
            if (pu.u_minus_j_ev > 0.0 && pu.l >= 0)
                throw std::invalid_argument(
                    "GpwDft::build: Dudarev plus_u is not supported on the "
                    "Γ-point closed-shell RKS path (it would be silently "
                    "ignored) — use collinear UKS (n_alpha/n_beta) or "
                    "k-point sampling (k_mesh / k_points)");
    }
    if (kpath) {
        // The k-point branch is periodic + local-PP-on-grid (RKS or collinear
        // UKS); every other combination is rejected fail-fast (no silent Γ
        // fallback).
        auto reject = [](const std::string& what) {
            throw std::invalid_argument(
                "GpwDft::build: k-point sampling does not support " + what
                + " (k-point RKS/UKS path; no fallback)");
        };
        // Multi-k OT is product for closed-shell RKS (integer and metallic
        // FD/MP occupations, with or without Dudarev DFT+U) and collinear
        // UKS (integer occupations and metallic FD/MP, with or without
        // Dudarev DFT+U — including the joint UKS OT+U+smear free-energy
        // path). Multi-k Dudarev DFT+U (PLUS_U) is supported for energy SCF
        // on DIIS and OT paths (RKS + collinear UKS), for the fixed-P
        // analytic stress, and for analytic ionic forces (occupation-matrix
        // W-fold in gpw_forces.cpp). REL_CUTOFF multigrid product
        // collocation (ngrids > 1 && rel_cutoff > 0) is supported for
        // closed-shell multi-k RKS (gpw-multigrid-kpoints) and collinear
        // multi-k UKS (gpw-uks-multigrid-kpoints; per-spin Bloch D/V stacks).
        if (ngrids > 1 && !(rel_cutoff > 0.0))
            reject("multigrid (ngrids > 1) without a positive rel_cutoff "
                   "(set rel_cutoff [Ry], the CP2K REL_CUTOFF)");
        if (nonperiodic)
            reject("nonperiodic (Martyna-Tuckerman) boundaries");
        if (!local_pp_on_grid)
            reject("the analytic per-pair local PP (local_pp_on_grid=false)");
        if (level_shift != 0.0)
            reject("level shifting");
        // Nonlocal vdW (vdw_df / vdw_df2 / rvv10) is product on closed-shell
        // multi-k RKS (SCF V_nl in build_k_energy + post-SCF E_c^nl in
        // fold_dispersion_k; single-grid and REL_CUTOFF multigrid —
        // gpw-vdw-kpoints) and collinear multi-k UKS (V_nl of the total
        // density added to both spin potentials in build_uks_k_energy /
        // DIIS twin + post-SCF E_c^nl in fold_dispersion_k_uks —
        // single-grid gpw-vdw-uks-kpoints; REL_CUTOFF multigrid product
        // gpw-vdw-uks-kpoint-multigrid; Γ UKS product via gpw-vdw-uks /
        // gpw-vdw-uks-multigrid).
        if (is_uks() && xc != XcKind::PADE && xc != XcKind::PBE
            && xc != XcKind::PBEsol && xc != XcKind::RevPBE
            && xc != XcKind::RPBE && xc != XcKind::BLYP
            && xc != XcKind::OptB88VdW)
            reject("k-UKS with this XcKind (need PADE, PBE, PBEsol, RevPBE, "
                   "RPBE, BLYP, or OptB88VdW)");
        if (!k_points.empty()) {
            double wsum = 0.0;
            for (const double w : k_point_weights) {
                if (!(w > 0.0))
                    throw std::invalid_argument(
                        "GpwDft::build: explicit k-point weights must be "
                        "> 0");
                wsum += w;
            }
            if (std::abs(wsum - 1.0) > 1e-8)
                throw std::invalid_argument(
                    "GpwDft::build: explicit k-point weights must sum to 1 "
                    "(got "
                    + std::to_string(wsum) + "); no implicit normalization");
        }
    }
    built_ = true;
}

GpwResult& GpwDft::run(const std::vector<GpwAtom>& atoms,
                       const RealMatrix* guess_C, const RealMatrix* guess_P,
                       const RealMatrix* guess_P_beta,
                       const std::vector<linalg::CplxMatrix>* guess_P_k,
                       const std::vector<linalg::CplxMatrix>* guess_P_k_beta) {
    if (!built_)
        throw std::logic_error(
            "GpwDft::run called before build() — validate the "
            "configuration first (no silent auto-build)");
    if (guess_P != nullptr || guess_P_beta != nullptr) {
        // Density guess: DIIS replaces P; OT purifies density → natural
        // orbitals for the OT manifold (RKS: total P; UKS: dual-spin —
        // SGCP OT corrector, second-gen-cpmd-uks-ot).
        if (is_uks()) {
            // UKS needs both spin channels; a partial guess would cold-
            // start one channel and look like a working XL path.
            if (guess_P == nullptr || guess_P_beta == nullptr)
                throw std::invalid_argument(
                    "GpwDft::run: UKS density guess requires both "
                    "guess_P (alpha) and guess_P_beta (no partial spin "
                    "guess)");
        } else if (guess_P_beta != nullptr) {
            throw std::invalid_argument(
                "GpwDft::run: guess_P_beta is UKS-only (closed-shell RKS "
                "takes total density on guess_P)");
        } else if (guess_P == nullptr) {
            // Unreachable if only guess_P_beta were set without UKS —
            // handled above. Keep the branch explicit for RKS null P.
        }
    }
    // Multi-k density guess + OT is product: each predicted P_k (RKS total)
    // or P_k^σ (UKS dual-spin) is purified to natural orbitals that seed the
    // per-k OT manifold(s) — the complex mirror of the Γ guess_P warm start
    // (SGCP OT corrector: second-gen-cpmd-ot-kpoints RKS,
    // second-gen-cpmd-ot-uks-kpoints UKS).
    if (!is_uks() && guess_P_k_beta != nullptr)
        throw std::invalid_argument(
            "GpwDft::run: guess_P_k_beta is multi-k UKS-only "
            "(closed-shell multi-k RKS takes total density on guess_P_k)");
    if (is_uks() && (guess_P_k != nullptr) != (guess_P_k_beta != nullptr))
        throw std::invalid_argument(
            "GpwDft::run: multi-k UKS density guess requires both "
            "guess_P_k (alpha) and guess_P_k_beta (no partial spin guess)");
    result_ = run_gpw_scf_impl(atoms, *this, guess_C, guess_P, guess_P_beta,
                               guess_P_k, guess_P_k_beta);
    return result_;
}

RealMatrix GpwDft::overlap(const std::vector<GpwAtom>& atoms) const {
    if (!built_)
        throw std::logic_error(
            "GpwDft::overlap called before build() — validate the "
            "configuration first (no silent auto-build)");
    if (uses_kpath(*this))
        throw std::invalid_argument(
            "GpwDft::overlap: this configuration solves on the k-point path "
            "(k_mesh / k_points / smearing) and has no real Γ overlap — call "
            "overlap_k() instead");
    const bool periodic = !nonperiodic;
    // Same setup + same assembly routine run_gpw_scf_impl uses; T and Vpp are
    // simply not requested, which leaves S bit-identical (every element is an
    // independent screened pair contraction).
    const OneElectronSetup setup(atoms, cell, periodic, eps_default);
    const int nb = setup.nb();
    RealMatrix S = xt::zeros<double>({nb, nb});
    setup.gamma(atoms, cell, periodic, local_pp_on_grid, S, /*T=*/nullptr,
                /*Vpp=*/nullptr);
    return S;
}

std::vector<linalg::CplxMatrix>
GpwDft::overlap_k(const std::vector<GpwAtom>& atoms) const {
    if (!built_)
        throw std::logic_error(
            "GpwDft::overlap_k called before build() — validate the "
            "configuration first (no silent auto-build)");
    if (!uses_kpath(*this))
        throw std::invalid_argument(
            "GpwDft::overlap_k: this configuration solves at Γ (no k_mesh / "
            "k_points / smearing) — call overlap() instead");
    const bool periodic = !nonperiodic;
    const OneElectronSetup setup(atoms, cell, periodic, eps_default);
    // internal::build_overlap_kinetic_k is the one phased S_k/T_k image sweep
    // the multi-k SCF branches call; T_k is built with S_k in that sweep and
    // discarded here.
    std::vector<linalg::CplxMatrix> S_k, T_k;
    internal::build_overlap_kinetic_k(setup.basis, setup.radii, cell,
                                      setup.mimg, kBuildScreen,
                                      sampled_kpoints(*this), S_k, T_k);
    return S_k;
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

namespace {
GpwDft copy_gpw(const GpwDft& m) {
    GpwDft p;
    for (int d = 0; d < 3; ++d)
        p.cell[d] = m.cell[d];
    p.cutoff_ry = m.cutoff_ry;
    p.ngrids = m.ngrids;
    p.rel_cutoff = m.rel_cutoff;
    p.n_occ = m.n_occ;
    p.n_alpha = m.n_alpha;
    p.n_beta = m.n_beta;
    p.max_iter = m.max_iter;
    p.tol = m.tol;
    p.eps_default = m.eps_default;
    p.mix = m.mix;
    p.diis_depth = m.diis_depth;
    p.lindep_threshold = m.lindep_threshold;
    p.level_shift = m.level_shift;
    p.xc = m.xc;
    p.nonperiodic = m.nonperiodic;
    p.local_pp_on_grid = m.local_pp_on_grid;
    p.d3 = m.d3;
    p.d3bj = m.d3bj;
    p.d3_atm = m.d3_atm;
    p.d3_rcut_bohr = m.d3_rcut_bohr;
    p.d4 = m.d4;
    p.d4_atm = m.d4_atm;
    p.ts = m.ts;
    p.xdm = m.xdm;
    p.mbd = m.mbd;
    for (int d = 0; d < 3; ++d)
        p.mbd_k_mesh[d] = m.mbd_k_mesh[d];
    p.vdw_df = m.vdw_df;
    p.vdw_df2 = m.vdw_df2;
    p.rvv10 = m.rvv10;
    p.hirshfeld = m.hirshfeld;
    p.use_ot = m.use_ot;
    for (int d = 0; d < 3; ++d)
        p.k_mesh[d] = m.k_mesh[d];
    p.k_time_reversal = m.k_time_reversal;
    p.k_points = m.k_points;
    p.k_point_weights = m.k_point_weights;
    p.plus_u = m.plus_u;
    p.atom_spin_guess = m.atom_spin_guess;
    p.mp2_n_virt = m.mp2_n_virt;
    p.mp2_os_scale = m.mp2_os_scale;
    p.mp2_ss_scale = m.mp2_ss_scale;
    p.smearing = m.smearing;
    p.degauss = m.degauss;
    p.mp_order = m.mp_order;
    p.extra_bands = m.extra_bands;
    return p;
}
} // namespace
std::vector<std::array<double, 3>>
gpw_forces_fd(const std::vector<GpwAtom>& atoms, const GpwDft& cfg, double h) {
    const int na = static_cast<int>(atoms.size());
    std::vector<std::array<double, 3>> forces(na, {0.0, 0.0, 0.0});
    for (int a = 0; a < na; ++a) {
        for (int k = 0; k < 3; ++k) {
            std::vector<GpwAtom> plus = atoms, minus = atoms;
            plus[a] = displaced(atoms[a], k, +h);
            minus[a] = displaced(atoms[a], k, -h);
            GpwDft plus_solver = copy_gpw(cfg);
            plus_solver.build(plus);
            const double e_plus = plus_solver.run(plus).e_total;
            GpwDft minus_solver = copy_gpw(cfg);
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
