/// @file cpu/scf/two_electron.cpp
/// @brief Dense ERI tensor assembly (internal::IntegralWorkspace::eri_dense) via the
///        amortized shell-block engine native::eri_block + OpenMP.

#include "atomiverse/cpu/scf/internal/integrals.h"

#include <array>
#include <vector>

#include "atomiverse/core/log.h"
#include "atomiverse/cpu/scf/cart.h"
#include "atomiverse/cpu/scf/internal/shell_pair.h"
#include "atomiverse/cpu/scf/native_integrals.h"

#ifdef _OPENMP
#include <omp.h>
#endif

namespace ATV {
namespace cpu {
namespace scf {

xt::xtensor<double, 4>
internal::IntegralWorkspace::eri_dense(const ShellList& shells) {
    const int n = shells.n_basis;
    xt::xtensor<double, 4> ERI = xt::zeros<double>({n, n, n, n});

    // Group the per-AO shell list into libcint-style l-blocks (s → n_cart=1,
    // p → 3, d → 6, …) so the amortized Rys/xsimd native::eri_block kernel
    // replaces the per-AO scalar native::eri: one Boys/R-table build per
    // shell-block primitive quartet instead of one per AO integral. This is
    // the same ShellBlock machinery that ShellBlockPairList::g_rhf uses.
    const std::vector<native::ShellBlock> blocks =
        blocks_from_shells(shells.shells);
    const int n_blocks = static_cast<int>(blocks.size());

    // Canonical shell-block pairs (bi >= bj), matching ShellBlockPairList.
    std::vector<std::array<int, 2>> block_pairs;
    block_pairs.reserve(static_cast<std::size_t>(n_blocks) * (n_blocks + 1)
                        / 2);
    for (int bi = 0; bi < n_blocks; ++bi)
        for (int bj = 0; bj <= bi; ++bj)
            block_pairs.push_back({bi, bj});
    const int n_bp = static_cast<int>(block_pairs.size());

    const auto t_start = LogClock::now();

    // Parallelize over the canonical OUTER block pair `left`. Concurrency is
    // race-free by DISJOINTNESS, not by reduction: every cell (p,q,r,s) this
    // 8-fold scatter writes has max(blockpair{p,q}, blockpair{r,s}) == left
    // (the inner ket pair index `right` is always ≤ the outer bra pair index
    // `left`, so a cell's "owning left" is uniquely max, its "owning right" is
    // min). Distinct `left` iterations therefore touch mutually exclusive
    // tensor cells → plain assignment, NO thread-local buffer, NO atomics, NO
    // reduction (unlike g_rhf, which ACCUMULATES J/K and so needs one). Within
    // a single diagonal (left == right) quartet the 8-fold scatter may re-write
    // a cell, but with the identical value on the same thread — benign.
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
    for (int left = 0; left < n_bp; ++left) {
        const native::ShellBlock& A = blocks[block_pairs[left][0]];
        const native::ShellBlock& B = blocks[block_pairs[left][1]];
        const int na = A.n_cart, nb = B.n_cart;

        for (int right = 0; right <= left; ++right) {
            const native::ShellBlock& C = blocks[block_pairs[right][0]];
            const native::ShellBlock& D = blocks[block_pairs[right][1]];
            const int nc = C.n_cart, nd = D.n_cart;

            // Max block size (ff|ff) = 10⁴ doubles; zero only the actually-used
            // na*nb*nc*nd prefix (eri_block accumulates into it), matching the
            // g_rhf convention rather than `= {}`-clearing the whole buffer.
            double blk[kMaxNCartOrbital * kMaxNCartOrbital * kMaxNCartOrbital
                       * kMaxNCartOrbital];
            const int blk_size = na * nb * nc * nd;
            for (int z = 0; z < blk_size; ++z)
                blk[z] = 0.0;
            native::eri_block(A, B, C, D, blk);

            // 8-fold-symmetry scatter into the disjoint cells owned by `left`.
            for (int ia = 0; ia < na; ++ia) {
                const int a = A.ao_start + ia;
                for (int ib = 0; ib < nb; ++ib) {
                    const int b = B.ao_start + ib;
                    const double* blk_ab = blk + (ia * nb + ib) * (nc * nd);
                    for (int ic = 0; ic < nc; ++ic) {
                        const int c = C.ao_start + ic;
                        for (int id = 0; id < nd; ++id) {
                            const int d = D.ao_start + id;
                            const double v = blk_ab[ic * nd + id];
                            ERI(a, b, c, d) = v;
                            ERI(b, a, c, d) = v;
                            ERI(a, b, d, c) = v;
                            ERI(b, a, d, c) = v;
                            ERI(c, d, a, b) = v;
                            ERI(d, c, a, b) = v;
                            ERI(c, d, b, a) = v;
                            ERI(d, c, b, a) = v;
                        }
                    }
                }
            }
        }
    }
    ATV_DEBUG("[eri_dense] basis={} blocks={} ms={:.1f}", n, n_blocks,
              elapsed_ms(t_start));

    return ERI;
}

} // namespace scf
} // namespace cpu
} // namespace ATV
