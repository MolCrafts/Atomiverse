/// @file cpu/scf/internal/shell_pair.cpp
/// @brief Build canonical shell pair list with Cauchy-Schwarz screening.

#include "atomiverse/cpu/scf/internal/shell_pair.h"
#include "atomiverse/core/log.h"
#include "atomiverse/cpu/scf/basis_set.h"
#include "atomiverse/cpu/scf/cart.h"
#include "atomiverse/cpu/scf/internal/integrals.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <utility>

namespace ATV {
namespace cpu {
namespace scf {

std::vector<native::ShellBlock>
blocks_from_shells(const std::vector<native::Shell>& shells) {
    const int n_shells = static_cast<int>(shells.size());
    std::vector<native::ShellBlock> blocks;
    blocks.reserve(shells.size());

    int shell_idx = 0, ao_idx = 0;
    while (shell_idx < n_shells) {
        const auto& sh = shells[static_cast<std::size_t>(shell_idx)];
        const int n_cart = n_cart_of(sh.l);

        native::ShellBlock blk;
        blk.n_prim = sh.n_prim;
        blk.alpha = sh.alpha;
        blk.coeff = sh.coeff;
        blk.center[0] = sh.center[0];
        blk.center[1] = sh.center[1];
        blk.center[2] = sh.center[2];
        blk.l = sh.l;
        blk.n_cart = n_cart;
        blk.ao_start = ao_idx;
        blocks.push_back(blk);

        shell_idx += n_cart;
        ao_idx += n_cart;
    }
    return blocks;
}

ShellPairList::ShellPairList(const internal::IntegralView& ints, double thr) {
    const int n = ints.n_ao_cart(); // Cartesian engine dimension

    ShellPairList spl;
    spl.n_basis = n;
    spl.threshold = thr;

    // Copy backing storage and fix up shell pointers to own data.
    // ints.shells[i].alpha/coeff point into method shell_data; after copy they
    // must be redirected into spl.shell_data.
    spl.shell_data = ints.shell_data;
    spl.atoms = ints.atoms;
    spl.shells.resize(n);
    for (int i = 0; i < n; ++i) {
        spl.shells[i] = ints.shells[i];
        const int np = spl.shells[i].n_prim;
        spl.shells[i].alpha = spl.shell_data[i].data();
        spl.shells[i].coeff = spl.shell_data[i].data() + np;
    }

    // Compute Schwarz bounds Q_ij = sqrt((ij|ij)) and collect screened pairs.
    //
    // The scalar native::eri path uses McMurchie-Davidson with R_tab sized
    // for l_max=1 (Ltot <= 4), so d-shells (Ltot up to 8) would read past
    // the buffer. Any pair containing an l>=2 shell bypasses screening —
    // we keep the pair unconditionally with schwarz=+inf. Downstream
    // direct-SCF paths using this AO-level list (UHF/UKS today) lose some
    // screening efficiency for d-containing systems; restricted-HF uses
    // the Rys-backed ShellBlockPairList and is unaffected.
    spl.pairs.reserve(static_cast<std::size_t>(n * (n + 1) / 2));
    int n_screened = 0;
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j <= i; ++j) {
            double schwarz;
            if (spl.shells[i].l >= 2 || spl.shells[j].l >= 2) {
                schwarz = std::numeric_limits<double>::infinity();
            } else {
                const double diag = native::eri(spl.shells[i], spl.shells[j],
                                                spl.shells[i], spl.shells[j]);
                schwarz = std::sqrt(std::max(0.0, diag));
            }
            if (schwarz > thr) {
                spl.pairs.push_back({i, j, schwarz});
            } else {
                ++n_screened;
            }
        }
    }

    ATV_DEBUG("[shell_pair] built n_basis={} pairs={} screened={} thr={:.1e}",
              n, spl.pairs.size(), n_screened, thr);

    *this = std::move(spl);
}

ShellBlockPairList::ShellBlockPairList(const internal::IntegralView& ints,
                                       double thr) {
    const int n = ints.n_ao_cart(); // Cartesian engine dimension
    const int n_raw_shells = static_cast<int>(ints.shells.size());

    ShellBlockPairList sbl;
    sbl.n_basis = n;
    sbl.threshold = thr;

    // ── Step 1: Group shells into blocks (shared helper).
    sbl.blocks = blocks_from_shells(ints.shells);
    const int n_blocks = static_cast<int>(sbl.blocks.size());

    // ── Step 2: Copy the primitive data into owned storage so the
    // block pair list can outlive the view, then redirect ShellBlock
    // alpha/coeff pointers into it.
    sbl.block_data.resize(n_blocks);
    int ao_idx = 0;
    for (int i = 0; i < n_blocks; ++i) {
        // ao_idx marches with the block; shell_data is indexed by
        // AO-level shell idx, and consecutive AOs in one block share
        // the same shell_data entry, so taking the first one suffices.
        sbl.block_data[i] = ints.shell_data[static_cast<std::size_t>(ao_idx)];
        sbl.blocks[i].alpha = sbl.block_data[i].data();
        sbl.blocks[i].coeff = sbl.block_data[i].data() + sbl.blocks[i].n_prim;
        ao_idx += sbl.blocks[i].n_cart;
    }
    (void)n_raw_shells;

    ATV_DEBUG("[shell_pair] block grouping done n_basis={} n_blocks={}", n,
              n_blocks);

    // ── Step 2: Canonical block pairs + Schwarz bounds ────────────────
    // For each canonical pair (si >= sj), compute the block-level Schwarz
    // bound Q_{AB} = sqrt(max_{ia,ib} |(ia ib|ia ib)|) using eri_block().
    sbl.pairs.reserve(static_cast<std::size_t>(n_blocks * (n_blocks + 1) / 2));
    int n_screened = 0;

    // Temporary buffer for the self-ERI block. Max across all shell
    // types is (ff|ff) = n_cart_of(kMaxL)^4 — 10^4 = 10000 doubles for
    // f-shells.
    double blk_buf[kMaxNCartOrbital * kMaxNCartOrbital * kMaxNCartOrbital
                   * kMaxNCartOrbital];

    for (int si = 0; si < n_blocks; ++si) {
        for (int sj = 0; sj <= si; ++sj) {
            const native::ShellBlock& Bsi = sbl.blocks[si];
            const native::ShellBlock& Bsj = sbl.blocks[sj];
            const int na = Bsi.n_cart, nb = Bsj.n_cart;

            // Compute eri_block(Bsi, Bsj, Bsi, Bsj)
            const int buf_size = na * nb * na * nb;
            for (int k = 0; k < buf_size; ++k)
                blk_buf[k] = 0.0;
            native::eri_block(Bsi, Bsj, Bsi, Bsj, blk_buf);

            // Block-level Schwarz bound: sqrt(max_{ia,ib} blk[ia,ib,ia,ib])
            double max_sq = 0.0;
            for (int ia = 0; ia < na; ++ia)
                for (int ib = 0; ib < nb; ++ib) {
                    // Index: [ia,ib,ia,ib] in layout (na, nb, na, nb)
                    const int idx = ((ia * nb + ib) * na + ia) * nb + ib;
                    if (blk_buf[idx] > max_sq)
                        max_sq = blk_buf[idx];
                }

            const double schwarz = std::sqrt(std::max(0.0, max_sq));
            if (schwarz > thr) {
                sbl.pairs.push_back({si, sj, schwarz});
            } else {
                ++n_screened;
            }
        }
    }

    ATV_DEBUG("[shell_pair] block pair list built n_blocks={} pairs={} "
              "screened={} thr={:.1e}",
              n_blocks, sbl.pairs.size(), n_screened, thr);

    *this = std::move(sbl);
}

} // namespace scf
} // namespace cpu
} // namespace ATV
