/// @file tests/cpu/scf/test_rys_eri.cpp
/// @brief Correctness test for the native Rys-quadrature ERI engine.
///
/// Strategy: build a real molecular system via the existing native
/// internal::IntegralWorkspace + ShellBlockPairList, then compute the full
/// ERI tensor two ways — once through the canonical McMurchie–Davidson
/// `native::eri_block` and once through `native::rys_eri_block`. The two
/// must agree (cross-check of the two native engines).

#if defined(ATV_HAS_HF)

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

#include <xtensor/containers/xtensor.hpp>

#include "atomiverse/core/units.h"
#include "atomiverse/cpu/scf/internal/integrals.h"
#include "atomiverse/cpu/scf/internal/shell_pair.h"
#include "atomiverse/cpu/scf/native_integrals.h"
#include "atomiverse/cpu/scf/rys_eri.h"

#ifdef _OPENMP
#include <omp.h>
#endif


using namespace ATV;
using namespace ATV::cpu::scf;

namespace {

/// @brief Build a 3-atom water frame (C2v, OH ≈ 0.96 Å).
cpu::Frame make_water() {
    cpu::Frame f(3);
    f.span(ATV::kAtom_x)[0] = 0.0;
    f.span(ATV::kAtom_y)[0] = 0.0;
    f.span(ATV::kAtom_z)[0] = 0.0;
    f.span(ATV::kAtom_x)[1] = 0.7572;
    f.span(ATV::kAtom_y)[1] = 0.5859;
    f.span(ATV::kAtom_z)[1] = 0.0;
    f.span(ATV::kAtom_x)[2] = -0.7572;
    f.span(ATV::kAtom_y)[2] = 0.5859;
    f.span(ATV::kAtom_z)[2] = 0.0;
    f.span(ATV::kAtom_type)[0] = 8;
    f.span(ATV::kAtom_type)[1] = 1;
    f.span(ATV::kAtom_type)[2] = 1;
    f.span(ATV::kAtom_mass)[0] = 15.999;
    f.span(ATV::kAtom_mass)[1] = 1.008;
    f.span(ATV::kAtom_mass)[2] = 1.008;
    f.box = ATV::cpu::Box::cubic(100.0);
    return f;
}

/// @brief Build a carbon monoxide frame.
cpu::Frame make_co() {
    cpu::Frame f(2);
    f.span(ATV::kAtom_x)[0] = 0.0;
    f.span(ATV::kAtom_y)[0] = 0.0;
    f.span(ATV::kAtom_z)[0] = 0.0;
    f.span(ATV::kAtom_x)[1] = 1.128;
    f.span(ATV::kAtom_y)[1] = 0.0;
    f.span(ATV::kAtom_z)[1] = 0.0;
    f.span(ATV::kAtom_type)[0] = 6;
    f.span(ATV::kAtom_type)[1] = 8;
    f.span(ATV::kAtom_mass)[0] = 12.011;
    f.span(ATV::kAtom_mass)[1] = 15.999;
    f.box = ATV::cpu::Box::cubic(100.0);
    return f;
}

/// @brief Compute the full (ij|kl) ERI tensor via native::rys_eri_block.
///
/// Loops every shell-block quartet (no canonical ordering / no
/// screening). This is for test convenience only — the production
/// Fock build uses the screened, canonical path.
xt::xtensor<double, 4> build_eri_via_rys(const ShellBlockPairList& sp) {
    int n = sp.n_basis;
    xt::xtensor<double, 4> ERI = xt::zeros<double>(
        {static_cast<std::size_t>(n), static_cast<std::size_t>(n),
         static_cast<std::size_t>(n), static_cast<std::size_t>(n)});

    const int nb = static_cast<int>(sp.blocks.size());
    for (int bi = 0; bi < nb; ++bi) {
        const auto& Bi = sp.blocks[bi];
        for (int bj = 0; bj < nb; ++bj) {
            const auto& Bj = sp.blocks[bj];
            if (Bi.l < Bj.l)
                continue; // rys_eri_block requires li >= lj
            for (int bk = 0; bk < nb; ++bk) {
                const auto& Bk = sp.blocks[bk];
                for (int bl = 0; bl < nb; ++bl) {
                    const auto& Bl = sp.blocks[bl];
                    if (Bk.l < Bl.l)
                        continue;

                    const int na = Bi.n_cart;
                    const int nb_c = Bj.n_cart;
                    const int nc = Bk.n_cart;
                    const int nd = Bl.n_cart;
                    std::vector<double> buf(na * nb_c * nc * nd, 0.0);
                    native::rys_eri_block(Bi, Bj, Bk, Bl, buf.data());

                    // Canonical call populates (ia,ib,ic,id). Use
                    // (ab|cd) = (ba|cd) = (ab|dc) = (ba|dc) (8-fold
                    // symmetry) to fill the three swapped orderings —
                    // those quartets are otherwise skipped by the
                    // li<lj / lk<ll guards above.
                    for (int ia = 0; ia < na; ++ia)
                        for (int ib = 0; ib < nb_c; ++ib)
                            for (int ic = 0; ic < nc; ++ic)
                                for (int id = 0; id < nd; ++id) {
                                    const double v =
                                        buf[((ia * nb_c + ib) * nc + ic) * nd
                                            + id];
                                    const int ai = Bi.ao_start + ia;
                                    const int aj = Bj.ao_start + ib;
                                    const int ak = Bk.ao_start + ic;
                                    const int al = Bl.ao_start + id;
                                    ERI(ai, aj, ak, al) = v;
                                    ERI(aj, ai, ak, al) = v;
                                    ERI(ai, aj, al, ak) = v;
                                    ERI(aj, ai, al, ak) = v;
                                }
                }
            }
        }
    }
    return ERI;
}

/// @brief Fetch the (ij|kl) entry from a dense 4-index tensor.
inline double eri_at(const xt::xtensor<double, 4>& T, int i, int j, int k,
                     int l) {
    return T(i, j, k, l);
}

void compare_tensors(const xt::xtensor<double, 4>& a,
                     const xt::xtensor<double, 4>& b, double tol,
                     const char* label) {
    const int n = static_cast<int>(a.shape()[0]);
    int bad = 0;
    double worst = 0.0;
    int wi = -1, wj = -1, wk = -1, wl = -1;

    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            for (int k = 0; k < n; ++k)
                for (int l = 0; l < n; ++l) {
                    double diff =
                        std::abs(eri_at(a, i, j, k, l) - eri_at(b, i, j, k, l));
                    if (diff > worst) {
                        worst = diff;
                        wi = i;
                        wj = j;
                        wk = k;
                        wl = l;
                    }
                    if (diff > tol)
                        ++bad;
                }
    EXPECT_EQ(bad, 0) << label << " — " << bad << " ERI elements exceed " << tol
                      << " (worst=" << worst << " at (" << wi << "," << wj
                      << "," << wk << "," << wl
                      << "): a=" << eri_at(a, wi, wj, wk, wl)
                      << " b=" << eri_at(b, wi, wj, wk, wl) << ")";
}

} // namespace

// ── Rys ERI should match the McMurchie-Davidson reference ────────────

// The old McMurchie-Davidson `eri_block` is only accurate to ~1e-7
// against libcint (documented drift), so its comparison tolerance has
// to be looser than the direct-vs-libcint test below. This test still
// catches gross algebraic errors in rys_eri_block.
TEST(RysEri, MatchesMcMurchieDavidson_H2O) {
    auto frame = make_water();
    auto pack = cpu::scf::internal::IntegralWorkspace{frame, cpu::scf::BasisSet::STO3G, 0,
                                       /*need_eri=*/true};
    auto sp = ShellBlockPairList{pack, 1e-14};

    auto eri_rys = build_eri_via_rys(sp);
    compare_tensors(pack.ERI, eri_rys, 5e-7, "H2O rys vs native M-D");
}

TEST(RysEri, MatchesMcMurchieDavidson_CO) {
    auto frame = make_co();
    auto pack = cpu::scf::internal::IntegralWorkspace{frame, cpu::scf::BasisSet::STO3G, 0,
                                       /*need_eri=*/true};
    auto sp = ShellBlockPairList{pack, 1e-14};

    auto eri_rys = build_eri_via_rys(sp);
    compare_tensors(pack.ERI, eri_rys, 5e-7, "CO rys vs native M-D");
}


// ── thread_local ERI-reuse determinism guard (spec scf-review-06, ac-005) ─
//
// The planned optimization converts the per-quartet `ang_idx`
// (rys_eri_block_canonical) and the non-canonical `scratch` (rys_eri_block)
// from heap-allocated locals to `static thread_local` buffers. This test pins
// that the SAME (la,lb,lc,ld) block quartet evaluated serially and inside a
// `#pragma omp parallel` region — every thread recomputing the identical
// quartet into its own buffer — yields bit-for-bit identical results, so no
// thread_local scratch bleeds across workers. Passes today (each call has its
// own local buffers); it stays the guard after the thread_local conversion
// lands. A d-shell quartet is chosen so the angular-index precompute
// (`ang_idx`) is non-trivial.
TEST(RysEri, ThreadLocalReuseDeterminism_DShell) {
    auto frame = make_water();
    // 6-31G* places a Cartesian d-shell on oxygen.
    auto pack = cpu::scf::internal::IntegralWorkspace{frame, cpu::scf::BasisSet::Pople631Gs, 0,
                                       /*need_eri=*/false};
    auto sp = ShellBlockPairList{pack, 1e-14};

    const int nb = static_cast<int>(sp.blocks.size());
    ASSERT_GT(nb, 1) << "need at least two distinct blocks";

    // Highest-l block (the d-shell) and lowest-l block, so the canonical
    // kernel constraints li>=lj and lk>=ll hold for (A B | A B).
    int hi = 0, lo = 0;
    for (int b = 1; b < nb; ++b) {
        if (sp.blocks[b].l > sp.blocks[hi].l)
            hi = b;
        if (sp.blocks[b].l < sp.blocks[lo].l)
            lo = b;
    }
    ASSERT_GT(sp.blocks[hi].l, sp.blocks[lo].l)
        << "6-31G* H2O must expose distinct angular momenta (d present)";

    const native::ShellBlock& A = sp.blocks[hi];
    const native::ShellBlock& B = sp.blocks[lo];
    const native::ShellBlock& C = sp.blocks[hi];
    const native::ShellBlock& D = sp.blocks[lo];
    const int sz = A.n_cart * B.n_cart * C.n_cart * D.n_cart;
    ASSERT_GT(sz, 1);

    // Serial reference.
    std::vector<double> serial(static_cast<std::size_t>(sz), 0.0);
    native::rys_eri_block(A, B, C, D, serial.data());

    int nT = 1;
#ifdef _OPENMP
    nT = std::max(2, omp_get_max_threads());
#endif
    std::vector<std::vector<double>> par(
        static_cast<std::size_t>(nT),
        std::vector<double>(static_cast<std::size_t>(sz), 0.0));

    // Each thread recomputes the same quartet repeatedly into its own buffer;
    // repetition raises the chance of exposing any cross-thread scratch bleed.
#ifdef _OPENMP
#pragma omp parallel num_threads(nT)
#endif
    {
        int tid = 0;
#ifdef _OPENMP
        tid = omp_get_thread_num();
#endif
        for (int rep = 0; rep < 8; ++rep) {
            std::fill(par[tid].begin(), par[tid].end(), 0.0);
            native::rys_eri_block(A, B, C, D, par[tid].data());
        }
    }

    for (int t = 0; t < nT; ++t)
        for (int e = 0; e < sz; ++e)
            EXPECT_EQ(par[t][e], serial[e])
                << "thread " << t << " element " << e
                << " differs from the serial rys_eri_block value "
                   "(thread_local scratch bleed)";
}

#endif // ATV_HAS_HF
