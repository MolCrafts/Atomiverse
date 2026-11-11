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
    f.span(ATV::kAtom_atomic_number)[0] = 8;
    f.span(ATV::kAtom_atomic_number)[1] = 1;
    f.span(ATV::kAtom_atomic_number)[2] = 1;
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
    f.span(ATV::kAtom_atomic_number)[0] = 6;
    f.span(ATV::kAtom_atomic_number)[1] = 8;
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
    auto pack = cpu::scf::internal::IntegralWorkspace{
        frame, cpu::scf::BasisSet::STO3G, 0,
        /*need_eri=*/true};
    auto sp = ShellBlockPairList{pack, 1e-14};

    auto eri_rys = build_eri_via_rys(sp);
    compare_tensors(pack.ERI, eri_rys, 5e-7, "H2O rys vs native M-D");
}

TEST(RysEri, MatchesMcMurchieDavidson_CO) {
    auto frame = make_co();
    auto pack = cpu::scf::internal::IntegralWorkspace{
        frame, cpu::scf::BasisSet::STO3G, 0,
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
    auto pack = cpu::scf::internal::IntegralWorkspace{
        frame, cpu::scf::BasisSet::Pople631Gs, 0,
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

// The scalar MD path is independent of the production Rys block dispatcher.
// Large separations exercise its asymptotic branch and translation stability.
TEST(RysEri, WidelySeparatedSpQuartetsMatchScalarMD) {
    auto one = make_water();
    cpu::Frame frame(6);
    for (int a = 0; a < 6; ++a) {
        const int i = a % 3;
        frame.span(kAtom_atomic_number)[a] = one.span(kAtom_atomic_number)[i];
        frame.span(kAtom_x)[a] = one.span(kAtom_x)[i] + (a >= 3 ? 28.0 : 0.0);
        frame.span(kAtom_y)[a] = one.span(kAtom_y)[i];
        frame.span(kAtom_z)[a] = one.span(kAtom_z)[i];
    }
    frame.box = cpu::Box::cubic(100.0);
    cpu::scf::internal::IntegralWorkspace pack{frame, BasisSet::STO3G, 0,
                                               false};
    const auto dense = build_eri_via_rys(ShellBlockPairList{pack, 1e-14});
    double worst = 0.0, got = 0.0, want = 0.0;
    int at[4]{};
    const int n = pack.n_basis;
    for (int a = 0; a < n; ++a)
        for (int b = 0; b < n; ++b)
            for (int c = 0; c < n; ++c)
                for (int d = 0; d < n; ++d) {
                    const double ref =
                        native::eri(pack.shells[a], pack.shells[b],
                                    pack.shells[c], pack.shells[d]);
                    const double err = std::abs(dense(a, b, c, d) - ref);
                    if (err > worst) {
                        worst = err;
                        got = dense(a, b, c, d);
                        want = ref;
                        at[0] = a;
                        at[1] = b;
                        at[2] = c;
                        at[3] = d;
                    }
                }
    EXPECT_LT(worst, 2e-11)
        << "quartet=" << at[0] << ',' << at[1] << ',' << at[2] << ',' << at[3]
        << " Rys=" << got << " MD=" << want;
}

TEST(RysEri, SinglePAccumulatesInAllShellOrders) {
    auto frame = make_water();
    frame.span(kAtom_z)[1] = 0.4;
    frame.span(kAtom_z)[2] = -0.3;
    const internal::IntegralWorkspace pack{frame, BasisSet::STO3G, 0, false};
    const ShellBlockPairList pairs{pack, 1e-14};
    const native::ShellBlock* p = nullptr;
    std::vector<const native::ShellBlock*> s;
    for (const auto& block : pairs.blocks) {
        if (block.l == 0)
            s.push_back(&block);
        else if (block.l == 1)
            p = &block;
    }
    ASSERT_NE(p, nullptr);
    ASSERT_GE(s.size(), 3u);

    const native::ShellBlock* orders[][4] = {
        {p, s[0], s[1], s[2]},
        {s[0], p, s[1], s[2]},
        {s[1], s[2], p, s[0]},
        {s[1], s[2], s[0], p},
    };
    const double initial[] = {0.25, -0.5, 0.75};
    for (const auto& order : orders) {
        double result[] = {initial[0], initial[1], initial[2]};
        native::rys_eri_block(*order[0], *order[1], *order[2], *order[3],
                              result);
        for (int axis = 0; axis < 3; ++axis) {
            const auto shell =
                [&](const native::ShellBlock& block) -> const native::Shell& {
                return pack.shells[block.ao_start + (block.l == 1 ? axis : 0)];
            };
            const double ref = native::eri(shell(*order[0]), shell(*order[1]),
                                           shell(*order[2]), shell(*order[3]));
            EXPECT_NEAR(result[axis], initial[axis] + ref, 2e-11);
        }
    }
}

TEST(RysEri, MixedAngularPairsAccumulateInAllShellOrders) {
    auto frame = make_water();
    frame.span(kAtom_z)[1] = 0.4;
    frame.span(kAtom_z)[2] = -0.3;
    const internal::IntegralWorkspace pack{frame, BasisSet::Def2SVP, 0, false};
    const ShellBlockPairList pairs{pack, 1e-14};
    const auto find_shell = [&](int l, int side) {
        return std::find_if(pairs.blocks.begin(), pairs.blocks.end(),
                            [=](const native::ShellBlock& block) {
                                const double x = block.center[0];
                                return block.l == l
                                       && ((x > 0) - (x < 0)) == side;
                            });
    };
    const auto d = find_shell(2, 0);
    const auto p_bra = find_shell(1, 1);
    const auto s_bra = find_shell(0, 1);
    const auto p_ket = find_shell(1, 0);
    const auto s_ket = find_shell(0, -1);
    ASSERT_NE(d, pairs.blocks.end());
    ASSERT_NE(p_bra, pairs.blocks.end());
    ASSERT_NE(s_bra, pairs.blocks.end());
    ASSERT_NE(p_ket, pairs.blocks.end());
    ASSERT_NE(s_ket, pairs.blocks.end());

    const native::ShellBlock* quartets[][4] = {
        {&*d, &*s_bra, &*p_ket, &*s_ket},
        {&*d, &*p_bra, &*p_ket, &*s_ket},
        {&*d, &*d, &*d, &*p_bra},
        {&*d, &*d, &*d, &*d},
        {&*s_bra, &*s_ket, &*p_bra, &*p_ket},
        {&*p_bra, &*s_ket, &*p_ket, &*s_bra},
        {&*p_bra, &*s_ket, &*p_bra, &*p_ket},
        {&*p_bra, &*p_ket, &*s_bra, &*s_ket},
        {&*p_bra, &*p_ket, &*p_ket, &*s_bra},
    };
    for (const auto& shells : quartets) {
        const auto& A = *shells[0];
        const auto& B = *shells[1];
        const auto& C = *shells[2];
        const auto& D = *shells[3];
        SCOPED_TRACE(testing::Message() << A.l << B.l << '|' << C.l << D.l);
        const int na = A.n_cart, nb = B.n_cart, nc = C.n_cart, nd = D.n_cart;
        std::vector<double> reference;
        reference.reserve(na * nb * nc * nd);
        for (int a = 0; a < na; ++a)
            for (int b = 0; b < nb; ++b)
                for (int c = 0; c < nc; ++c)
                    for (int d = 0; d < nd; ++d)
                        reference.push_back(
                            native::eri(pack.shells[A.ao_start + a],
                                        pack.shells[B.ao_start + b],
                                        pack.shells[C.ao_start + c],
                                        pack.shells[D.ao_start + d]));

        for (bool swap_ab : {false, true}) {
            for (bool swap_cd : {false, true}) {
                SCOPED_TRACE(testing::Message() << "swap_ab=" << swap_ab
                                                << " swap_cd=" << swap_cd);
                const int out_nb = swap_ab ? na : nb;
                const int out_nc = swap_cd ? nd : nc;
                const int out_nd = swap_cd ? nc : nd;
                std::vector<double> result(reference.size());
                for (std::size_t i = 0; i < result.size(); ++i)
                    result[i] = 0.25 + i / 1024.0;

                for (int count : {1, 2}) {
                    native::rys_eri_block(swap_ab ? B : A, swap_ab ? A : B,
                                          swap_cd ? D : C, swap_cd ? C : D,
                                          result.data());
                    for (int a = 0; a < na; ++a)
                        for (int b = 0; b < nb; ++b)
                            for (int c = 0; c < nc; ++c)
                                for (int d = 0; d < nd; ++d) {
                                    const int ref =
                                        ((a * nb + b) * nc + c) * nd + d;
                                    const int i = swap_ab ? b : a;
                                    const int j = swap_ab ? a : b;
                                    const int k = swap_cd ? d : c;
                                    const int l = swap_cd ? c : d;
                                    const int out =
                                        ((i * out_nb + j) * out_nc + k) * out_nd
                                        + l;
                                    EXPECT_NEAR(result[out],
                                                0.25 + out / 1024.0
                                                    + count * reference[ref],
                                                2e-11);
                                }
                }
            }
        }
    }
}

TEST(RysEri, UncachedHighAngularOffsetsMatchScalarReference) {
    const double alpha = 1.2;
    const double coeff = 0.8;
    const native::ShellBlock fa{1, &alpha, &coeff, {0.1, 0.2, -0.3}, 3, 10, 0};
    const native::ShellBlock fb{1, &alpha, &coeff, {0.6, 0.3, 0.4}, 3, 10, 0};
    const native::ShellBlock fc{1, &alpha, &coeff, {0.8, -0.4, 0.1}, 3, 10, 0};
    const native::ShellBlock fd{1, &alpha, &coeff, {-0.2, 0.1, 0.5}, 3, 10, 0};
    auto sa = fa;
    sa.l = 0;
    sa.n_cart = n_cart_of(sa.l);
    auto gb = fb;
    gb.l = 4;
    gb.n_cart = n_cart_of(gb.l);

    const auto component = [](const native::ShellBlock& block, int cart) {
        return native::Shell{
            block.n_prim, block.alpha,
            block.coeff,  {block.center[0], block.center[1], block.center[2]},
            block.l,      cart};
    };
    const native::ShellBlock* quartets[][2] = {{&sa, &gb}, {&fa, &fb}};
    for (const auto& bra : quartets) {
        const auto& A = *bra[0];
        const auto& B = *bra[1];
        SCOPED_TRACE(testing::Message() << A.l << B.l << "|33");
        const int na = A.n_cart, nb = B.n_cart, nc = fc.n_cart, nd = fd.n_cart;
        std::vector<double> out(na * nb * nc * nd, 0.125);
        native::rys_eri_block(A, B, fc, fd, out.data());
        for (int a : {0, na / 2, na - 1})
            for (int b : {0, nb / 2, nb - 1})
                for (int c : {0, nc / 2, nc - 1})
                    for (int d : {0, nd / 2, nd - 1}) {
                        const auto ref =
                            native::eri(component(A, a), component(B, b),
                                        component(fc, c), component(fd, d));
                        EXPECT_NEAR(out[((a * nb + b) * nc + c) * nd + d],
                                    0.125 + ref, 2e-11);
                    }
    }
}

#endif // ATV_HAS_HF
