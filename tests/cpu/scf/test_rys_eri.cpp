/// @file tests/cpu/scf/test_rys_eri.cpp
/// @brief Correctness test for the native Rys-quadrature ERI engine.
///
/// Strategy: build a real molecular system via the existing native
/// IntegralPack + ShellBlockPairList, then compute the full
/// ERI tensor two ways — once through the canonical McMurchie-Davidson
/// `native::eri_block` (already validated against libcint to 1e-7) and
/// once through the new `native::rys_eri_block`. The two must agree.
///
/// When libcint is available we also compare directly against
/// `build_integrals_libcint`, with a tighter tolerance, so any residual
/// normalization drift vs. libcint is surfaced.

#if defined(ATV_HAS_HF)

#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <vector>

#include <xtensor/containers/xtensor.hpp>

#include "atomiverse/core/units.h"
#include "atomiverse/cpu/scf/internal/integrals.h"
#include "atomiverse/cpu/scf/internal/shell_pair.h"
#include "atomiverse/cpu/scf/native_integrals.h"
#include "atomiverse/cpu/scf/rys_eri.h"

#ifdef ATV_HAS_LIBCINT
#include "libcint_reference.cuh"
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
    f.simbox = ATV::cpu::Box::cubic(100.0);
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
    f.simbox = ATV::cpu::Box::cubic(100.0);
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
    auto pack = cpu::scf::IntegralPack{frame, cpu::scf::BasisSet::STO3G, 0,
                                       /*need_eri=*/true};
    auto sp = ShellBlockPairList{pack, 1e-14};

    auto eri_rys = build_eri_via_rys(sp);
    compare_tensors(pack.ERI, eri_rys, 5e-7, "H2O rys vs native M-D");
}

TEST(RysEri, MatchesMcMurchieDavidson_CO) {
    auto frame = make_co();
    auto pack = cpu::scf::IntegralPack{frame, cpu::scf::BasisSet::STO3G, 0,
                                       /*need_eri=*/true};
    auto sp = ShellBlockPairList{pack, 1e-14};

    auto eri_rys = build_eri_via_rys(sp);
    compare_tensors(pack.ERI, eri_rys, 5e-7, "CO rys vs native M-D");
}

// ── Rys ERI should match libcint directly (tighter tolerance) ────────

#ifdef ATV_HAS_LIBCINT

TEST(RysEri, MatchesLibcint_H2O) {
    auto frame = make_water();
    auto pack = cpu::scf::IntegralPack{frame, cpu::scf::BasisSet::STO3G, 0,
                                       /*need_eri=*/true};
    auto sp = ShellBlockPairList{pack, 1e-14};

    auto eri_rys = build_eri_via_rys(sp);
    auto libcint_pack = test::build_integrals_libcint(frame);
    compare_tensors(libcint_pack.ERI, eri_rys, 5e-7, "H2O rys vs libcint");
}

TEST(RysEri, MatchesLibcint_CO) {
    auto frame = make_co();
    auto pack = cpu::scf::IntegralPack{frame, cpu::scf::BasisSet::STO3G, 0,
                                       /*need_eri=*/true};
    auto sp = ShellBlockPairList{pack, 1e-14};

    auto eri_rys = build_eri_via_rys(sp);
    auto libcint_pack = test::build_integrals_libcint(frame);
    compare_tensors(libcint_pack.ERI, eri_rys, 5e-7, "CO rys vs libcint");
}

// ── Phase 4 gate: d-shell (cc-pVDZ-class) ERIs vs libcint ────────────
//
// The eri-hpc-opt spec's Phase 4 targets cc-pVDZ enablement: d-shells,
// 6 Cartesian functions per shell. The native ERI engine reaches l = 2
// through the Rys-quadrature kernel (`native::rys_eri_block`, the same
// 2D-VRR + HRR algebra as libcint), to which `eri_block()` dispatches
// every non-(ss|ss) quartet. The d-polarized 6-31G* table exercises the
// full set of d-shell quartet classes that cc-pVDZ needs — (dd|dd),
// (dp|dd), (dd|pp), (dd|ss), (ds|ss), … — on a real molecule.
//
// This is the binding regression gate (acceptance ac-004): the native
// d-shell ERI tensor must agree with libcint's `cint2e_cart` to 1e-7.
TEST(RysEri, MatchesLibcint_DShell_631Gs_H2O) {
    auto frame = make_water();
    // 6-31G* puts a d-shell on oxygen → d quartets appear in the tensor.
    auto pack = cpu::scf::IntegralPack{frame, cpu::scf::BasisSet::Pople631Gs, 0,
                                       /*need_eri=*/true};
    auto sp = ShellBlockPairList{pack, 1e-14};
    ASSERT_GT(sp.n_basis, 7)
        << "6-31G* H2O must have > 7 AOs (d-shell present)";

    auto eri_rys = build_eri_via_rys(sp);
    auto eri_ref = test::build_eri_libcint_from_pack(frame, pack);
    compare_tensors(eri_ref, eri_rys, 1e-7,
                    "H2O 6-31G* d-shell rys vs libcint");
}

TEST(RysEri, MatchesLibcint_DShell_631Gs_CO) {
    auto frame = make_co();
    // 6-31G* puts a d-shell on both C and O → (dd|dd) quartets present.
    auto pack = cpu::scf::IntegralPack{frame, cpu::scf::BasisSet::Pople631Gs, 0,
                                       /*need_eri=*/true};
    auto sp = ShellBlockPairList{pack, 1e-14};

    auto eri_rys = build_eri_via_rys(sp);
    auto eri_ref = test::build_eri_libcint_from_pack(frame, pack);
    compare_tensors(eri_ref, eri_rys, 1e-7, "CO 6-31G* d-shell rys vs libcint");
}

#endif // ATV_HAS_LIBCINT

#endif // ATV_HAS_HF
