/// @file cpu/scf/libcint_bridge.cpp
/// @brief libcint atm/bas/env bridge matching PySCF normalization.

#include "atomiverse/cpu/scf/libcint_bridge.h"

#ifdef ATV_HAS_LIBCINT

extern "C" {
#include <cint.h>
}

#include "atomiverse/core/frame_schema.h"
#include "atomiverse/core/gto.h"
#include "atomiverse/core/units.h"
#include "atomiverse/cpu/scf/basis_data/sto3g.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <tuple>
#include <utility>
#include <vector>

namespace ATV {
namespace cpu {
namespace scf {

namespace {

inline int cart_dim(int l) { return (l + 1) * (l + 2) / 2; }

double gaussian_int(int n, double alpha) {
    const double p = 0.5 * (static_cast<double>(n) + 1.0);
    return 0.5 * std::tgamma(p) / std::pow(alpha, p);
}

std::array<double, 3>
normalize_contracted_ao(int l, const std::array<double, 3>& exps,
                        const std::array<double, 3>& coeffs_with_prim_norm) {
    double norm2 = 0.0;
    for (int p = 0; p < 3; ++p)
        for (int q = 0; q < 3; ++q)
            norm2 += coeffs_with_prim_norm[p] * coeffs_with_prim_norm[q]
                     * gaussian_int(l * 2 + 2, exps[p] + exps[q]);

    const double scale = 1.0 / std::sqrt(norm2);
    return {
        coeffs_with_prim_norm[0] * scale,
        coeffs_with_prim_norm[1] * scale,
        coeffs_with_prim_norm[2] * scale,
    };
}

void append_shell(LibcintContext& ctx, int atom_index, int l,
                  const std::array<double, 3>& alpha_in,
                  const std::array<double, 3>& coeff_in) {
    std::array<std::tuple<double, double>, 3> pairs = {{
        {alpha_in[0], coeff_in[0]},
        {alpha_in[1], coeff_in[1]},
        {alpha_in[2], coeff_in[2]},
    }};
    std::sort(pairs.begin(), pairs.end(), [](const auto& a, const auto& b) {
        return std::get<0>(a) > std::get<0>(b);
    });

    std::array<double, 3> exps{};
    std::array<double, 3> coeffs{};
    for (int p = 0; p < 3; ++p) {
        exps[p] = std::get<0>(pairs[p]);
        coeffs[p] = std::get<1>(pairs[p]) * ATV::gto_norm(l, exps[p]);
    }
    coeffs = normalize_contracted_ao(l, exps, coeffs);

    const int ptr_exp = static_cast<int>(ctx.env.size());
    ctx.env.insert(ctx.env.end(), exps.begin(), exps.end());
    const int ptr_coeff = static_cast<int>(ctx.env.size());
    ctx.env.insert(ctx.env.end(), coeffs.begin(), coeffs.end());

    ctx.shell_offsets.push_back(ctx.nao);
    ctx.bas.push_back(atom_index);
    ctx.bas.push_back(l);
    ctx.bas.push_back(3);
    ctx.bas.push_back(1);
    ctx.bas.push_back(0);
    ctx.bas.push_back(ptr_exp);
    ctx.bas.push_back(ptr_coeff);
    ctx.bas.push_back(0);

    ctx.nao += cart_dim(l);
    ctx.nshells += 1;
}

} // namespace

LibcintContext::LibcintContext(const Frame& frame) {
    LibcintContext ctx;
    ctx.natm = static_cast<int>(frame.nlocal());
    ctx.atm.resize(ctx.natm * ATM_SLOTS, 0);
    ctx.env.resize(PTR_ENV_START, 0.0);

    const int* __restrict sp_type = frame.span(kAtom_type);
    const StorageReal* __restrict sp_x = frame.span(kAtom_x);
    const StorageReal* __restrict sp_y = frame.span(kAtom_y);
    const StorageReal* __restrict sp_z = frame.span(kAtom_z);

    for (int a = 0; a < ctx.natm; ++a) {
        const int Z = sp_type[a];
        const int coord_ptr = static_cast<int>(ctx.env.size());
        ctx.env.push_back(static_cast<double>(sp_x[a])
                          * units::kAngstromInBohr);
        ctx.env.push_back(static_cast<double>(sp_y[a])
                          * units::kAngstromInBohr);
        ctx.env.push_back(static_cast<double>(sp_z[a])
                          * units::kAngstromInBohr);

        ctx.atm[a * ATM_SLOTS + CHARGE_OF] = Z;
        ctx.atm[a * ATM_SLOTS + PTR_COORD] = coord_ptr;
        ctx.atm[a * ATM_SLOTS + NUC_MOD_OF] = POINT_NUC;

        // Iterate the checked-in STO-3G table from basis_data (generic path).
        // Each shell carries its own l / alpha / coeff; there is no sp
        // fusion at this layer -- the libcint bridge emits one libcint bas
        // entry per shell (s, then s, then p for 2nd-row atoms).
        const auto& table = basis_data::sto3g();
        if (Z < 0 || static_cast<std::size_t>(Z) >= table.size()
            || table[Z].shells.empty())
            continue;
        for (const auto& sh : table[Z].shells) {
            if (sh.alpha.size() != 3 || sh.coeff.size() != 3)
                throw std::runtime_error(
                    "libcint_bridge: expected 3 primitives per STO-3G shell");
            const std::array<double, 3> exps = {sh.alpha[0], sh.alpha[1],
                                                sh.alpha[2]};
            const std::array<double, 3> coeffs = {sh.coeff[0], sh.coeff[1],
                                                  sh.coeff[2]};
            append_shell(ctx, a, sh.l, exps, coeffs);
        }
    }

    *this = std::move(ctx);
}

} // namespace scf
} // namespace cpu
} // namespace ATV

#endif
