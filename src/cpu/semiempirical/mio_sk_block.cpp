/// @file src/cpu/semiempirical/mio_sk_block.cpp
/// mio-1-1 Slater–Koster two-centre block + its bond-vector derivative.
/// Shared by DFTB0/2/3 so the value and the gradient cannot drift apart.
#include "atomiverse/cpu/semiempirical/internal/mio_sk_block.h"

#include <cmath>

#include "atomiverse/cpu/semiempirical/internal/mio_sk_table.h"

namespace ATV {
namespace cpu {
namespace semiempirical {
namespace internal {

void MioSketchPair::add_block(double dx, double dy, double dz, double r,
                              linalg::RealMatrix& h0, linalg::RealMatrix& s,
                              int ia0, int ib0) const {
    if (r < 1e-12)
        return;

    // Ordered A→B table: H_sps is s on A, p on B. Reverse for p_A–s_B.
    const MioSkTable ab(a_.z, b_.z);
    const MioSkTable ba(b_.z, a_.z);

    const double Hss = ab.hopping_ss(r);
    const double Hsp_ab = ab.hopping_sp(r); // sA–pB
    const double Hsp_ba = ba.hopping_sp(r); // sB–pA → used as pA–sB magnitude
    const double Hpps = ab.hopping_pps(r);
    const double Hppp = ab.hopping_ppp(r);
    const double Sss = ab.overlap_ss(r);
    const double Ssp_ab = ab.overlap_sp(r);
    const double Ssp_ba = ba.overlap_sp(r);
    const double Spps = ab.overlap_pps(r);
    const double Sppp = ab.overlap_ppp(r);

    auto add_hs = [&](int i, int j, double hv, double sv) {
        h0(ia0 + i, ib0 + j) += hv;
        s(ia0 + i, ib0 + j) += sv;
    };

    add_hs(0, 0, Hss, Sss);

    if (a_.n_ao == 1 && b_.n_ao == 1)
        return;

    const double inv_r = 1.0 / r;
    const double l = dx * inv_r;
    const double m = dy * inv_r;
    const double n = dz * inv_r;

    if (a_.n_ao >= 1 && b_.n_ao == 4) {
        add_hs(0, 1, l * Hsp_ab, l * Ssp_ab);
        add_hs(0, 2, m * Hsp_ab, m * Ssp_ab);
        add_hs(0, 3, n * Hsp_ab, n * Ssp_ab);
    }
    if (a_.n_ao == 4 && b_.n_ao >= 1) {
        // pA–sB: DFTB+ takes spσ from the reverse ordered table with a bond-
        // direction sign flip (same convention as the legacy sketch path).
        add_hs(1, 0, -l * Hsp_ba, -l * Ssp_ba);
        add_hs(2, 0, -m * Hsp_ba, -m * Ssp_ba);
        add_hs(3, 0, -n * Hsp_ba, -n * Ssp_ba);
    }
    if (a_.n_ao == 4 && b_.n_ao == 4) {
        const double dir[3] = {l, m, n};
        for (int a = 0; a < 3; ++a) {
            for (int b = 0; b < 3; ++b) {
                const double d_ab = (a == b) ? 1.0 : 0.0;
                const double hv = dir[a] * dir[b] * (Hpps - Hppp) + d_ab * Hppp;
                const double sv = dir[a] * dir[b] * (Spps - Sppp) + d_ab * Sppp;
                add_hs(1 + a, 1 + b, hv, sv);
            }
        }
    }
    (void)sk_;
}

void MioSketchPair::set_block_derivative(double dx, double dy, double dz,
                                         double r,
                                         linalg::RealMatrix (&dh0)[3],
                                         linalg::RealMatrix (&ds)[3], int ia0,
                                         int ib0) const {
    if (r < 1e-12)
        return;
    const double inv_r = 1.0 / r;
    const double dir[3] = {dx * inv_r, dy * inv_r, dz * inv_r};

    const MioSkTable ab(a_.z, b_.z);
    const MioSkTable ba(b_.z, a_.z);

    const double Hsp_ab = ab.hopping_sp(r);
    const double Hsp_ba = ba.hopping_sp(r);
    const double Hpps = ab.hopping_pps(r);
    const double Hppp = ab.hopping_ppp(r);
    const double Ssp_ab = ab.overlap_sp(r);
    const double Ssp_ba = ba.overlap_sp(r);
    const double Spps = ab.overlap_pps(r);
    const double Sppp = ab.overlap_ppp(r);

    const double dHss = ab.d_hopping_ss_dr(r);
    const double dHsp_ab = ab.d_hopping_sp_dr(r);
    const double dHsp_ba = ba.d_hopping_sp_dr(r);
    const double dHpps = ab.d_hopping_pps_dr(r);
    const double dHppp = ab.d_hopping_ppp_dr(r);
    const double dSss = ab.d_overlap_ss_dr(r);
    const double dSsp_ab = ab.d_overlap_sp_dr(r);
    const double dSsp_ba = ba.d_overlap_sp_dr(r);
    const double dSpps = ab.d_overlap_pps_dr(r);
    const double dSppp = ab.d_overlap_ppp_dr(r);

    auto ddir = [&](int alpha, int k) {
        const double dab = (alpha == k) ? 1.0 : 0.0;
        return (dab - dir[alpha] * dir[k]) * inv_r;
    };

    for (int k = 0; k < 3; ++k) {
        const double dr_dk = dir[k];
        dh0[k](ia0, ib0) = dHss * dr_dk;
        ds[k](ia0, ib0) = dSss * dr_dk;

        if (a_.n_ao == 1 && b_.n_ao == 1)
            continue;

        if (a_.n_ao >= 1 && b_.n_ao == 4) {
            for (int a = 0; a < 3; ++a) {
                dh0[k](ia0, ib0 + 1 + a) =
                    ddir(a, k) * Hsp_ab + dir[a] * dHsp_ab * dr_dk;
                ds[k](ia0, ib0 + 1 + a) =
                    ddir(a, k) * Ssp_ab + dir[a] * dSsp_ab * dr_dk;
            }
        }
        if (a_.n_ao == 4 && b_.n_ao >= 1) {
            for (int a = 0; a < 3; ++a) {
                dh0[k](ia0 + 1 + a, ib0) =
                    -(ddir(a, k) * Hsp_ba + dir[a] * dHsp_ba * dr_dk);
                ds[k](ia0 + 1 + a, ib0) =
                    -(ddir(a, k) * Ssp_ba + dir[a] * dSsp_ba * dr_dk);
            }
        }
        if (a_.n_ao == 4 && b_.n_ao == 4) {
            const double dHsig = dHpps - dHppp;
            const double dSsig = dSpps - dSppp;
            const double Hsig = Hpps - Hppp;
            const double Ssig = Spps - Sppp;
            for (int a = 0; a < 3; ++a) {
                for (int b = 0; b < 3; ++b) {
                    const double dab = (a == b) ? 1.0 : 0.0;
                    dh0[k](ia0 + 1 + a, ib0 + 1 + b) =
                        (ddir(a, k) * dir[b] + dir[a] * ddir(b, k)) * Hsig
                        + dir[a] * dir[b] * dHsig * dr_dk + dab * dHppp * dr_dk;
                    ds[k](ia0 + 1 + a, ib0 + 1 + b) =
                        (ddir(a, k) * dir[b] + dir[a] * ddir(b, k)) * Ssig
                        + dir[a] * dir[b] * dSsig * dr_dk + dab * dSppp * dr_dk;
                }
            }
        }
    }
    (void)sk_;
}

} // namespace internal
} // namespace semiempirical
} // namespace cpu
} // namespace ATV
