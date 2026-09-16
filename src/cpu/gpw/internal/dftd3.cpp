/// @file cpu/gpw/internal/dftd3.cpp
/// @brief PBE DFT-D3 (zero damping) energy + analytic forces (see header).
///
/// A direct transcription of CP2K's qs_dispersion_d3 / qs_dispersion_cnum for a
/// single (Γ-point, orthorhombic) cell: coordination numbers and the two-body
/// C6/C8 dispersion are summed over lattice images out to `rcut`, and the
/// analytic forces include both the explicit R-dependence of the damped
/// 1/R^n terms and the implicit dependence through the coordination numbers
/// (the dC6/dCN chain). Energies are in Hartree, forces in Hartree/Bohr.

#include "atomiverse/cpu/gpw/internal/dftd3.h"

#include <algorithm>
#include <cmath>

#include "atomiverse/cpu/gpw/internal/periodic.h"

namespace ATV {
namespace cpu {
namespace gpw {

namespace {

// --- PBE zero-damping parameters (CP2K qs_dispersion_utils.F / pairpot.F) ---
constexpr double kS6 = 1.0;
constexpr double kS8 = 0.722;
constexpr double kSr6 = 1.217;
constexpr double kSr8 = 1.0;
constexpr double kAlp6 = 14.0;
constexpr double kAlp8 = 16.0; // alp6 + 2
constexpr double kK1 = 16.0;
constexpr double kK3 = -4.0;
constexpr double kEpsCn = 1.0e-6;

/// @brief One stored CN-derivative contribution dCN_i/dR towards neighbour k.
/// Mirrors CP2K's dcnum_type entries: `dval` = dCN/dR along the unit vector to
/// neighbour `k`, `rik` = R_i - R_k (the displacement used for the direction).
struct DCN {
    int k;                     ///< neighbour atom index.
    double dval;               ///< dCN/dr_ik (CP2K dcnab).
    std::array<double, 3> rik; ///< R_i - R_k [Bohr].
};

/// @brief D3 coordination-number counting function and its radial derivative.
///
/// CP2K cnparam_d3: rr = rcovab/rab, ee = exp(-k1(rr-1)),
/// fz = ½(1 - tanh(rab - 2 rcovab)) (a second cutoff for long-range smoothness),
/// cn = ff·fz with ff = 1/(1+ee). dcn is d(cn)/d(rab).
/// @param rab Pair distance [Bohr].
/// @param rcovab Combined (k2-folded) covalent radius rcov_A + rcov_B [Bohr].
/// @param cnab Out: contribution to the coordination number.
/// @param dcnab Out: d(cnab)/d(rab).
void cnparam_d3(double rab, double rcovab, double& cnab, double& dcnab) {
    const double rr = rcovab / rab;
    const double ee = std::exp(-kK1 * (rr - 1.0));
    const double th = std::tanh(rab - 2.0 * rcovab);
    const double fz = 0.5 * (1.0 - th);
    const double dfz = 0.5 * (th * th - 1.0);
    const double ff = 1.0 / (1.0 + ee);
    cnab = ff * fz;
    dcnab = -fz * ff * ff * kK1 * rr / rab * ee + ff * dfz;
}

/// @brief Zero-damping factor f_{d,n}(rab) and its derivative (CP2K damping_d3).
///
/// f = fab·fcc, fab = 1/(1+6(rab/(srn·rcutab))^{-alpn}); fcc is a fifth-order
/// smoothstep that takes f from 1 to 0 over [rcut-1, rcut].
/// @param rab Pair distance [Bohr].
/// @param rcutab Damping radius R0_AB [Bohr].
/// @param srn Range scaling s_{r,n}.
/// @param alpn Damping exponent α_n.
/// @param rcut Hard cutoff [Bohr].
/// @param fdab Out: damping factor.
/// @param dfdab Out: d(fdab)/d(rab).
void damping_d3(double rab, double rcutab, double srn, double alpn, double rcut,
                double& fdab, double& dfdab) {
    const double rl = rcut - 1.0;
    const double ru = rcut;
    double fcc, dfcc;
    if (rab >= ru) {
        fcc = 0.0;
        dfcc = 0.0;
    } else if (rab <= rl) {
        fcc = 1.0;
        dfcc = 0.0;
    } else {
        const double z = rab * rab - rl * rl;
        const double dz = 2.0 * rab;
        const double zz = z * z * z;
        const double d = ru * ru - rl * rl;
        const double dd = d * d * d;
        const double a = -10.0 / dd;
        const double b = 15.0 / (dd * d);
        const double c = -6.0 / (dd * d * d);
        fcc = 1.0 + zz * (a + b * z + c * z * z);
        dfcc = zz * dz / z * (3.0 * a + 4.0 * b * z + 5.0 * c * z * z);
    }
    const double rr = 6.0 * std::pow(rab / (srn * rcutab), -alpn);
    const double fab = 1.0 / (1.0 + rr);
    const double dfab = fab * fab * rr * alpn / rab;
    fdab = fab * fcc;
    dfdab = dfab * fcc + fab * dfcc;
}

/// @brief CN-interpolated C6^AB and its CN-derivatives (CP2K getc6).
///
/// Gaussian-weighted average over reference states:
/// C6 = Σ_ij C6ref_ij W_ij / Σ_ij W_ij, W_ij = exp(k3((cnref_i-cnA)²+(cnref_j-cnB)²)).
/// @param refs Reference states for the ordered pair (A,B).
/// @param nci Coordination number of A.
/// @param ncj Coordination number of B.
/// @param c6 Out: interpolated C6 [a.u.].
/// @param dc6a Out: dC6/dCN_A.
/// @param dc6b Out: dC6/dCN_B.
void getc6(const std::vector<D3C6Ref>& refs, double nci, double ncj, double& c6,
           double& dc6a, double& dc6b) {
    double rsave = 1.0e99, c6mem = -1.0e99;
    double rsum = 0.0, csum = 0.0, dza = 0.0, dwa = 0.0, dzb = 0.0, dwb = 0.0;
    double last_c6 = 0.0;
    for (const auto& rf : refs) {
        const double cr6 = rf.c6;
        if (cr6 <= 0.0)
            continue;
        last_c6 = cr6;
        const double cn1 = rf.cn_a;
        const double cn2 = rf.cn_b;
        const double r = (cn1 - nci) * (cn1 - nci) + (cn2 - ncj) * (cn2 - ncj);
        if (r < rsave) {
            rsave = r;
            c6mem = cr6;
        }
        const double tmp1 = std::exp(kK3 * r);
        const double dtmpa = -2.0 * kK3 * (cn1 - nci) * tmp1;
        const double dtmpb = -2.0 * kK3 * (cn2 - ncj) * tmp1;
        rsum += tmp1;
        csum += tmp1 * cr6;
        dza += dtmpa * cr6;
        dwa += dtmpa;
        dzb += dtmpb * cr6;
        dwb += dtmpb;
    }
    if (last_c6 == 0.0)
        c6mem = 0.0;
    if (rsum > 1.0e-66) {
        c6 = csum / rsum;
        dc6a = (dza - c6 * dwa) / rsum;
        dc6b = (dzb - c6 * dwb) / rsum;
    } else {
        c6 = c6mem;
        dc6a = 0.0;
        dc6b = 0.0;
    }
}

/// @brief Axilrod–Teller–Muto three-body C9 (CP2K qs_dispersion_d3 doabc).
///
/// Adds −e9 to `energy` (e9 defined as in CP2K), explicit-R forces, and
/// CN-chain contributions into `dEdCN`. s9 = s6 for D3. Pair distances for
/// each leg must be < rcut; geometric mean rabc damps with α=16, sr=4/3.
void accumulate_atm_c9(const std::vector<int>& Z,
                       const std::vector<std::array<double, 3>>& pos_raw,
                       const double cell_bohr[3], bool periodic,
                       const D3Params& params, const std::vector<double>& cn,
                       double rcut, double s9, double& energy,
                       std::vector<std::array<double, 3>>& forces,
                       std::vector<double>& dEdCN) {
    const int n = static_cast<int>(Z.size());
    if (n < 3 && !periodic)
        return;
    const double rcut2 = rcut * rcut;
    int m[3] = {0, 0, 0};
    image_counts(cell_bohr, periodic, rcut, m);

    // CP2K evaluates doabc on pbc()-wrapped coordinates (ra = pbc(r), Fortran
    // ANINT = round-half-away-from-zero). The triple legs are gated by HARD
    // r2 < rcut2 cutoffs over a finite +-m image window, so the included
    // boundary set — and with it the per-atom force distribution — depends on
    // where each atom sits inside the cell. Raw (unwrapped) coordinates keep
    // the energy (invariant to 1e-19 here) but redistribute ~1e-7 Ha/Bohr of
    // boundary-triple force between translation-equivalent atoms: on the
    // periodic 6 A water-dimer twin the two molecules' ATM forces came out
    // exactly swapped relative to CP2K. Wrap first; forces stay attributed by
    // atom index.
    std::vector<std::array<double, 3>> pos_bohr(pos_raw.begin(),
                                                pos_raw.end());
    if (periodic)
        for (auto& p : pos_bohr)
            for (int d = 0; d < 3; ++d)
                if (cell_bohr[d] > 0.0)
                    p[d] -= cell_bohr[d] * std::round(p[d] / cell_bohr[d]);

    // CP2K-style enumeration over AB then C in cell_c: the AB pairs replicate
    // the half neighbor list sab_vdw exactly as the two-body loop does —
    // iatom <= jatom with ONE orientation per image for i < j, and both +T/-T
    // images for the self-pair i == j (compensated by fac = 0.5 below, CP2K
    // qs_dispersion_d3.F). Enumerating off-home i != j pairs in both
    // orientations instead counts every CASE 3 triple class twice (6 anchored
    // representatives x fac0 = 1/3): on the periodic 6 A water-dimer twin
    // that inflated the ATM energy from +7.85e-7 to +6.25e-6 Ha — caught by
    // the regressions/gpw/gpw-d3-atm-forces CP2K pin.
    for (int iatom = 0; iatom < n; ++iatom) {
        const int za = Z[iatom];
        for (int jatom = iatom; jatom < n; ++jatom) {
            const int zb = Z[jatom];
            const auto& refs_ab = params.c6ab[za][zb];
            if (refs_ab.empty())
                continue;
            for (int tbx = -m[0]; tbx <= m[0]; ++tbx)
                for (int tby = -m[1]; tby <= m[1]; ++tby)
                    for (int tbz = -m[2]; tbz <= m[2]; ++tbz) {
                        if (jatom == iatom && tbx == 0 && tby == 0 && tbz == 0)
                            continue;
                        const double rax = pos_bohr[iatom][0];
                        const double ray = pos_bohr[iatom][1];
                        const double raz = pos_bohr[iatom][2];
                        const double rbx =
                            pos_bohr[jatom][0] + tbx * cell_bohr[0];
                        const double rby =
                            pos_bohr[jatom][1] + tby * cell_bohr[1];
                        const double rbz =
                            pos_bohr[jatom][2] + tbz * cell_bohr[2];
                        const double dxab = rax - rbx;
                        const double dyab = ray - rby;
                        const double dzab = raz - rbz;
                        const double r2ab =
                            dxab * dxab + dyab * dyab + dzab * dzab;
                        if (r2ab >= rcut2 || r2ab <= 1.0e-12)
                            continue;
                        const bool is000 = (tbx == 0 && tby == 0 && tbz == 0);

                        for (int tcx = -m[0]; tcx <= m[0]; ++tcx)
                            for (int tcy = -m[1]; tcy <= m[1]; ++tcy)
                                for (int tcz = -m[2]; tcz <= m[2]; ++tcz) {
                                    const bool c000 =
                                        (tcx == 0 && tcy == 0 && tcz == 0);
                                    int kstart = 0;
                                    double fac0 = 1.0;
                                    if (is000 && c000) {
                                        // CASE 1: ordered triples in (000)
                                        kstart = std::max(jatom, iatom) + 1;
                                        fac0 = 1.0;
                                    } else if (is000) {
                                        // CASE 2: AB in (000), C elsewhere
                                        kstart = 0;
                                        fac0 = 1.0;
                                    } else {
                                        // CASE 3: A in (000), B and C off-home
                                        if (tcx == tbx && tcy == tby
                                            && tcz == tbz)
                                            continue; // BC same cell → case 2
                                        if (c000)
                                            continue;
                                        kstart = 0;
                                        fac0 = 1.0 / 3.0;
                                    }
                                    for (int katom = kstart; katom < n;
                                         ++katom) {
                                        const int zc = Z[katom];
                                        const auto& refs_bc =
                                            params.c6ab[zb][zc];
                                        const auto& refs_ca =
                                            params.c6ab[zc][za];
                                        if (refs_bc.empty() || refs_ca.empty())
                                            continue;
                                        const double rcx = pos_bohr[katom][0]
                                                           + tcx * cell_bohr[0];
                                        const double rcy = pos_bohr[katom][1]
                                                           + tcy * cell_bohr[1];
                                        const double rcz = pos_bohr[katom][2]
                                                           + tcz * cell_bohr[2];
                                        const double dxbc = rbx - rcx;
                                        const double dybc = rby - rcy;
                                        const double dzbc = rbz - rcz;
                                        const double r2bc = dxbc * dxbc
                                                            + dybc * dybc
                                                            + dzbc * dzbc;
                                        if (r2bc >= rcut2)
                                            continue;
                                        const double dxca = rcx - rax;
                                        const double dyca = rcy - ray;
                                        const double dzca = rcz - raz;
                                        const double r2ca = dxca * dxca
                                                            + dyca * dyca
                                                            + dzca * dzca;
                                        if (r2ca >= rcut2)
                                            continue;
                                        const double r2abc = r2ab * r2bc * r2ca;
                                        if (r2abc <= 0.001)
                                            continue;
                                        // Empirical scaling gate (CP2K)
                                        if (r2abc
                                            > 0.01 * rcut2 * rcut2 * rcut2)
                                            continue;

                                        double fac = 1.0;
                                        if (iatom == jatom || iatom == katom
                                            || jatom == katom)
                                            fac = 0.5;
                                        if (iatom == jatom && iatom == katom)
                                            fac = 1.0 / 3.0;
                                        fac *= fac0;

                                        double cc6ab, dcc6aba, dcc6abb;
                                        double cc6bc, dcc6bcb, dcc6bcc;
                                        double cc6ca, dcc6cac, dcc6caa;
                                        getc6(refs_ab, cn[iatom], cn[jatom],
                                              cc6ab, dcc6aba, dcc6abb);
                                        getc6(refs_bc, cn[jatom], cn[katom],
                                              cc6bc, dcc6bcb, dcc6bcc);
                                        getc6(refs_ca, cn[katom], cn[iatom],
                                              cc6ca, dcc6cac, dcc6caa);
                                        if (cc6ab <= 0.0 || cc6bc <= 0.0
                                            || cc6ca <= 0.0)
                                            continue;
                                        const double c9 =
                                            -std::sqrt(cc6ab * cc6bc * cc6ca);
                                        const double rabc =
                                            std::pow(r2abc, 1.0 / 6.0);
                                        const double r0 =
                                            std::cbrt(params.r0ab[za][zb]
                                                      * params.r0ab[zb][zc]
                                                      * params.r0ab[zc][za]);
                                        double fdabc, dfdabc;
                                        // α=16 (paper; CP2K post-2017 fix)
                                        damping_d3(rabc, r0, 4.0 / 3.0, 16.0,
                                                   rcut, fdabc, dfdabc);
                                        const double s1 = r2ab + r2bc - r2ca;
                                        const double s2 = r2ab + r2ca - r2bc;
                                        const double s3 = r2ca + r2bc - r2ab;
                                        const double ang =
                                            0.375 * s1 * s2 * s3 / r2abc + 1.0;
                                        const double r2abc15 =
                                            std::pow(r2abc, 1.5);
                                        const double e9 = s9 * fac * fdabc * c9
                                                          * ang / r2abc15;
                                        energy -= e9;

                                        // Explicit geometry forces (CP2K).
                                        // rab = rb−ra, rbc = rc−rb, rca = ra−rc
                                        const double rab[3] = {
                                            rbx - rax, rby - ray, rbz - raz};
                                        const double rbc[3] = {
                                            rcx - rbx, rcy - rby, rcz - rbz};
                                        const double rca[3] = {
                                            rax - rcx, ray - rcy, raz - rcz};
                                        double de91 = s9 * fac * dfdabc * c9
                                                      * ang / r2abc15;
                                        de91 = -de91 / 3.0 * rabc + 3.0 * e9;
                                        const double dea =
                                            s9 * fac * fdabc * c9
                                            / std::pow(r2abc, 2.5) * 0.75;
                                        const double s123 = s1 * s2 * s3;
                                        // CP2K uses rab=rb−ra with force_a−=fdij
                                        // on energy E=−e9. Our two-body path uses
                                        // r=ra−rb; for ATM we keep CP2K's rleg
                                        // orientation but apply the opposite
                                        // force_a/force_b assignment so F=−∇E
                                        // matches finite differences of E.
                                        auto add_leg = [&](const double rleg[3],
                                                           double r2leg,
                                                           double sm, int ia,
                                                           int ib) {
                                            for (int d = 0; d < 3; ++d) {
                                                const double fdij =
                                                    de91 * rleg[d] / r2leg
                                                    + dea * s123 * rleg[d]
                                                          / r2leg
                                                    - dea * sm * rleg[d];
                                                forces[ia][d] += fdij;
                                                forces[ib][d] -= fdij;
                                            }
                                        };
                                        // AB: sm = s2*s3 + s1*s3 − s1*s2
                                        add_leg(rab, r2ab,
                                                s2 * s3 + s1 * s3 - s1 * s2,
                                                iatom, jatom);
                                        // BC: sm = s2*s3 − s1*s3 + s1*s2
                                        add_leg(rbc, r2bc,
                                                s2 * s3 - s1 * s3 + s1 * s2,
                                                jatom, katom);
                                        // CA: sm = −s2*s3 + s1*s3 + s1*s2
                                        add_leg(rca, r2ca,
                                                -s2 * s3 + s1 * s3 + s1 * s2,
                                                katom, iatom);

                                        // CN-chain: dEdCN += ∂E/∂CN with E=−e9
                                        // ∂E/∂cc6ab = −0.5 e9 / cc6ab
                                        dEdCN[iatom] +=
                                            (-0.5 * e9 / cc6ab) * dcc6aba
                                            + (-0.5 * e9 / cc6ca) * dcc6caa;
                                        dEdCN[jatom] +=
                                            (-0.5 * e9 / cc6ab) * dcc6abb
                                            + (-0.5 * e9 / cc6bc) * dcc6bcb;
                                        dEdCN[katom] +=
                                            (-0.5 * e9 / cc6bc) * dcc6bcc
                                            + (-0.5 * e9 / cc6ca) * dcc6cac;
                                    }
                                }
                    }
        }
    }
}

} // namespace

D3Result dftd3_pbe(const std::vector<int>& Z,
                   const std::vector<std::array<double, 3>>& pos_bohr,
                   const double cell_bohr[3], bool periodic,
                   const D3Params& params, double rcut_bohr,
                   bool calculate_c9) {
    const int n = static_cast<int>(Z.size());
    D3Result res;
    res.forces.assign(n, {0.0, 0.0, 0.0});
    if (n == 0)
        return res;

    const double rcut = rcut_bohr;
    const double rcut2 = rcut * rcut;

    // Lattice-image half-counts covering rcut (orthorhombic Γ-point).
    // Zero-damping CP2K pin uses image_counts (not pair_image_counts).
    int m[3] = {0, 0, 0};
    image_counts(cell_bohr, periodic, rcut, m);

    // ---- Coordination numbers (and dCN neighbour lists for forces) ----
    std::vector<double> cn(n, 0.0);
    std::vector<std::vector<DCN>> dcn(n);
    for (int i = 0; i < n; ++i) {
        const int zi = Z[i];
        for (int j = 0; j < n; ++j) {
            const int zj = Z[j];
            const double rcovab = params.rcov[zi] + params.rcov[zj];
            for (int tx = -m[0]; tx <= m[0]; ++tx)
                for (int ty = -m[1]; ty <= m[1]; ++ty)
                    for (int tz = -m[2]; tz <= m[2]; ++tz) {
                        if (j == i && tx == 0 && ty == 0 && tz == 0)
                            continue;
                        const double dx =
                            pos_bohr[i][0] - pos_bohr[j][0] - tx * cell_bohr[0];
                        const double dy =
                            pos_bohr[i][1] - pos_bohr[j][1] - ty * cell_bohr[1];
                        const double dz =
                            pos_bohr[i][2] - pos_bohr[j][2] - tz * cell_bohr[2];
                        const double r2 = dx * dx + dy * dy + dz * dz;
                        if (r2 > rcut2 || r2 < 1.0e-12)
                            continue;
                        const double rab = std::sqrt(r2);
                        double cnab, dcnab;
                        cnparam_d3(rab, rcovab, cnab, dcnab);
                        if (cnab <= kEpsCn)
                            continue;
                        cn[i] += cnab;
                        dcn[i].push_back({j, dcnab, {dx, dy, dz}});
                    }
        }
    }

    // ---- Two-body C6 / C8 dispersion (energy + forces) ----
    // Accumulate dE/dCN_i; the CN-chain force is applied afterwards so the
    // total CN-gradient per atom multiplies its dCN neighbour list once.
    std::vector<double> dEdCN(n, 0.0);
    double energy = 0.0;

    for (int i = 0; i < n; ++i) {
        const int za = Z[i];
        for (int j = i; j < n; ++j) {
            const int zb = Z[j];
            const auto& refs = params.c6ab[za][zb];
            if (refs.empty())
                continue;
            double c6, dc6a, dc6b;
            getc6(refs, cn[i], cn[j], c6, dc6a, dc6b);
            const double r2r4ab = params.r2r4[za] * params.r2r4[zb];
            const double c8 = 3.0 * c6 * r2r4ab;
            const double dc8a = 3.0 * dc6a * r2r4ab;
            const double dc8b = 3.0 * dc6b * r2r4ab;
            const double r0ab = params.r0ab[za][zb];

            for (int tx = -m[0]; tx <= m[0]; ++tx)
                for (int ty = -m[1]; ty <= m[1]; ++ty)
                    for (int tz = -m[2]; tz <= m[2]; ++tz) {
                        if (j == i && tx == 0 && ty == 0 && tz == 0)
                            continue;
                        // ordered self-pair images counted once (fac=0.5).
                        const double fac = (i == j) ? 0.5 : 1.0;
                        // rij points from j(+T) to i  (CP2K r = r_i - r_j).
                        const double rx =
                            pos_bohr[i][0] - pos_bohr[j][0] - tx * cell_bohr[0];
                        const double ry =
                            pos_bohr[i][1] - pos_bohr[j][1] - ty * cell_bohr[1];
                        const double rz =
                            pos_bohr[i][2] - pos_bohr[j][2] - tz * cell_bohr[2];
                        const double r2 = rx * rx + ry * ry + rz * rz;
                        if (r2 > rcut2 || r2 <= 1.0e-6)
                            continue;
                        const double dr = std::sqrt(r2);
                        const double r6 = r2 * r2 * r2;
                        const double r8 = r6 * r2;

                        double fdab6, dfdab6, fdab8, dfdab8;
                        damping_d3(dr, r0ab, kSr6, kAlp6, rcut, fdab6, dfdab6);
                        damping_d3(dr, r0ab, kSr8, kAlp8, rcut, fdab8, dfdab8);
                        const double e6 = kS6 * fac * c6 * fdab6 / r6;
                        const double e8 = kS8 * fac * c8 * fdab8 / r8;
                        energy -= e6 + e8;

                        // explicit-R force (CP2K de6/de8); F on i from this pair.
                        const double de6 =
                            -kS6 * c6 / r6 * (dfdab6 - 6.0 * fdab6 / dr);
                        const double de8 =
                            -kS8 * c8 / r8 * (dfdab8 - 8.0 * fdab8 / dr);
                        const double pref = (de6 + de8) * fac / dr;
                        const double fx = pref * rx;
                        const double fy = pref * ry;
                        const double fz = pref * rz;
                        res.forces[i][0] -= fx;
                        res.forces[i][1] -= fy;
                        res.forces[i][2] -= fz;
                        res.forces[j][0] += fx;
                        res.forces[j][1] += fy;
                        res.forces[j][2] += fz;
                        res.virial[0] -= pref * rx * rx;
                        res.virial[1] -= pref * ry * ry;
                        res.virial[2] -= pref * rz * rz;
                        res.virial[3] -= 2.0 * pref * rx * ry;
                        res.virial[4] -= 2.0 * pref * rx * rz;
                        res.virial[5] -= 2.0 * pref * ry * rz;

                        // CN-chain: dE/dCN_i and dE/dCN_j for this pair's C6/C8.
                        dEdCN[i] += -kS6 * fac * dc6a * fdab6 / r6
                                    - kS8 * fac * dc8a * fdab8 / r8;
                        dEdCN[j] += -kS6 * fac * dc6b * fdab6 / r6
                                    - kS8 * fac * dc8b * fdab8 / r8;
                    }
        }
    }

    // ---- Three-body ATM / C9 (optional; s9 = s6 for D3) ----
    if (calculate_c9) {
        accumulate_atm_c9(Z, pos_bohr, cell_bohr, periodic, params, cn, rcut,
                          /*s9=*/kS6, energy, res.forces, dEdCN);
    }

    // ---- Forces from the r-dependence of the coordination numbers ----
    // For each atom i, dE/dR via CN flows through every dCN_i/dR neighbour k:
    //   F_i -= dEdCN[i] · dCN_i/dr · r_ik/|r_ik|,  F_k += same.
    for (int i = 0; i < n; ++i) {
        for (const auto& d : dcn[i]) {
            const double drk =
                std::sqrt(d.rik[0] * d.rik[0] + d.rik[1] * d.rik[1]
                          + d.rik[2] * d.rik[2]);
            const double s = dEdCN[i] * d.dval / drk;
            const double fx = s * d.rik[0];
            const double fy = s * d.rik[1];
            const double fz = s * d.rik[2];
            res.forces[i][0] -= fx;
            res.forces[i][1] -= fy;
            res.forces[i][2] -= fz;
            res.forces[d.k][0] += fx;
            res.forces[d.k][1] += fy;
            res.forces[d.k][2] += fz;
            res.virial[0] -= s * d.rik[0] * d.rik[0];
            res.virial[1] -= s * d.rik[1] * d.rik[1];
            res.virial[2] -= s * d.rik[2] * d.rik[2];
            res.virial[3] -= 2.0 * s * d.rik[0] * d.rik[1];
            res.virial[4] -= 2.0 * s * d.rik[0] * d.rik[2];
            res.virial[5] -= 2.0 * s * d.rik[1] * d.rik[2];
        }
    }

    res.energy = energy;
    return res;
}

// ---------------------------------------------------------------------------
// Becke–Johnson (rational) damping — molecular / PySCF path
// ---------------------------------------------------------------------------

namespace {

/// @brief Original Grimme D3 coordination-number increment (no CP2K tanh).
///
/// cn = 1/(1+exp(-k1*((rcovA+rcovB)/r - 1))), k1 = 16.
/// @param rab Pair distance [Bohr].
/// @param rcovab Combined covalent radius (k2-folded) [Bohr].
/// @param cnab Out: CN contribution.
/// @param dcnab Out: d(cnab)/d(rab).
void cnparam_grimme(double rab, double rcovab, double& cnab, double& dcnab) {
    const double rr = rcovab / rab;
    const double ee = std::exp(-kK1 * (rr - 1.0));
    const double ff = 1.0 / (1.0 + ee);
    cnab = ff;
    dcnab = -ff * ff * kK1 * rr / rab * ee;
}

} // namespace

D3Result dftd3_pbe_bj(const std::vector<int>& Z,
                      const std::vector<std::array<double, 3>>& pos_bohr,
                      const double cell_bohr[3], bool periodic,
                      const D3Params& params, double rcut_bohr, double s6,
                      double s8, double a1, double a2, bool cp2k_cn,
                      bool calculate_c9) {
    const int n = static_cast<int>(Z.size());
    D3Result res;
    res.forces.assign(n, {0.0, 0.0, 0.0});
    if (n == 0)
        return res;

    const double rcut = rcut_bohr;
    const double rcut2 = rcut * rcut;

    // Pair lattice sum: extra image shell so |R_i−R_j−T| ≤ rcut is complete.
    int m[3] = {0, 0, 0};
    pair_image_counts(cell_bohr, periodic, rcut, m);

    // Coordination numbers: Grimme (molecular/PySCF) or CP2K tanh (GPW).
    std::vector<double> cn(n, 0.0);
    std::vector<std::vector<DCN>> dcn(n);
    for (int i = 0; i < n; ++i) {
        const int zi = Z[i];
        for (int j = 0; j < n; ++j) {
            const int zj = Z[j];
            const double rcovab = params.rcov[zi] + params.rcov[zj];
            for (int tx = -m[0]; tx <= m[0]; ++tx)
                for (int ty = -m[1]; ty <= m[1]; ++ty)
                    for (int tz = -m[2]; tz <= m[2]; ++tz) {
                        if (j == i && tx == 0 && ty == 0 && tz == 0)
                            continue;
                        const double dx =
                            pos_bohr[i][0] - pos_bohr[j][0] - tx * cell_bohr[0];
                        const double dy =
                            pos_bohr[i][1] - pos_bohr[j][1] - ty * cell_bohr[1];
                        const double dz =
                            pos_bohr[i][2] - pos_bohr[j][2] - tz * cell_bohr[2];
                        const double r2 = dx * dx + dy * dy + dz * dz;
                        if (r2 > rcut2 || r2 < 1.0e-12)
                            continue;
                        const double rab = std::sqrt(r2);
                        double cnab, dcnab;
                        if (cp2k_cn)
                            cnparam_d3(rab, rcovab, cnab, dcnab);
                        else
                            cnparam_grimme(rab, rcovab, cnab, dcnab);
                        if (cnab <= kEpsCn)
                            continue;
                        cn[i] += cnab;
                        dcn[i].push_back({j, dcnab, {dx, dy, dz}});
                    }
        }
    }

    std::vector<double> dEdCN(n, 0.0);
    double energy = 0.0;

    for (int i = 0; i < n; ++i) {
        const int za = Z[i];
        for (int j = i; j < n; ++j) {
            const int zb = Z[j];
            const auto& refs = params.c6ab[za][zb];
            if (refs.empty())
                continue;
            double c6, dc6a, dc6b;
            getc6(refs, cn[i], cn[j], c6, dc6a, dc6b);
            if (c6 <= 0.0)
                continue;
            const double r2r4ab = params.r2r4[za] * params.r2r4[zb];
            const double c8 = 3.0 * c6 * r2r4ab;
            const double dc8a = 3.0 * dc6a * r2r4ab;
            const double dc8b = 3.0 * dc6b * r2r4ab;
            // BJ damping radius f = a1 √(C8/C6) + a2  (geometry-independent
            // for fixed CN; CN-derivatives of f enter via dC6/dCN, dC8/dCN).
            const double r0 = a1 * std::sqrt(c8 / c6) + a2;
            const double r0_2 = r0 * r0;
            const double r0_6 = r0_2 * r0_2 * r0_2;
            const double r0_8 = r0_6 * r0_2;

            for (int tx = -m[0]; tx <= m[0]; ++tx)
                for (int ty = -m[1]; ty <= m[1]; ++ty)
                    for (int tz = -m[2]; tz <= m[2]; ++tz) {
                        if (j == i && tx == 0 && ty == 0 && tz == 0)
                            continue;
                        const double fac = (i == j) ? 0.5 : 1.0;
                        const double rx =
                            pos_bohr[i][0] - pos_bohr[j][0] - tx * cell_bohr[0];
                        const double ry =
                            pos_bohr[i][1] - pos_bohr[j][1] - ty * cell_bohr[1];
                        const double rz =
                            pos_bohr[i][2] - pos_bohr[j][2] - tz * cell_bohr[2];
                        const double r2 = rx * rx + ry * ry + rz * rz;
                        if (r2 > rcut2 || r2 <= 1.0e-6)
                            continue;
                        const double dr = std::sqrt(r2);
                        const double r6 = r2 * r2 * r2;
                        const double r8 = r6 * r2;

                        const double den6 = r6 + r0_6;
                        const double den8 = r8 + r0_8;
                        const double e6 = s6 * fac * c6 / den6;
                        const double e8 = s8 * fac * c8 / den8;
                        energy -= e6 + e8;

                        // E = −s6 fac C6/den6 − s8 fac C8/den8
                        // ∂E/∂R = s6 fac C6 · 6 R^5 / den6² + s8 fac C8 · 8 R^7 / den8²
                        //       = s6 fac C6 · (6 r6/R) / den6² + …
                        // F_i = −∇_i E = −(∂E/∂R) r̂ with r̂ = (R_i−R_j)/R
                        // Store as F_i -= pref * r_vec with pref = (∂E/∂R)/R.
                        const double dE_dR =
                            s6 * fac * c6 * (6.0 * r6 / dr) / (den6 * den6)
                            + s8 * fac * c8 * (8.0 * r8 / dr) / (den8 * den8);
                        const double pref = dE_dR / dr;
                        const double fx = pref * rx;
                        const double fy = pref * ry;
                        const double fz = pref * rz;
                        res.forces[i][0] -= fx;
                        res.forces[i][1] -= fy;
                        res.forces[i][2] -= fz;
                        res.forces[j][0] += fx;
                        res.forces[j][1] += fy;
                        res.forces[j][2] += fz;
                        // Pair virial with image vector r (F_i = −pref r):
                        // matches Σ_a r_a ⊗ F_a for molecular pairs and the
                        // image form for PBC. Off-diag doubled like DFTB2.
                        res.virial[0] -= pref * rx * rx;
                        res.virial[1] -= pref * ry * ry;
                        res.virial[2] -= pref * rz * rz;
                        res.virial[3] -= 2.0 * pref * rx * ry;
                        res.virial[4] -= 2.0 * pref * rx * rz;
                        res.virial[5] -= 2.0 * pref * ry * rz;

                        // CN-chain: ∂E/∂CN through C6/C8 (f also depends on
                        // √(C8/C6); include d f / d C for forces accuracy).
                        // E_pair = −s6 C6/den6 − s8 C8/den8
                        // ∂E/∂C6 = −s6/den6 + s6 C6/den6² · ∂(r0^6)/∂C6
                        //        − s8 C8/den8² · ∂(r0^8)/∂C6  (and C8=3 C6 r2r4)
                        // With r0 = a1 √(C8/C6)+a2 = a1√(3 r2r4)+a2 independent
                        // of C6 scale (C8/C6 = 3 r2r4 constant w.r.t C6), so
                        // r0 is CN-dependent only through the interpolated ratio
                        // when dc6/dcn differ — use finite product rule on e.
                        // For energy-only consumers this is unused; for forces:
                        dEdCN[i] +=
                            -s6 * fac * dc6a / den6 - s8 * fac * dc8a / den8;
                        dEdCN[j] +=
                            -s6 * fac * dc6b / den6 - s8 * fac * dc8b / den8;
                    }
        }
    }

    if (calculate_c9) {
        // D3 ATM uses s9 = s6 (CP2K qs_dispersion_d3), independent of BJ s8.
        accumulate_atm_c9(Z, pos_bohr, cell_bohr, periodic, params, cn, rcut,
                          /*s9=*/s6, energy, res.forces, dEdCN);
    }

    for (int i = 0; i < n; ++i) {
        for (const auto& d : dcn[i]) {
            const double drk =
                std::sqrt(d.rik[0] * d.rik[0] + d.rik[1] * d.rik[1]
                          + d.rik[2] * d.rik[2]);
            const double s = dEdCN[i] * d.dval / drk;
            const double fx = s * d.rik[0];
            const double fy = s * d.rik[1];
            const double fz = s * d.rik[2];
            res.forces[i][0] -= fx;
            res.forces[i][1] -= fy;
            res.forces[i][2] -= fz;
            res.forces[d.k][0] += fx;
            res.forces[d.k][1] += fy;
            res.forces[d.k][2] += fz;
            // CN-chain forces also carry a pair virial on rik.
            res.virial[0] -= s * d.rik[0] * d.rik[0];
            res.virial[1] -= s * d.rik[1] * d.rik[1];
            res.virial[2] -= s * d.rik[2] * d.rik[2];
            res.virial[3] -= 2.0 * s * d.rik[0] * d.rik[1];
            res.virial[4] -= 2.0 * s * d.rik[0] * d.rik[2];
            res.virial[5] -= 2.0 * s * d.rik[1] * d.rik[2];
        }
    }

    res.energy = energy;
    return res;
}

} // namespace gpw
} // namespace cpu
} // namespace ATV
