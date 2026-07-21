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

} // namespace

D3Result dftd3_pbe(const std::vector<int>& Z,
                   const std::vector<std::array<double, 3>>& pos_bohr,
                   const double cell_bohr[3], bool periodic,
                   const D3Params& params, double rcut_bohr) {
    const int n = static_cast<int>(Z.size());
    D3Result res;
    res.forces.assign(n, {0.0, 0.0, 0.0});
    if (n == 0)
        return res;

    const double rcut = rcut_bohr;
    const double rcut2 = rcut * rcut;

    // Lattice-image half-counts covering rcut (orthorhombic Γ-point).
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

                        // CN-chain: dE/dCN_i and dE/dCN_j for this pair's C6/C8.
                        dEdCN[i] += -kS6 * fac * dc6a * fdab6 / r6
                                    - kS8 * fac * dc8a * fdab8 / r8;
                        dEdCN[j] += -kS6 * fac * dc6b * fdab6 / r6
                                    - kS8 * fac * dc8b * fdab8 / r8;
                    }
        }
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
        }
    }

    res.energy = energy;
    return res;
}

} // namespace gpw
} // namespace cpu
} // namespace ATV
