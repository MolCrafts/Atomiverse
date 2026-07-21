/// @file src/cpu/semiempirical/nddo_two_electron.cpp
/// @brief AM1 reference reppd/rotatd/w2mat s/p two-electron integral port.

#include "atomiverse/cpu/semiempirical/nddo_two_electron.h"

#include <array>
#include <cmath>
#include <stdexcept>

#include "atomiverse/cpu/semiempirical/am1_parameters.h"

namespace ATV {
namespace cpu {
namespace semiempirical {

namespace {

constexpr double kA0 = 0.529177210903;
constexpr double kEv = 27.211386245988;

struct PairRotation {
    std::array<std::array<double, 3>, 3> sp{};
    std::array<std::array<std::array<double, 3>, 3>, 6> pp{};
};

std::array<double, 3> atom_xyz(const StorageReal* sp_x, const StorageReal* sp_y,
                               const StorageReal* sp_z, int atom) {
    return {static_cast<double>(sp_x[atom]), static_cast<double>(sp_y[atom]),
            static_cast<double>(sp_z[atom])};
}

double distance(const std::array<double, 3>& a,
                const std::array<double, 3>& b) {
    const double dx = a[0] - b[0];
    const double dy = a[1] - b[1];
    const double dz = a[2] - b[2];
    return std::sqrt(dx * dx + dy * dy + dz * dz);
}

PairRotation pair_rotation(const std::array<double, 3>& xi,
                           const std::array<double, 3>& xj) {
    constexpr double kSmall = 1.0e-7;

    const double x11 = xj[0] - xi[0];
    const double x22 = xj[1] - xi[1];
    const double x33 = xj[2] - xi[2];
    const double b = x11 * x11 + x22 * x22;
    const double r = std::sqrt(b + x33 * x33);
    const double sqb = std::sqrt(b);
    const double sb = r == 0.0 ? 0.0 : sqb / r;

    double ca = 0.0;
    double sa = 0.0;
    double cb = 0.0;
    if (sb > kSmall) {
        ca = x11 / sqb;
        sa = x22 / sqb;
        cb = x33 / r;
    } else {
        if (x33 < 0.0) {
            ca = -1.0;
            cb = -1.0;
        } else if (x33 > 0.0) {
            ca = 1.0;
            cb = 1.0;
        }
    }

    PairRotation rot;
    auto& p = rot.sp;
    p[0][0] = ca * sb;
    p[1][0] = ca * cb;
    p[2][0] = -sa;
    p[0][1] = sa * sb;
    p[1][1] = sa * cb;
    p[2][1] = ca;
    p[0][2] = cb;
    p[1][2] = -sb;
    p[2][2] = 0.0;

    auto& pp = rot.pp;
    for (int k = 0; k < 3; ++k) {
        pp[0][k][k] = p[k][0] * p[k][0];
        pp[1][k][k] = p[k][1] * p[k][1];
        pp[2][k][k] = p[k][2] * p[k][2];
        pp[3][k][k] = p[k][0] * p[k][1];
        pp[4][k][k] = p[k][0] * p[k][2];
        pp[5][k][k] = p[k][1] * p[k][2];
        for (int l = 0; l < k; ++l) {
            pp[0][k][l] = 2.0 * p[k][0] * p[l][0];
            pp[1][k][l] = 2.0 * p[k][1] * p[l][1];
            pp[2][k][l] = 2.0 * p[k][2] * p[l][2];
            pp[3][k][l] = p[k][0] * p[l][1] + p[k][1] * p[l][0];
            pp[4][k][l] = p[k][0] * p[l][2] + p[k][2] * p[l][0];
            pp[5][k][l] = p[k][1] * p[l][2] + p[k][2] * p[l][1];
        }
    }
    return rot;
}

int indx(int i, int j) {
    const int hi = i > j ? i : j;
    const int lo = i > j ? j : i;
    return hi * (hi - 1) / 2 + lo;
}

int indexd(int i, int j) {
    const int hi = i > j ? i : j;
    const int lo = i > j ? j : i;
    return -(lo * (lo - 1)) / 2 + hi + 9 * (lo - 1);
}

int sp_ind2(int ij, int kl) {
    struct Entry {
        int ij;
        int kl;
        int value;
    };
    static constexpr Entry kEntries[] = {
        {1, 1, 1},   {1, 2, 2},    {1, 10, 3},   {1, 18, 4},   {1, 25, 5},
        {2, 1, 6},   {2, 2, 7},    {2, 10, 8},   {2, 18, 9},   {2, 25, 10},
        {10, 1, 11}, {10, 2, 12},  {10, 10, 13}, {10, 18, 14}, {10, 25, 15},
        {3, 3, 16},  {3, 11, 17},  {11, 3, 18},  {11, 11, 19}, {18, 1, 20},
        {18, 2, 21}, {18, 10, 22}, {18, 18, 23}, {18, 25, 24}, {4, 4, 25},
        {4, 12, 26}, {12, 4, 27},  {12, 12, 28}, {19, 19, 29}, {25, 1, 30},
        {25, 2, 31}, {25, 10, 32}, {25, 18, 33}, {25, 25, 34},
    };
    for (const auto& entry : kEntries) {
        if (entry.ij == ij && entry.kl == kl) {
            return entry.value - 1;
        }
    }
    return -1;
}

std::array<double, 22> reppd_sp(const AM1Parameter& pi, const AM1Parameter& pj,
                                double r_angstrom) {
    const double ev1 = kEv / 2.0;
    const double ev2 = ev1 / 2.0;
    const double ev3 = ev2 / 2.0;
    const double ev4 = ev3 / 2.0;
    constexpr double td = 2.0;
    constexpr double pp = 0.5;

    std::array<double, 22> ri{};
    std::array<double, 72> arg{};
    std::array<double, 72> sqr{};
    const double r = r_angstrom / kA0;
    const bool si = pi.n_orbitals >= 3;
    const bool sj = pj.n_orbitals >= 3;

    double aee = pp / pi.am + pp / pj.am;
    aee = aee * aee;
    if (!si && !sj) {
        ri[0] = kEv / std::sqrt(r * r + aee);
    } else if (si && !sj) {
        const double da = pi.dd;
        double qa = pi.qq * td;
        double ade = pp / pi.ad + pp / pj.am;
        ade *= ade;
        double aqe = pp / pi.aq + pp / pj.am;
        aqe *= aqe;
        const double rsq = r * r;
        arg[0] = rsq + aee;
        arg[1] = (r + da) * (r + da) + ade;
        arg[2] = (r - da) * (r - da) + ade;
        arg[3] = (r + qa) * (r + qa) + aqe;
        arg[4] = (r - qa) * (r - qa) + aqe;
        arg[5] = rsq + aqe;
        arg[6] = arg[5] + qa * qa;
        for (int i = 0; i < 7; ++i) {
            sqr[i] = std::sqrt(arg[i]);
        }
        const double ee = kEv / sqr[0];
        ri[0] = ee;
        ri[1] = ev1 / sqr[1] - ev1 / sqr[2];
        ri[2] = ee + ev2 / sqr[3] + ev2 / sqr[4] - ev1 / sqr[5];
        ri[3] = ee + ev1 / sqr[6] - ev1 / sqr[5];
    } else if (!si && sj) {
        const double db = pj.dd;
        double qb = pj.qq * td;
        double aed = pp / pi.am + pp / pj.ad;
        aed *= aed;
        double aeq = pp / pi.am + pp / pj.aq;
        aeq *= aeq;
        const double rsq = r * r;
        arg[0] = rsq + aee;
        arg[1] = (r - db) * (r - db) + aed;
        arg[2] = (r + db) * (r + db) + aed;
        arg[3] = (r - qb) * (r - qb) + aeq;
        arg[4] = (r + qb) * (r + qb) + aeq;
        arg[5] = rsq + aeq;
        arg[6] = arg[5] + qb * qb;
        for (int i = 0; i < 7; ++i) {
            sqr[i] = std::sqrt(arg[i]);
        }
        const double ee = kEv / sqr[0];
        ri[0] = ee;
        ri[4] = ev1 / sqr[1] - ev1 / sqr[2];
        ri[10] = ee + ev2 / sqr[3] + ev2 / sqr[4] - ev1 / sqr[5];
        ri[11] = ee + ev1 / sqr[6] - ev1 / sqr[5];
    } else {
        const double da = pi.dd;
        const double db = pj.dd;
        double qa = pi.qq * td;
        double qb = pj.qq * td;
        double ade = pp / pi.ad + pp / pj.am;
        ade *= ade;
        double aqe = pp / pi.aq + pp / pj.am;
        aqe *= aqe;
        double aed = pp / pi.am + pp / pj.ad;
        aed *= aed;
        double aeq = pp / pi.am + pp / pj.aq;
        aeq *= aeq;
        double axx = pp / pi.ad + pp / pj.ad;
        axx *= axx;
        double adq = pp / pi.ad + pp / pj.aq;
        adq *= adq;
        double aqd = pp / pi.aq + pp / pj.ad;
        aqd *= aqd;
        double aqq = pp / pi.aq + pp / pj.aq;
        aqq *= aqq;
        const double rsq = r * r;
        arg[0] = rsq + aee;
        arg[1] = (r + da) * (r + da) + ade;
        arg[2] = (r - da) * (r - da) + ade;
        arg[3] = (r - qa) * (r - qa) + aqe;
        arg[4] = (r + qa) * (r + qa) + aqe;
        arg[5] = rsq + aqe;
        arg[6] = arg[5] + qa * qa;
        arg[7] = (r - db) * (r - db) + aed;
        arg[8] = (r + db) * (r + db) + aed;
        arg[9] = (r - qb) * (r - qb) + aeq;
        arg[10] = (r + qb) * (r + qb) + aeq;
        arg[11] = rsq + aeq;
        arg[12] = arg[11] + qb * qb;
        arg[13] = rsq + axx + (da - db) * (da - db);
        arg[14] = rsq + axx + (da + db) * (da + db);
        arg[15] = (r + da - db) * (r + da - db) + axx;
        arg[16] = (r - da + db) * (r - da + db) + axx;
        arg[17] = (r - da - db) * (r - da - db) + axx;
        arg[18] = (r + da + db) * (r + da + db) + axx;
        arg[19] = (r + da) * (r + da) + adq;
        arg[20] = arg[19] + qb * qb;
        arg[21] = (r - da) * (r - da) + adq;
        arg[22] = arg[21] + qb * qb;
        arg[23] = (r - db) * (r - db) + aqd;
        arg[24] = arg[23] + qa * qa;
        arg[25] = (r + db) * (r + db) + aqd;
        arg[26] = arg[25] + qa * qa;
        arg[27] = (r + da - qb) * (r + da - qb) + adq;
        arg[28] = (r - da - qb) * (r - da - qb) + adq;
        arg[29] = (r + da + qb) * (r + da + qb) + adq;
        arg[30] = (r - da + qb) * (r - da + qb) + adq;
        arg[31] = (r + qa - db) * (r + qa - db) + aqd;
        arg[32] = (r + qa + db) * (r + qa + db) + aqd;
        arg[33] = (r - qa - db) * (r - qa - db) + aqd;
        arg[34] = (r - qa + db) * (r - qa + db) + aqd;
        arg[35] = rsq + aqq;
        arg[36] = arg[35] + (qa - qb) * (qa - qb);
        arg[37] = arg[35] + (qa + qb) * (qa + qb);
        arg[38] = arg[35] + qa * qa;
        arg[39] = arg[35] + qb * qb;
        arg[40] = arg[38] + qb * qb;
        arg[41] = (r - qb) * (r - qb) + aqq;
        arg[42] = arg[41] + qa * qa;
        arg[43] = (r + qb) * (r + qb) + aqq;
        arg[44] = arg[43] + qa * qa;
        arg[45] = (r + qa) * (r + qa) + aqq;
        arg[46] = arg[45] + qb * qb;
        arg[47] = (r - qa) * (r - qa) + aqq;
        arg[48] = arg[47] + qb * qb;
        arg[49] = (r + qa - qb) * (r + qa - qb) + aqq;
        arg[50] = (r + qa + qb) * (r + qa + qb) + aqq;
        arg[51] = (r - qa - qb) * (r - qa - qb) + aqq;
        arg[52] = (r - qa + qb) * (r - qa + qb) + aqq;
        qa = pi.qq;
        qb = pj.qq;
        arg[53] = (da - qb) * (da - qb) + (r - qb) * (r - qb) + adq;
        arg[54] = (da - qb) * (da - qb) + (r + qb) * (r + qb) + adq;
        arg[55] = (da + qb) * (da + qb) + (r - qb) * (r - qb) + adq;
        arg[56] = (da + qb) * (da + qb) + (r + qb) * (r + qb) + adq;
        arg[57] = (r + qa) * (r + qa) + (qa - db) * (qa - db) + aqd;
        arg[58] = (r - qa) * (r - qa) + (qa - db) * (qa - db) + aqd;
        arg[59] = (r + qa) * (r + qa) + (qa + db) * (qa + db) + aqd;
        arg[60] = (r - qa) * (r - qa) + (qa + db) * (qa + db) + aqd;
        arg[61] = arg[35] + td * (qa - qb) * (qa - qb);
        arg[62] = arg[35] + td * (qa + qb) * (qa + qb);
        arg[63] = arg[35] + td * (qa * qa + qb * qb);
        arg[64] = (r + qa - qb) * (r + qa - qb) + (qa - qb) * (qa - qb) + aqq;
        arg[65] = (r + qa - qb) * (r + qa - qb) + (qa + qb) * (qa + qb) + aqq;
        arg[66] = (r + qa + qb) * (r + qa + qb) + (qa - qb) * (qa - qb) + aqq;
        arg[67] = (r + qa + qb) * (r + qa + qb) + (qa + qb) * (qa + qb) + aqq;
        arg[68] = (r - qa - qb) * (r - qa - qb) + (qa - qb) * (qa - qb) + aqq;
        arg[69] = (r - qa - qb) * (r - qa - qb) + (qa + qb) * (qa + qb) + aqq;
        arg[70] = (r - qa + qb) * (r - qa + qb) + (qa - qb) * (qa - qb) + aqq;
        arg[71] = (r - qa + qb) * (r - qa + qb) + (qa + qb) * (qa + qb) + aqq;
        for (int i = 0; i < 72; ++i) {
            sqr[i] = std::sqrt(arg[i]);
        }
        const double ee = kEv / sqr[0];
        const double dze = (-ev1 / sqr[1]) + ev1 / sqr[2];
        const double qzze = ev2 / sqr[3] + ev2 / sqr[4] - ev1 / sqr[5];
        const double qxxe = ev1 / sqr[6] - ev1 / sqr[5];
        const double edz = (-ev1 / sqr[7]) + ev1 / sqr[8];
        const double eqzz = ev2 / sqr[9] + ev2 / sqr[10] - ev1 / sqr[11];
        const double eqxx = ev1 / sqr[12] - ev1 / sqr[11];
        const double dxdx = ev1 / sqr[13] - ev1 / sqr[14];
        const double dzdz =
            ev2 / sqr[15] + ev2 / sqr[16] - ev2 / sqr[17] - ev2 / sqr[18];
        const double dzqxx =
            ev2 / sqr[19] - ev2 / sqr[20] - ev2 / sqr[21] + ev2 / sqr[22];
        const double qxxdz =
            ev2 / sqr[23] - ev2 / sqr[24] - ev2 / sqr[25] + ev2 / sqr[26];
        const double dzqzz = (-ev3 / sqr[27]) + ev3 / sqr[28] - ev3 / sqr[29]
                             + ev3 / sqr[30] - ev2 / sqr[21] + ev2 / sqr[19];
        const double qzzdz = (-ev3 / sqr[31]) + ev3 / sqr[32] - ev3 / sqr[33]
                             + ev3 / sqr[34] + ev2 / sqr[23] - ev2 / sqr[25];
        const double qxxqxx = ev3 / sqr[36] + ev3 / sqr[37] - ev2 / sqr[38]
                              - ev2 / sqr[39] + ev2 / sqr[35];
        const double qxxqyy =
            ev2 / sqr[40] - ev2 / sqr[38] - ev2 / sqr[39] + ev2 / sqr[35];
        const double qxxqzz = ev3 / sqr[42] + ev3 / sqr[44] - ev3 / sqr[41]
                              - ev3 / sqr[43] - ev2 / sqr[38] + ev2 / sqr[35];
        const double qzzqxx = ev3 / sqr[46] + ev3 / sqr[48] - ev3 / sqr[45]
                              - ev3 / sqr[47] - ev2 / sqr[39] + ev2 / sqr[35];
        const double qzzqzz = ev4 / sqr[49] + ev4 / sqr[50] + ev4 / sqr[51]
                              + ev4 / sqr[52] - ev3 / sqr[47] - ev3 / sqr[45]
                              - ev3 / sqr[41] - ev3 / sqr[43] + ev2 / sqr[35];
        const double dxqxz =
            (-ev2 / sqr[53]) + ev2 / sqr[54] + ev2 / sqr[55] - ev2 / sqr[56];
        const double qxzdx =
            (-ev2 / sqr[57]) + ev2 / sqr[58] + ev2 / sqr[59] - ev2 / sqr[60];
        const double qxzqxz = ev3 / sqr[64] - ev3 / sqr[66] - ev3 / sqr[68]
                              + ev3 / sqr[70] - ev3 / sqr[65] + ev3 / sqr[67]
                              + ev3 / sqr[69] - ev3 / sqr[71];
        ri[0] = ee;
        ri[1] = -dze;
        ri[2] = ee + qzze;
        ri[3] = ee + qxxe;
        ri[4] = -edz;
        ri[5] = dzdz;
        ri[6] = dxdx;
        ri[7] = (-edz) - qzzdz;
        ri[8] = (-edz) - qxxdz;
        ri[9] = -qxzdx;
        ri[10] = ee + eqzz;
        ri[11] = ee + eqxx;
        ri[12] = (-dze) - dzqzz;
        ri[13] = (-dze) - dzqxx;
        ri[14] = -dxqxz;
        ri[15] = ee + eqzz + qzze + qzzqzz;
        ri[16] = ee + eqzz + qxxe + qxxqzz;
        ri[17] = ee + eqxx + qzze + qzzqxx;
        ri[18] = ee + eqxx + qxxe + qxxqxx;
        ri[19] = qxzqxz;
        ri[20] = ee + eqxx + qxxe + qxxqyy;
        ri[21] = pp * (qxxqxx - qxxqyy);
    }

    static constexpr double kNri[22] = {
        1.0, -1.0, 1.0,  1.0,  -1.0, 1.0, 1.0, -1.0, -1.0, -1.0, 1.0,
        1.0, -1.0, -1.0, -1.0, 1.0,  1.0, 1.0, 1.0,  1.0,  1.0,  1.0};
    for (int i = 0; i < 22; ++i) {
        ri[i] *= kNri[i];
    }
    return ri;
}

std::array<double, 34> rep_from_ri(const std::array<double, 22>& ri) {
    static constexpr int kIpos[34] = {
        1,  5,  11, 12, 12, 2,  6,  13, 14, 14, 3,  8,  16, 18, 18, 7,  15,
        10, 20, 4,  9,  17, 19, 21, 7,  15, 10, 20, 22, 4,  9,  17, 21, 19};
    std::array<double, 34> rep{};
    for (int i = 0; i < 34; ++i) {
        rep[i] = ri[kIpos[i] - 1];
    }
    return rep;
}

double& w_at(std::vector<double>& values, int limkl, int pair_i, int pair_j) {
    return values[static_cast<std::size_t>(pair_i)
                      * static_cast<std::size_t>(limkl)
                  + static_cast<std::size_t>(pair_j)];
}

} // namespace

double NDDOTwoCenterIntegralBlock::at(int pair_i, int pair_j) const {
    if (pair_i < 0 || pair_i >= n_pairs_i || pair_j < 0
        || pair_j >= n_pairs_j) {
        throw std::runtime_error("NDDO two-center integral index out of range");
    }
    return values[static_cast<std::size_t>(pair_i)
                      * static_cast<std::size_t>(n_pairs_j)
                  + static_cast<std::size_t>(pair_j)];
}

NDDOTwoCenterIntegralBlock build_two_center_integral_block(
    const NDDOSystem& system, const StorageReal* sp_x, const StorageReal* sp_y,
    const StorageReal* sp_z, int atom_i, int atom_j) {
    if (atom_i < 0 || atom_j < 0
        || atom_i >= static_cast<int>(system.atoms.size())
        || atom_j >= static_cast<int>(system.atoms.size())) {
        throw std::runtime_error("NDDO two-center atom index out of range");
    }
    const auto& ai = system.atoms[static_cast<std::size_t>(atom_i)];
    const auto& aj = system.atoms[static_cast<std::size_t>(atom_j)];
    if (ai.n_orbitals > 4 || aj.n_orbitals > 4) {
        throw std::runtime_error(
            "AM1 d-orbital two-center electron-repulsion integrals are not "
            "ported yet");
    }
    const auto* pi = ai.params;
    const auto* pj = aj.params;
    if (pi == nullptr || pj == nullptr) {
        throw std::runtime_error(
            "AM1 parameters unavailable for two-center integrals");
    }

    const auto xi = atom_xyz(sp_x, sp_y, sp_z, atom_i);
    const auto xj = atom_xyz(sp_x, sp_y, sp_z, atom_j);
    const double rij = distance(xi, xj);
    if (rij < 1.0e-12) {
        throw std::runtime_error(
            "NDDO two-center integrals require distinct atom coordinates");
    }

    const int ii = ai.n_orbitals;
    const int kk = aj.n_orbitals;
    const int limij = indx(ii, ii);
    const int limkl = indx(kk, kk);
    const auto rot = pair_rotation(xi, xj);
    const auto rep = rep_from_ri(reppd_sp(*pi, *pj, rij));

    std::array<std::array<double, 46>, 46> v{};
    std::array<std::array<bool, 46>, 46> logv{};
    static constexpr int kMet[46] = {
        0, 1, 2, 3, 2, 3, 3, 2, 3, 3, 3, 4, 5, 5, 5, 6, 4, 5, 5, 5, 6, 6, 4,
        5, 5, 5, 6, 6, 6, 4, 5, 5, 5, 6, 6, 6, 6, 4, 5, 5, 5, 6, 6, 6, 6, 6};

    for (int i1 = 1; i1 <= ii; ++i1) {
        for (int j1 = 1; j1 <= i1; ++j1) {
            const int ij = indexd(i1, j1);
            for (int k1 = 1; k1 <= kk; ++k1) {
                for (int l1 = 1; l1 <= k1; ++l1) {
                    const int kl = indexd(k1, l1);
                    const int nd = sp_ind2(ij, kl);
                    if (nd < 0) {
                        continue;
                    }
                    const double wrepp = rep[static_cast<std::size_t>(nd)];
                    const int ll = indx(k1, l1);
                    switch (kMet[ll]) {
                    case 1:
                        v[ij][1] = wrepp;
                        break;
                    case 2: {
                        const int k = k1 - 2;
                        v[ij][2] += rot.sp[k][0] * wrepp;
                        v[ij][4] += rot.sp[k][1] * wrepp;
                        v[ij][7] += rot.sp[k][2] * wrepp;
                        break;
                    }
                    case 3: {
                        const int k = k1 - 2;
                        const int l = l1 - 2;
                        v[ij][3] += rot.pp[0][k][l] * wrepp;
                        v[ij][6] += rot.pp[1][k][l] * wrepp;
                        v[ij][10] += rot.pp[2][k][l] * wrepp;
                        v[ij][5] += rot.pp[3][k][l] * wrepp;
                        v[ij][8] += rot.pp[4][k][l] * wrepp;
                        v[ij][9] += rot.pp[5][k][l] * wrepp;
                        break;
                    }
                    default:
                        throw std::runtime_error(
                            "AM1 d-orbital two-center transform is not ported "
                            "yet");
                    }
                }
            }
            for (int kl = 1; kl <= limkl; ++kl) {
                logv[ij][kl] = v[ij][kl] != 0.0;
            }
        }
    }

    NDDOTwoCenterIntegralBlock out;
    out.atom_i = atom_i;
    out.atom_j = atom_j;
    out.n_pairs_i = limij;
    out.n_pairs_j = limkl;
    out.values.assign(
        static_cast<std::size_t>(limij) * static_cast<std::size_t>(limkl), 0.0);

    for (int i1 = 1; i1 <= ii; ++i1) {
        for (int j1 = 1; j1 <= i1; ++j1) {
            const int ij = indexd(i1, j1);
            const int jj = indx(i1, j1);
            const int mm = kMet[jj];
            for (int k = 1; k <= kk; ++k) {
                for (int l = 1; l <= k; ++l) {
                    const int kl = indx(k, l);
                    if (!logv[ij][kl]) {
                        continue;
                    }
                    const double wrepp = v[ij][kl];
                    switch (mm) {
                    case 1:
                        w_at(out.values, limkl, indx(1, 1) - 1, kl - 1) = wrepp;
                        break;
                    case 2: {
                        const int p_local = i1 - 2;
                        for (int axis = 0; axis < 3; ++axis) {
                            w_at(out.values, limkl, indx(axis + 2, 1) - 1,
                                 kl - 1) += rot.sp[p_local][axis] * wrepp;
                        }
                        break;
                    }
                    case 3: {
                        const int p_i = i1 - 2;
                        const int p_j = j1 - 2;
                        for (int axis = 0; axis < 3; ++axis) {
                            w_at(out.values, limkl,
                                 indx(axis + 2, axis + 2) - 1, kl - 1) +=
                                rot.pp[axis][p_i][p_j] * wrepp;
                        }
                        w_at(out.values, limkl, indx(3, 2) - 1, kl - 1) +=
                            rot.pp[3][p_i][p_j] * wrepp;
                        w_at(out.values, limkl, indx(4, 2) - 1, kl - 1) +=
                            rot.pp[4][p_i][p_j] * wrepp;
                        w_at(out.values, limkl, indx(4, 3) - 1, kl - 1) +=
                            rot.pp[5][p_i][p_j] * wrepp;
                        break;
                    }
                    default:
                        throw std::runtime_error(
                            "AM1 d-orbital two-center transform is not ported "
                            "yet");
                    }
                }
            }
        }
    }

    return out;
}

} // namespace semiempirical
} // namespace cpu
} // namespace ATV
