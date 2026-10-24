/// @file src/cpu/semiempirical/nddo_two_electron.cpp
/// @brief AM1 reference reppd/rotatd/w2mat s/p two-electron integral port.

#include "atomiverse/cpu/semiempirical/nddo_two_electron.h"

#include "atomiverse/cpu/semiempirical/internal/nddo_derivative.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <stdexcept>
#include <type_traits>

#include "atomiverse/semiempirical/am1_parameters.h"

namespace ATV {
namespace cpu {
namespace semiempirical {

namespace {
using internal::sqrt;
using std::sqrt;

constexpr double kA0 = 0.529177210903;
constexpr double kEv = 27.211386245988;

template <class Scalar>
struct PairRotation {
    std::array<std::array<Scalar, 3>, 3> sp{};
    std::array<std::array<std::array<Scalar, 3>, 3>, 6> pp{};
};
std::array<double, 3> atom_xyz(const StorageReal* x, const StorageReal* y,
                               const StorageReal* z, int i) {
    return {static_cast<double>(x[i]), static_cast<double>(y[i]),
            static_cast<double>(z[i])};
}
template <class Scalar>
Scalar distance(const std::array<Scalar, 3>& a,
                const std::array<Scalar, 3>& b) {
    const Scalar dx = a[0] - b[0], dy = a[1] - b[1], dz = a[2] - b[2];
    return sqrt(dx * dx + dy * dy + dz * dz);
}
template <class Scalar>
PairRotation<Scalar> pair_rotation(const std::array<Scalar, 3>& xi,
                                   const std::array<Scalar, 3>& xj) {
    PairRotation<Scalar> rot;
    rot.sp = internal::nddo_pair_axes(
        std::array<Scalar, 3>{xj[0] - xi[0], xj[1] - xi[1], xj[2] - xi[2]});
    auto& p = rot.sp;
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

template <class Scalar>
std::array<Scalar, 22> reppd_sp(const ATV::semiempirical::AM1Parameter& pi,
                                const ATV::semiempirical::AM1Parameter& pj,
                                Scalar r_angstrom) {
    const Scalar ev1 = kEv / 2.0;
    const Scalar ev2 = ev1 / 2.0;
    const Scalar ev3 = ev2 / 2.0;
    const Scalar ev4 = ev3 / 2.0;
    constexpr double td = 2.0;
    constexpr double pp = 0.5;

    std::array<Scalar, 22> ri{};
    std::array<Scalar, 72> arg{};
    std::array<Scalar, 72> sqr{};
    const Scalar r = r_angstrom / kA0;
    const bool si = pi.n_orbitals >= 3;
    const bool sj = pj.n_orbitals >= 3;

    Scalar aee = pp / pi.am + pp / pj.am;
    aee = aee * aee;
    if (!si && !sj) {
        ri[0] = kEv / sqrt(r * r + aee);
    } else if (si && !sj) {
        const Scalar da = pi.dd;
        Scalar qa = pi.qq * td;
        Scalar ade = pp / pi.ad + pp / pj.am;
        ade *= ade;
        Scalar aqe = pp / pi.aq + pp / pj.am;
        aqe *= aqe;
        const Scalar rsq = r * r;
        arg[0] = rsq + aee;
        arg[1] = (r + da) * (r + da) + ade;
        arg[2] = (r - da) * (r - da) + ade;
        arg[3] = (r + qa) * (r + qa) + aqe;
        arg[4] = (r - qa) * (r - qa) + aqe;
        arg[5] = rsq + aqe;
        arg[6] = arg[5] + qa * qa;
        for (int i = 0; i < 7; ++i) {
            sqr[i] = sqrt(arg[i]);
        }
        const Scalar ee = kEv / sqr[0];
        ri[0] = ee;
        ri[1] = ev1 / sqr[1] - ev1 / sqr[2];
        ri[2] = ee + ev2 / sqr[3] + ev2 / sqr[4] - ev1 / sqr[5];
        ri[3] = ee + ev1 / sqr[6] - ev1 / sqr[5];
    } else if (!si && sj) {
        const Scalar db = pj.dd;
        Scalar qb = pj.qq * td;
        Scalar aed = pp / pi.am + pp / pj.ad;
        aed *= aed;
        Scalar aeq = pp / pi.am + pp / pj.aq;
        aeq *= aeq;
        const Scalar rsq = r * r;
        arg[0] = rsq + aee;
        arg[1] = (r - db) * (r - db) + aed;
        arg[2] = (r + db) * (r + db) + aed;
        arg[3] = (r - qb) * (r - qb) + aeq;
        arg[4] = (r + qb) * (r + qb) + aeq;
        arg[5] = rsq + aeq;
        arg[6] = arg[5] + qb * qb;
        for (int i = 0; i < 7; ++i) {
            sqr[i] = sqrt(arg[i]);
        }
        const Scalar ee = kEv / sqr[0];
        ri[0] = ee;
        ri[4] = ev1 / sqr[1] - ev1 / sqr[2];
        ri[10] = ee + ev2 / sqr[3] + ev2 / sqr[4] - ev1 / sqr[5];
        ri[11] = ee + ev1 / sqr[6] - ev1 / sqr[5];
    } else {
        const Scalar da = pi.dd;
        const Scalar db = pj.dd;
        Scalar qa = pi.qq * td;
        Scalar qb = pj.qq * td;
        Scalar ade = pp / pi.ad + pp / pj.am;
        ade *= ade;
        Scalar aqe = pp / pi.aq + pp / pj.am;
        aqe *= aqe;
        Scalar aed = pp / pi.am + pp / pj.ad;
        aed *= aed;
        Scalar aeq = pp / pi.am + pp / pj.aq;
        aeq *= aeq;
        Scalar axx = pp / pi.ad + pp / pj.ad;
        axx *= axx;
        Scalar adq = pp / pi.ad + pp / pj.aq;
        adq *= adq;
        Scalar aqd = pp / pi.aq + pp / pj.ad;
        aqd *= aqd;
        Scalar aqq = pp / pi.aq + pp / pj.aq;
        aqq *= aqq;
        const Scalar rsq = r * r;
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
            sqr[i] = sqrt(arg[i]);
        }
        const Scalar ee = kEv / sqr[0];
        const Scalar dze = (-ev1 / sqr[1]) + ev1 / sqr[2];
        const Scalar qzze = ev2 / sqr[3] + ev2 / sqr[4] - ev1 / sqr[5];
        const Scalar qxxe = ev1 / sqr[6] - ev1 / sqr[5];
        const Scalar edz = (-ev1 / sqr[7]) + ev1 / sqr[8];
        const Scalar eqzz = ev2 / sqr[9] + ev2 / sqr[10] - ev1 / sqr[11];
        const Scalar eqxx = ev1 / sqr[12] - ev1 / sqr[11];
        const Scalar dxdx = ev1 / sqr[13] - ev1 / sqr[14];
        const Scalar dzdz =
            ev2 / sqr[15] + ev2 / sqr[16] - ev2 / sqr[17] - ev2 / sqr[18];
        const Scalar dzqxx =
            ev2 / sqr[19] - ev2 / sqr[20] - ev2 / sqr[21] + ev2 / sqr[22];
        const Scalar qxxdz =
            ev2 / sqr[23] - ev2 / sqr[24] - ev2 / sqr[25] + ev2 / sqr[26];
        const Scalar dzqzz = (-ev3 / sqr[27]) + ev3 / sqr[28] - ev3 / sqr[29]
                             + ev3 / sqr[30] - ev2 / sqr[21] + ev2 / sqr[19];
        const Scalar qzzdz = (-ev3 / sqr[31]) + ev3 / sqr[32] - ev3 / sqr[33]
                             + ev3 / sqr[34] + ev2 / sqr[23] - ev2 / sqr[25];
        const Scalar qxxqxx = ev3 / sqr[36] + ev3 / sqr[37] - ev2 / sqr[38]
                              - ev2 / sqr[39] + ev2 / sqr[35];
        const Scalar qxxqyy =
            ev2 / sqr[40] - ev2 / sqr[38] - ev2 / sqr[39] + ev2 / sqr[35];
        const Scalar qxxqzz = ev3 / sqr[42] + ev3 / sqr[44] - ev3 / sqr[41]
                              - ev3 / sqr[43] - ev2 / sqr[38] + ev2 / sqr[35];
        const Scalar qzzqxx = ev3 / sqr[46] + ev3 / sqr[48] - ev3 / sqr[45]
                              - ev3 / sqr[47] - ev2 / sqr[39] + ev2 / sqr[35];
        const Scalar qzzqzz = ev4 / sqr[49] + ev4 / sqr[50] + ev4 / sqr[51]
                              + ev4 / sqr[52] - ev3 / sqr[47] - ev3 / sqr[45]
                              - ev3 / sqr[41] - ev3 / sqr[43] + ev2 / sqr[35];
        const Scalar dxqxz =
            (-ev2 / sqr[53]) + ev2 / sqr[54] + ev2 / sqr[55] - ev2 / sqr[56];
        const Scalar qxzdx =
            (-ev2 / sqr[57]) + ev2 / sqr[58] + ev2 / sqr[59] - ev2 / sqr[60];
        const Scalar qxzqxz = ev3 / sqr[64] - ev3 / sqr[66] - ev3 / sqr[68]
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

template <class Scalar>
std::array<Scalar, 34> rep_from_ri(const std::array<Scalar, 22>& ri) {
    static constexpr int kIpos[34] = {
        1,  5,  11, 12, 12, 2,  6,  13, 14, 14, 3,  8,  16, 18, 18, 7,  15,
        10, 20, 4,  9,  17, 19, 21, 7,  15, 10, 20, 22, 4,  9,  17, 21, 19};
    std::array<Scalar, 34> rep{};
    for (int i = 0; i < 34; ++i) {
        rep[i] = ri[kIpos[i] - 1];
    }
    return rep;
}

template <class Scalar>
Scalar& w_at(Scalar* values, int limkl, int pair_i, int pair_j) {
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

namespace {
template <class Scalar>
struct IntegralBlock {
    int atom_i = -1, atom_j = -1, n_pairs_i = 0, n_pairs_j = 0;
    std::array<Scalar, 100> values{};
};
template <class Scalar, class Block>
void integral_block(const NDDOSystem& system, const std::array<Scalar, 3>& xi,
                    const std::array<Scalar, 3>& xj, int atom_i, int atom_j,
                    Block& out) {
    if (atom_i < 0 || atom_j < 0
        || atom_i >= static_cast<int>(system.atoms.size())
        || atom_j >= static_cast<int>(system.atoms.size())) {
        throw std::runtime_error("NDDO two-center atom index out of range");
    }
    const auto& ai = system.atoms[static_cast<std::size_t>(atom_i)];
    const auto& aj = system.atoms[static_cast<std::size_t>(atom_j)];
    if ((ai.n_orbitals != 1 && ai.n_orbitals != 4)
        || (aj.n_orbitals != 1 && aj.n_orbitals != 4)) {
        throw std::runtime_error("NDDO two-center integrals require complete s "
                                 "or sp shells (1 or 4 orbitals)");
    }
    const auto* pi = ai.params;
    const auto* pj = aj.params;
    if (pi == nullptr || pj == nullptr) {
        throw std::runtime_error(
            "AM1 parameters unavailable for two-center integrals");
    }

    const Scalar rij = distance(xi, xj);
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

    // Only s/p blocks are supported here: indexd <= 25, packed pair <= 10.
    std::array<std::array<Scalar, 11>, 26> v{};

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
                    const Scalar wrepp = rep[static_cast<std::size_t>(nd)];
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
        }
    }

    out.atom_i = atom_i;
    out.atom_j = atom_j;
    out.n_pairs_i = limij;
    out.n_pairs_j = limkl;
    std::fill_n(out.values.data(), static_cast<std::size_t>(limij) * limkl,
                Scalar{});

    for (int i1 = 1; i1 <= ii; ++i1) {
        for (int j1 = 1; j1 <= i1; ++j1) {
            const int ij = indexd(i1, j1);
            const int jj = indx(i1, j1);
            const int mm = kMet[jj];
            for (int k = 1; k <= kk; ++k) {
                for (int l = 1; l <= k; ++l) {
                    const int kl = indx(k, l);

                    // A zero value may still carry a nonzero derivative.
                    if constexpr (std::is_same_v<Scalar, double>) {
                        if (v[ij][kl] == 0.0)
                            continue;
                    }
                    const Scalar wrepp = v[ij][kl];
                    switch (mm) {
                    case 1:
                        w_at(out.values.data(), limkl, indx(1, 1) - 1, kl - 1) =
                            wrepp;
                        break;
                    case 2: {
                        const int p_local = i1 - 2;
                        for (int axis = 0; axis < 3; ++axis) {
                            w_at(out.values.data(), limkl,
                                 indx(axis + 2, 1) - 1, kl - 1) +=
                                rot.sp[p_local][axis] * wrepp;
                        }
                        break;
                    }
                    case 3: {
                        const int p_i = i1 - 2;
                        const int p_j = j1 - 2;
                        for (int axis = 0; axis < 3; ++axis) {
                            w_at(out.values.data(), limkl,
                                 indx(axis + 2, axis + 2) - 1, kl - 1) +=
                                rot.pp[axis][p_i][p_j] * wrepp;
                        }
                        w_at(out.values.data(), limkl, indx(3, 2) - 1,
                             kl - 1) += rot.pp[3][p_i][p_j] * wrepp;
                        w_at(out.values.data(), limkl, indx(4, 2) - 1,
                             kl - 1) += rot.pp[4][p_i][p_j] * wrepp;
                        w_at(out.values.data(), limkl, indx(4, 3) - 1,
                             kl - 1) += rot.pp[5][p_i][p_j] * wrepp;
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
}
} // namespace
void build_two_center_integral_block(const NDDOSystem& system,
                                     const StorageReal* x, const StorageReal* y,
                                     const StorageReal* z, int i, int j,
                                     NDDOTwoCenterIntegralBlock& out) {
    // Check indices before touching coordinate spans.
    if (i < 0 || j < 0 || i >= static_cast<int>(system.atoms.size())
        || j >= static_cast<int>(system.atoms.size()))
        throw std::runtime_error("NDDO two-center atom index out of range");
    integral_block(system, atom_xyz(x, y, z, i), atom_xyz(x, y, z, j), i, j,
                   out);
}
std::array<std::array<double, 100>, 3> internal::two_center_integral_derivative(
    const NDDOSystem& system, const StorageReal* x, const StorageReal* y,
    const StorageReal* z, int i, int j) {
    if (i < 0 || j < 0 || i >= static_cast<int>(system.atoms.size())
        || j >= static_cast<int>(system.atoms.size()))
        throw std::runtime_error("NDDO two-center atom index out of range");
    const auto ai = atom_xyz(x, y, z, i), aj = atom_xyz(x, y, z, j);
    std::array<NDDODerivative, 3> xi, xj;
    for (int d = 0; d < 3; ++d) {
        xi[d] = ai[d];
        xj[d] = aj[d];
        xi[d].gradient[d] = 1.0;
    }
    IntegralBlock<NDDODerivative> block;
    integral_block(system, xi, xj, i, j, block);
    std::array<std::array<double, 100>, 3> out{};
    const int count = block.n_pairs_i * block.n_pairs_j;
    for (int d = 0; d < 3; ++d)
        for (int k = 0; k < count; ++k)
            out[d][k] = block.values[k].gradient[d];
    return out;
}

} // namespace semiempirical
} // namespace cpu
} // namespace ATV
