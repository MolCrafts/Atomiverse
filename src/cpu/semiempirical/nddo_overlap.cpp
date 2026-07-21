/// @file src/cpu/semiempirical/nddo_overlap.cpp
/// @brief AM1 reference diat/h1elec overlap port.

#include "atomiverse/cpu/semiempirical/nddo_overlap.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

#include "atomiverse/cpu/semiempirical/am1_parameters.h"

namespace ATV {
namespace cpu {
namespace semiempirical {
namespace {

constexpr double kA0 = 0.529177210903;
constexpr double kOverlapCutoff2 = 225.0;
constexpr double kH1Cutoff2 = 1.0e4;
constexpr double kSqrt3 = 1.7320508075688772935;
constexpr double kInvSqrt3 = 0.57735026918962576451;

constexpr double kFact[] = {
    1.0,
    1.0,
    2.0,
    6.0,
    24.0,
    120.0,
    720.0,
    5040.0,
    40320.0,
    362880.0,
    3628800.0,
    39916800.0,
    479001600.0,
    6227020800.0,
    8.71782912e10,
    1.307674368e12,
    2.092278989e13,
    3.556874281e14,
};

int parity_sign(int n) noexcept { return (n % 2 == 0) ? 1 : -1; }

double parity_two(int n) noexcept { return (n % 2 == 0) ? 0.0 : 2.0; }

const AM1Parameter& require_am1_parameter(int z) {
    const auto* p = am1_parameter(z);
    if (p == nullptr) {
        throw std::runtime_error("AM1 parameters unavailable for overlap atom");
    }
    return *p;
}

int principal_for(const AM1Parameter& p, int shell) noexcept {
    if (shell == 1)
        return p.principal_s;
    if (shell == 2)
        return p.principal_p;
    return p.principal_d;
}

double zeta_for(const AM1Parameter& p, int shell) noexcept {
    if (shell == 1)
        return p.zs;
    if (shell == 2)
        return p.zp;
    return std::max(p.zd, 0.3);
}

bool use_diat2(int z, const AM1Parameter& p) noexcept {
    return z <= 17 && z != 2 && z != 10 && p.n_orbitals < 5;
}

struct OverlapSetState {
    int isp = 1;
    int ips = 2;
    double sa = 0.0;
    double sb = 0.0;
    std::array<double, 8> a{};
    std::array<double, 8> b{};

    void aintgs(double x, int k) {
        const double c = std::exp(-x);
        a[1] = c / x;
        for (int i = 1; i <= k; ++i) {
            a[static_cast<std::size_t>(i + 1)] =
                (a[static_cast<std::size_t>(i)] * static_cast<double>(i) + c)
                / x;
        }
    }

    void bintgs(double x, int k) {
        const int io = 0;
        const double absx = std::abs(x);
        int last = 0;
        bool use_series = false;
        if (absx <= 3.0) {
            if (absx > 2.0) {
                if (k > 10) {
                    last = 15;
                    use_series = true;
                }
            } else if (absx > 1.0) {
                if (k > 7) {
                    last = 12;
                    use_series = true;
                }
            } else if (absx > 0.5) {
                if (k > 5) {
                    last = 7;
                    use_series = true;
                }
            } else if (absx > 1.0e-6) {
                last = 6;
                use_series = true;
            } else {
                for (int i = io; i <= k; ++i) {
                    b[static_cast<std::size_t>(i + 1)] =
                        parity_two(i + 1) / static_cast<double>(i + 1);
                }
                return;
            }
        }

        if (use_series) {
            for (int i = io; i <= k; ++i) {
                double y = 0.0;
                for (int m = io; m <= last; ++m) {
                    const double xf = (m == 0) ? 1.0 : kFact[m];
                    y += std::pow(-x, m) * parity_two(m + i + 1)
                         / (xf * static_cast<double>(m + i + 1));
                }
                b[static_cast<std::size_t>(i + 1)] = y;
            }
            return;
        }

        const double expx = std::exp(x);
        const double expmx = 1.0 / expx;
        b[1] = (expx - expmx) / x;
        for (int i = 1; i <= k; ++i) {
            b[static_cast<std::size_t>(i + 1)] =
                (static_cast<double>(i) * b[static_cast<std::size_t>(i)]
                 + static_cast<double>(parity_sign(i)) * expx - expmx)
                / x;
        }
    }

    void set(double s1, double s2, int z_a, int z_b, double rab, int ii) {
        if (z_a <= z_b) {
            isp = 1;
            ips = 2;
            sa = s1;
            sb = s2;
        } else {
            isp = 2;
            ips = 1;
            sa = s2;
            sb = s1;
        }
        int j = ii + 2;
        if (ii > 3)
            --j;
        const double alpha = 0.5 * rab * (sa + sb);
        const double beta = 0.5 * rab * (sb - sa);
        const int jcall = j - 1;
        aintgs(alpha, jcall);
        bintgs(beta, jcall);
    }
};

std::array<double, 13> bfn(double x) {
    constexpr int k = 12;
    constexpr int io = 0;
    std::array<double, 13> bf{};
    const double absx = std::abs(x);
    int last = 0;
    bool use_series = false;
    if (absx <= 3.0) {
        if (absx > 2.0) {
            last = 15;
            use_series = true;
        } else if (absx > 1.0) {
            last = 12;
            use_series = true;
        } else if (absx > 0.5) {
            last = 7;
            use_series = true;
        } else if (absx > 1.0e-6) {
            last = 6;
            use_series = true;
        } else {
            for (int i = io; i <= k; ++i) {
                bf[static_cast<std::size_t>(i)] =
                    parity_two(i + 1) / static_cast<double>(i + 1);
            }
            return bf;
        }
    }

    if (use_series) {
        for (int i = io; i <= k; ++i) {
            double y = 0.0;
            for (int m = io; m <= last; ++m) {
                const double xf = (m == 0) ? 1.0 : kFact[m];
                y += std::pow(-x, m) * parity_two(m + i + 1)
                     / (xf * static_cast<double>(m + i + 1));
            }
            bf[static_cast<std::size_t>(i)] = y;
        }
        return bf;
    }

    const double expx = std::exp(x);
    const double expmx = 1.0 / expx;
    bf[0] = (expx - expmx) / x;
    for (int i = 1; i <= k; ++i) {
        bf[static_cast<std::size_t>(i)] =
            (static_cast<double>(i) * bf[static_cast<std::size_t>(i - 1)]
             + static_cast<double>(parity_sign(i)) * expx - expmx)
            / x;
    }
    return bf;
}

double ss_overlap(int na, int nb, int la1, int lb1, int m1, double ua,
                  double ub, double r_angstrom) {
    static const double aff[3][3][3] = {
        {{1.0, 0.0, 0.0}, {0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}},
        {{1.0, 0.0, 0.0}, {0.70710678118654752440, 0.0, 0.0}, {0.0, 0.0, 0.0}},
        {{1.5, 0.0, -0.5},
         {1.2247448713915890491, 0.0, 0.0},
         {0.61237243569579452455, 0.0, 0.0}},
    };

    double bi[13][13] = {};
    for (int i = 0; i <= 12; ++i) {
        bi[i][0] = 1.0;
        bi[i][i] = 1.0;
    }
    for (int i = 0; i <= 11; ++i) {
        const int i1 = i - 1;
        for (int j = 1; j <= i1 + 1; ++j) {
            bi[i + 1][j] = bi[i][j] + bi[i][j - 1];
        }
    }

    const int m = m1 - 1;
    const int lb = lb1 - 1;
    const int la = la1 - 1;
    const double r = r_angstrom / kA0;
    const double p = (ua + ub) * r * 0.5;
    const double beta = (ua - ub) * r * 0.5;
    const double quo = 1.0 / p;

    std::array<double, 20> af{};
    af[0] = quo * std::exp(-p);
    for (int n = 1; n <= 19; ++n) {
        af[static_cast<std::size_t>(n)] =
            static_cast<double>(n) * quo * af[static_cast<std::size_t>(n - 1)]
            + af[0];
    }
    const auto bf = bfn(beta);

    double sum = 0.0;
    const int lam1 = la - m;
    const int lbm1 = lb - m;
    for (int i = 0; i <= lam1; i += 2) {
        const int ia = na + i - la;
        const int ic = la - i - m;
        for (int j = 0; j <= lbm1; j += 2) {
            const int ib = nb + j - lb;
            const int id = lb - j - m;
            double sum1 = 0.0;
            const int iab = ia + ib;
            for (int k1 = 0; k1 <= ia; ++k1) {
                for (int k2 = 0; k2 <= ib; ++k2) {
                    for (int k3 = 0; k3 <= ic; ++k3) {
                        for (int k4 = 0; k4 <= id; ++k4) {
                            for (int k5 = 0; k5 <= m; ++k5) {
                                const int iaf =
                                    iab - k1 - k2 + k3 + k4 + 2 * k5;
                                for (int k6 = 0; k6 <= m; ++k6) {
                                    const int ibf = k1 + k2 + k3 + k4 + 2 * k6;
                                    sum1 += bi[id][k4] * bi[ic][k3] * bi[ib][k2]
                                            * bi[ia][k1] * bi[m][k5] * bi[m][k6]
                                            * static_cast<double>(parity_sign(
                                                m + k2 + k4 + k5 + k6))
                                            * af[static_cast<std::size_t>(iaf)]
                                            * bf[static_cast<std::size_t>(ibf)];
                                }
                            }
                        }
                    }
                }
            }
            sum += sum1 * aff[la][m][i] * aff[lb][m][j];
        }
    }

    return sum * std::pow(r, na + nb + 1) * std::pow(ua, na) * std::pow(ub, nb)
           * 0.5
           * std::sqrt(ua * ub / (kFact[na + na] * kFact[nb + nb])
                       * static_cast<double>((la + la + 1) * (lb + lb + 1)));
}

void coe(double x2, double y2, double z2, int norbi, int norbj,
         std::array<double, 76>& c, double& r) {
    constexpr double rt34 = 0.86602540378444;
    constexpr double rt13 = 0.57735026918963;
    double xy = x2 * x2 + y2 * y2;
    r = std::sqrt(xy + z2 * z2);
    xy = std::sqrt(xy);

    double ca = 0.0;
    double cb = 0.0;
    double sa = 0.0;
    double sb = 0.0;
    if (xy >= 1.0e-10) {
        ca = x2 / xy;
        cb = z2 / r;
        sa = y2 / xy;
        sb = xy / r;
    } else if (z2 <= 0.0) {
        if (z2 != 0.0) {
            ca = -1.0;
            cb = -1.0;
        }
    } else {
        ca = 1.0;
        cb = 1.0;
    }

    c.fill(0.0);
    const int nij = std::max(norbi, norbj);
    c[37] = 1.0;
    if (nij < 2) {
        return;
    }

    c[56] = ca * cb;
    c[41] = ca * sb;
    c[26] = -sa;
    c[53] = -sb;
    c[38] = cb;
    c[23] = 0.0;
    c[50] = sa * cb;
    c[35] = sa * sb;
    c[20] = ca;

    if (nij < 5) {
        return;
    }
    const double c2a = 2.0 * ca * ca - 1.0;
    const double c2b = 2.0 * cb * cb - 1.0;
    const double s2a = 2.0 * sa * ca;
    const double s2b = 2.0 * sb * cb;
    c[75] = c2a * cb * cb + 0.5 * c2a * sb * sb;
    c[60] = 0.5 * c2a * s2b;
    c[45] = rt34 * c2a * sb * sb;
    c[30] = -s2a * sb;
    c[15] = -s2a * cb;
    c[72] = -0.5 * ca * s2b;
    c[57] = ca * c2b;
    c[42] = rt34 * ca * s2b;
    c[27] = -sa * cb;
    c[12] = sa * sb;
    c[69] = rt13 * sb * sb * 1.5;
    c[54] = -rt34 * s2b;
    c[39] = cb * cb - 0.5 * sb * sb;
    c[66] = -0.5 * sa * s2b;
    c[51] = sa * c2b;
    c[36] = rt34 * sa * s2b;
    c[21] = ca * cb;
    c[6] = -ca * sb;
    c[63] = s2a * cb * cb + 0.5 * s2a * sb * sb;
    c[48] = 0.5 * s2a * s2b;
    c[33] = rt34 * s2a * sb * sb;
    c[18] = c2a * sb;
    c[3] = c2a * cb;
}

double coeff(const std::array<double, 76>& c, int shell, int ml_slot,
             int component) {
    return c[static_cast<std::size_t>(shell + 3 * (ml_slot - 1)
                                      + 15 * (component - 1))];
}

void diat2(int z_a, double esa, double epa, double r12, int z_b, double esb,
           double epb, double s[4][4][4]) {
    static const int inmb[18] = {0, 1, 0, 2, 2, 3, 4,  5,  6,
                                 7, 0, 8, 8, 8, 9, 10, 11, 12};
    static const int iii[79] = {0, 1, 2, 4, 2, 4, 4, 2, 4, 4, 4, 2, 4, 4, 4, 4,
                                2, 4, 4, 4, 4, 4, 2, 4, 4, 4, 4, 4, 4, 3, 5, 5,
                                5, 5, 5, 5, 6, 3, 5, 5, 5, 5, 5, 5, 6, 6, 3, 5,
                                5, 5, 5, 5, 5, 6, 6, 6, 3, 5, 5, 5, 5, 5, 5, 6,
                                6, 6, 6, 3, 5, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6};

    const int jmax = std::max(inmb[z_a], inmb[z_b]);
    const int jmin = std::min(inmb[z_a], inmb[z_b]);
    const int nbond = (jmax * (jmax - 1)) / 2 + jmin;
    const int ii = iii[nbond];
    const double rab = r12 / kA0;
    OverlapSetState st;

    auto zero_s = [&]() {
        for (int i = 0; i < 4; ++i)
            for (int j = 0; j < 4; ++j)
                for (int k = 0; k < 4; ++k)
                    s[i][j][k] = 0.0;
    };
    zero_s();

    switch (ii) {
    default:
        st.set(esa, esb, z_a, z_b, rab, ii);
        s[1][1][1] = 0.25 * std::sqrt(std::pow(st.sa * st.sb * rab * rab, 3))
                     * (st.a[3] * st.b[1] - st.b[3] * st.a[1]);
        return;
    case 2: {
        st.set(esa, esb, z_a, z_b, rab, ii);
        double rab4 = std::pow(rab, 4) * 0.125;
        double w = std::sqrt(std::pow(st.sa, 3) * std::pow(st.sb, 5)) * rab4;
        s[1][1][1] = w * kInvSqrt3
                     * (st.a[4] * st.b[1] - st.b[4] * st.a[1]
                        + st.a[3] * st.b[2] - st.b[3] * st.a[2]);
        if (z_a > 1)
            st.set(epa, esb, z_a, z_b, rab, ii);
        if (z_b > 1)
            st.set(esa, epb, z_a, z_b, rab, ii);
        w = std::sqrt(std::pow(st.sa, 3) * std::pow(st.sb, 5)) * rab4;
        s[st.isp][st.ips][1] = w
                               * (st.a[3] * st.b[1] - st.b[3] * st.a[1]
                                  + st.a[4] * st.b[2] - st.b[4] * st.a[2]);
        return;
    }
    case 3: {
        st.set(esa, esb, z_a, z_b, rab, ii);
        double rab4 = std::pow(rab, 5) * 0.0625;
        double w =
            std::sqrt(std::pow(st.sa, 3) * std::pow(st.sb, 7) / 22.5) * rab4;
        s[1][1][1] = w
                     * (st.a[5] * st.b[1] - st.b[5] * st.a[1]
                        + (st.a[4] * st.b[2] - st.b[4] * st.a[2]) * 2.0);
        if (z_a > 1)
            st.set(epa, esb, z_a, z_b, rab, ii);
        if (z_b > 1)
            st.set(esa, epb, z_a, z_b, rab, ii);
        w = std::sqrt(std::pow(st.sa, 3) * std::pow(st.sb, 7) / 7.5) * rab4;
        s[st.isp][st.ips][1] =
            w
            * (st.a[4] * (st.b[1] + st.b[3]) - st.b[4] * (st.a[1] + st.a[3])
               + st.b[2] * (st.a[3] + st.a[5]) - st.a[2] * (st.b[3] + st.b[5]));
        return;
    }
    case 4: {
        st.set(esa, esb, z_a, z_b, rab, ii);
        double rab4 = std::pow(rab, 5) * 0.0625;
        double w = std::sqrt(std::pow(st.sa * st.sb, 5)) * rab4;
        s[1][1][1] =
            w
            * (st.a[5] * st.b[1] + st.b[5] * st.a[1] - 2.0 * st.a[3] * st.b[3])
            / 3.0;
        st.set(esa, epb, z_a, z_b, rab, ii);
        if (z_a > z_b)
            st.set(epa, esb, z_a, z_b, rab, ii);
        w = std::sqrt(std::pow(st.sa * st.sb, 5)) * rab4;
        double d =
            st.a[4] * (st.b[1] - st.b[3]) - st.a[2] * (st.b[3] - st.b[5]);
        double e =
            st.b[4] * (st.a[1] - st.a[3]) - st.b[2] * (st.a[3] - st.a[5]);
        s[st.isp][st.ips][1] = w * kInvSqrt3 * (d + e);
        st.set(epa, esb, z_a, z_b, rab, ii);
        if (z_a > z_b)
            st.set(esa, epb, z_a, z_b, rab, ii);
        w = std::sqrt(std::pow(st.sa * st.sb, 5)) * rab4;
        d = st.a[4] * (st.b[1] - st.b[3]) - st.a[2] * (st.b[3] - st.b[5]);
        e = st.b[4] * (st.a[1] - st.a[3]) - st.b[2] * (st.a[3] - st.a[5]);
        s[st.ips][st.isp][1] = w * kInvSqrt3 * (d - e);
        st.set(epa, epb, z_a, z_b, rab, ii);
        w = std::sqrt(std::pow(st.sa * st.sb, 5)) * rab4;
        s[2][2][1] =
            -w
            * (st.b[3] * (st.a[5] + st.a[1]) - st.a[3] * (st.b[5] + st.b[1]));
        s[2][2][2] =
            0.5 * w
            * (st.a[5] * (st.b[1] - st.b[3]) - st.b[5] * (st.a[1] - st.a[3])
               - st.a[3] * st.b[1] + st.b[3] * st.a[1]);
        return;
    }
    case 5: {
        st.set(esa, esb, z_a, z_b, rab, ii);
        double rab6 = std::pow(rab, 6) * 0.03125 / std::sqrt(7.5);
        double w = std::sqrt(std::pow(st.sa, 5) * std::pow(st.sb, 7)) * rab6;
        s[1][1][1] = w
                     * (st.a[6] * st.b[1] + st.a[5] * st.b[2]
                        - 2.0 * (st.a[4] * st.b[3] + st.a[3] * st.b[4])
                        + st.a[2] * st.b[5] + st.a[1] * st.b[6])
                     / 3.0;
        st.set(esa, epb, z_a, z_b, rab, ii);
        if (z_a > z_b)
            st.set(epa, esb, z_a, z_b, rab, ii);
        w = std::sqrt(std::pow(st.sa, 5) * std::pow(st.sb, 7)) * rab6;
        s[st.isp][st.ips][1] =
            w * kInvSqrt3
            * (st.a[6] * st.b[2] + st.a[5] * st.b[1]
               - 2.0 * (st.a[4] * st.b[4] + st.a[3] * st.b[3])
               + st.a[2] * st.b[6] + st.a[1] * st.b[5]);
        st.set(epa, esb, z_a, z_b, rab, ii);
        if (z_a > z_b)
            st.set(esa, epb, z_a, z_b, rab, ii);
        w = std::sqrt(std::pow(st.sa, 5) * std::pow(st.sb, 7)) * rab6;
        s[st.ips][st.isp][1] = -w * kInvSqrt3
                               * (st.a[5] * (2.0 * st.b[3] - st.b[1])
                                  - st.b[5] * (2.0 * st.a[3] - st.a[1])
                                  - st.a[2] * (st.b[6] - 2.0 * st.b[4])
                                  + st.b[2] * (st.a[6] - 2.0 * st.a[4]));
        st.set(epa, epb, z_a, z_b, rab, ii);
        w = std::sqrt(std::pow(st.sa, 5) * std::pow(st.sb, 7)) * rab6;
        s[2][2][1] =
            -w
            * (st.b[4] * (st.a[1] + st.a[5]) - st.a[4] * (st.b[1] + st.b[5])
               + st.b[3] * (st.a[2] + st.a[6]) - st.a[3] * (st.b[2] + st.b[6]));
        s[2][2][2] =
            0.5 * w
            * (st.a[6] * (st.b[1] - st.b[3]) - st.b[6] * (st.a[1] - st.a[3])
               + st.a[5] * (st.b[2] - st.b[4]) - st.b[5] * (st.a[2] - st.a[4])
               - st.a[4] * st.b[1] + st.b[4] * st.a[1] - st.a[3] * st.b[2]
               + st.b[3] * st.a[2]);
        return;
    }
    case 6: {
        st.set(esa, esb, z_a, z_b, rab, ii);
        double rab4 = std::pow(rab, 7) / 480.0;
        double w = std::sqrt(std::pow(st.sa * st.sb, 7)) * rab4;
        s[1][1][1] =
            w
            * (st.a[7] * st.b[1] - 3.0 * (st.a[5] * st.b[3] - st.a[3] * st.b[5])
               - st.a[1] * st.b[7])
            / 3.0;
        st.set(esa, epb, z_a, z_b, rab, ii);
        if (z_a > z_b)
            st.set(epa, esb, z_a, z_b, rab, ii);
        w = std::sqrt(std::pow(st.sa * st.sb, 7)) * rab4;
        double d = st.a[6] * (st.b[1] - st.b[3])
                   - 2.0 * st.a[4] * (st.b[3] - st.b[5])
                   + st.a[2] * (st.b[5] - st.b[7]);
        double e = st.b[6] * (st.a[1] - st.a[3])
                   - 2.0 * st.b[4] * (st.a[3] - st.a[5])
                   + st.b[2] * (st.a[5] - st.a[7]);
        s[st.isp][st.ips][1] = w * kInvSqrt3 * (d - e);
        st.set(epa, esb, z_a, z_b, rab, ii);
        if (z_a > z_b)
            st.set(esa, epb, z_a, z_b, rab, ii);
        w = std::sqrt(std::pow(st.sa * st.sb, 7)) * rab4;
        d = st.a[6] * (st.b[1] - st.b[3]) - 2.0 * st.a[4] * (st.b[3] - st.b[5])
            + st.a[2] * (st.b[5] - st.b[7]);
        e = st.b[6] * (st.a[1] - st.a[3]) - 2.0 * st.b[4] * (st.a[3] - st.a[5])
            + st.b[2] * (st.a[5] - st.a[7]);
        s[st.ips][st.isp][1] = -w * kInvSqrt3 * ((-d) - e);
        st.set(epa, epb, z_a, z_b, rab, ii);
        w = std::sqrt(std::pow(st.sa * st.sb, 7)) * rab4;
        d = st.a[3] * (st.b[7] + st.b[3] + st.b[3])
            - st.a[5] * (st.b[1] + st.b[5] + st.b[5]) - st.b[5] * st.a[1]
            + st.a[7] * st.b[3];
        s[2][2][1] = -w * d;
        d = st.a[7] * (st.b[1] - st.b[3]) + st.b[7] * (st.a[1] - st.a[3]);
        const double e2 = st.a[5] * (st.b[5] - st.b[3] - st.b[1])
                          + st.b[5] * (st.a[5] - st.a[3] - st.a[1])
                          + 2.0 * st.a[3] * st.b[3];
        s[2][2][2] = 0.5 * w * (d + e2);
        return;
    }
    }
}

} // namespace

double NDDOLocalMatrix9::at(int row, int col) const {
    if (row < 0 || row >= 9 || col < 0 || col >= 9) {
        throw std::runtime_error("NDDO local matrix index out of range");
    }
    return values[static_cast<std::size_t>(row) * 9u
                  + static_cast<std::size_t>(col)];
}

double& NDDOLocalMatrix9::at(int row, int col) {
    if (row < 0 || row >= 9 || col < 0 || col >= 9) {
        throw std::runtime_error("NDDO local matrix index out of range");
    }
    return values[static_cast<std::size_t>(row) * 9u
                  + static_cast<std::size_t>(col)];
}

NDDOLocalMatrix9 diatomic_overlap(const AM1Parameter& pi,
                                  const AM1Parameter& pj,
                                  const std::array<double, 3>& x_j_minus_i) {

    NDDOLocalMatrix9 out;
    const double r2 = x_j_minus_i[0] * x_j_minus_i[0]
                      + x_j_minus_i[1] * x_j_minus_i[1]
                      + x_j_minus_i[2] * x_j_minus_i[2];
    if (pi.principal_s == 0 || pj.principal_s == 0 || r2 >= kOverlapCutoff2
        || pi.n_orbitals == 0 || pj.n_orbitals == 0) {
        return out;
    }

    std::array<double, 76> c{};
    double r = 0.0;
    coe(x_j_minus_i[0], x_j_minus_i[1], x_j_minus_i[2], pi.n_orbitals,
        pj.n_orbitals, c, r);
    if (r < 0.001) {
        return out;
    }

    double s[4][4][4] = {};
    const int ia = std::min(pi.principal_s + 1, 3);
    int ib_shell_max = std::min(pj.principal_s + 1, 3);
    const int a = ia - 1;
    const int b = ib_shell_max - 1;

    if (use_diat2(pi.atomic_number, pi) && use_diat2(pj.atomic_number, pj)) {
        diat2(pi.atomic_number, pi.zs, pi.zp, r, pj.atomic_number, pj.zs, pj.zp,
              s);
    } else {
        const int newk = std::min(a, b);
        const int nk1 = newk + 1;
        for (int i = 1; i <= ia; ++i) {
            const int iss = i;
            ib_shell_max = b + 1;
            const int pq1 = principal_for(pi, i);
            for (int j = 1; j <= ib_shell_max; ++j) {
                const int jss = j;
                const int pq2 = principal_for(pj, j);
                for (int k = 1; k <= nk1; ++k) {
                    if (k > i || k > j)
                        continue;
                    const int kss = k;
                    const int p_i = std::max(pq1, iss);
                    const int p_j = std::max(pq2, jss);
                    s[i][j][k] =
                        ss_overlap(p_i, p_j, iss, jss, kss, zeta_for(pi, i),
                                   zeta_for(pj, j), r);
                }
            }
        }
    }

    static const int ival[4][6] = {
        {0, 0, 0, 0, 0, 0},
        {0, 1, 1, 1, 1, 0},
        {0, 0, 3, 4, 2, 0},
        {0, 9, 8, 7, 6, 5},
    };

    for (int i = 1; i <= ia; ++i) {
        const int kmin = 4 - i;
        const int kmax = 2 + i;
        for (int j = 1; j <= ib_shell_max; ++j) {
            double aa = 1.0;
            double bb = 1.0;
            if (j == 2) {
                aa = -1.0;
            } else if (j == 3) {
                bb = -1.0;
            }
            const int lmin = 4 - j;
            const int lmax = 2 + j;
            for (int k = kmin; k <= kmax; ++k) {
                for (int l = lmin; l <= lmax; ++l) {
                    const int ii = ival[i][k];
                    const int jj = ival[j][l];
                    if (ii == 0 || jj == 0) {
                        continue;
                    }
                    out.at(ii - 1, jj - 1) =
                        s[i][j][1] * coeff(c, i, k, 3) * coeff(c, j, l, 3) * aa
                        + s[i][j][2]
                              * (coeff(c, i, k, 4) * coeff(c, j, l, 4)
                                 + coeff(c, i, k, 2) * coeff(c, j, l, 2))
                              * bb
                        + s[i][j][3]
                              * (coeff(c, i, k, 5) * coeff(c, j, l, 5)
                                 + coeff(c, i, k, 1) * coeff(c, j, l, 1));
                }
            }
        }
    }
    return out;
}

NDDOLocalMatrix9 one_electron_two_center(const AM1Parameter& pi,
                                         const AM1Parameter& pj,
                                         const std::array<double, 3>& x_i,
                                         const std::array<double, 3>& x_j) {
    const std::array<double, 3> delta = {x_j[0] - x_i[0], x_j[1] - x_i[1],
                                         x_j[2] - x_i[2]};
    const double r2 =
        delta[0] * delta[0] + delta[1] * delta[1] + delta[2] * delta[2];
    if (r2 > kH1Cutoff2) {
        return {};
    }

    auto smat = diatomic_overlap(pi, pj, delta);
    std::array<double, 9> bi{};
    std::array<double, 9> bj{};
    bi[0] = 0.5 * pi.betas;
    bj[0] = 0.5 * pj.betas;
    for (int i = 1; i <= 3; ++i) {
        bi[static_cast<std::size_t>(i)] = 0.5 * pi.betap;
        bj[static_cast<std::size_t>(i)] = 0.5 * pj.betap;
    }
    for (int i = 4; i <= 8; ++i) {
        bi[static_cast<std::size_t>(i)] = 0.5 * pi.betad;
        bj[static_cast<std::size_t>(i)] = 0.5 * pj.betad;
    }

    for (int col = 0; col < pj.n_orbitals; ++col) {
        for (int row = 0; row < pi.n_orbitals; ++row) {
            smat.at(row, col) *= bi[static_cast<std::size_t>(row)]
                                 + bj[static_cast<std::size_t>(col)];
        }
    }
    return smat;
}

NDDOLocalMatrix9 diatomic_overlap(int z_i, int z_j,
                                  const std::array<double, 3>& x_j_minus_i) {
    return diatomic_overlap(require_am1_parameter(z_i),
                            require_am1_parameter(z_j), x_j_minus_i);
}

NDDOLocalMatrix9 one_electron_two_center(int z_i, int z_j,
                                         const std::array<double, 3>& x_i,
                                         const std::array<double, 3>& x_j) {
    return one_electron_two_center(require_am1_parameter(z_i),
                                   require_am1_parameter(z_j), x_i, x_j);
}

} // namespace semiempirical
} // namespace cpu
} // namespace ATV
