/// @file cuda/semiempirical/nddo_one_electron.cuh
/// @brief Device scalar CHNO STO and core laws, used by values and derivatives.
#pragma once
#include "nddo_pair.cuh"
namespace ATV::cuda::semiempirical::internal::detail {
template <class Scalar>
struct OverlapSet {
    int isp, ips;
    double sa, sb;
    Scalar a[6], b[6];

    __device__ void set(double s1, double s2, int za, int zb, Scalar rab,
                        int ii) {
        isp = za <= zb ? 1 : 2;
        ips = za <= zb ? 2 : 1;
        sa = za <= zb ? s1 : s2;
        sb = za <= zb ? s2 : s1;
        const int k = ii == 4 ? 4 : ii + 1;
        const Scalar alpha = 0.5 * rab * (sa + sb);
        const Scalar beta = 0.5 * rab * (sb - sa);
        const Scalar c = exp(-alpha);
        a[1] = c / alpha;
        for (int i = 1; i <= k; ++i)
            a[i + 1] = (a[i] * i + c) / alpha;
        if (fabs(beta) <= 1e-6) {
            for (int i = 0; i <= k; ++i)
                b[i + 1] = i % 2 == 0 ? 2.0 / (i + 1) : 0.0;
        } else if (fabs(beta) <= 0.5) {
            constexpr double fact[7] = {1, 1, 2, 6, 24, 120, 720};
            for (int i = 0; i <= k; ++i) {
                Scalar sum = 0;
                for (int m = 0; m <= 6; ++m)
                    if ((m + i) % 2 == 0)
                        sum += pow(-beta, m) * 2.0 / (fact[m] * (m + i + 1));
                b[i + 1] = sum;
            }
        } else {
            const Scalar eb = exp(beta), emb = 1.0 / eb;
            b[1] = (eb - emb) / beta;
            for (int i = 1; i <= k; ++i)
                b[i + 1] = (i * b[i] + (i % 2 == 0 ? eb : -eb) - emb) / beta;
        }
    }
};

// ss, sp, ps, pp-sigma, pp-pi in the molecular coordinate system. Only
// CHNO's n=1/2 formulas are present; no unused general d-orbital machinery.
template <class Scalar>
__device__ void sto_overlap(const NDDOAtom& pi, const NDDOAtom& pj, Scalar r,
                            Scalar r2, Scalar* s) {
    for (int i = 0; i < 5; ++i)
        s[i] = 0;
    if (r2 >= 225.0 || r < 0.001)
        return;
    constexpr double inv_sqrt3 = 0.57735026918962576451;
    const int za = pi.atomic_number, zb = pj.atomic_number;
    const int ii = za == 1 && zb == 1 ? 1 : (za == 1 || zb == 1 ? 2 : 4);
    const Scalar rab = r / kA0;
    OverlapSet<Scalar> st;
    st.set(pi.zs, pj.zs, za, zb, rab, ii);
    if (ii == 1) {
        s[0] = 0.25 * sqrt(pow(st.sa * st.sb * rab * rab, 3))
               * (st.a[3] * st.b[1] - st.b[3] * st.a[1]);
        return;
    }
    if (ii == 2) {
        const Scalar rab4 = pow(rab, 4) * 0.125;
        Scalar w = sqrt(pow(st.sa, 3) * pow(st.sb, 5)) * rab4;
        s[0] = w * inv_sqrt3
               * (st.a[4] * st.b[1] - st.b[4] * st.a[1] + st.a[3] * st.b[2]
                  - st.b[3] * st.a[2]);
        if (za > 1)
            st.set(pi.zp, pj.zs, za, zb, rab, ii);
        if (zb > 1)
            st.set(pi.zs, pj.zp, za, zb, rab, ii);
        w = sqrt(pow(st.sa, 3) * pow(st.sb, 5)) * rab4;
        s[st.isp] = w
                    * (st.a[3] * st.b[1] - st.b[3] * st.a[1] + st.a[4] * st.b[2]
                       - st.b[4] * st.a[2]);
        return;
    }
    const Scalar rab5 = pow(rab, 5) * 0.0625;
    Scalar w = sqrt(pow(st.sa * st.sb, 5)) * rab5;
    s[0] = w * (st.a[5] * st.b[1] + st.b[5] * st.a[1] - 2.0 * st.a[3] * st.b[3])
           / 3.0;
    st.set(pi.zs, pj.zp, za, zb, rab, ii);
    if (za > zb)
        st.set(pi.zp, pj.zs, za, zb, rab, ii);
    w = sqrt(pow(st.sa * st.sb, 5)) * rab5;
    Scalar d = st.a[4] * (st.b[1] - st.b[3]) - st.a[2] * (st.b[3] - st.b[5]);
    Scalar e = st.b[4] * (st.a[1] - st.a[3]) - st.b[2] * (st.a[3] - st.a[5]);
    s[st.isp] = w * inv_sqrt3 * (d + e);
    st.set(pi.zp, pj.zs, za, zb, rab, ii);
    if (za > zb)
        st.set(pi.zs, pj.zp, za, zb, rab, ii);
    w = sqrt(pow(st.sa * st.sb, 5)) * rab5;
    d = st.a[4] * (st.b[1] - st.b[3]) - st.a[2] * (st.b[3] - st.b[5]);
    e = st.b[4] * (st.a[1] - st.a[3]) - st.b[2] * (st.a[3] - st.a[5]);
    s[st.ips] = w * inv_sqrt3 * (d - e);
    st.set(pi.zp, pj.zp, za, zb, rab, ii);
    w = sqrt(pow(st.sa * st.sb, 5)) * rab5;
    s[3] = -w * (st.b[3] * (st.a[5] + st.a[1]) - st.a[3] * (st.b[5] + st.b[1]));
    s[4] = 0.5 * w
           * (st.a[5] * (st.b[1] - st.b[3]) - st.b[5] * (st.a[1] - st.a[3])
              - st.a[3] * st.b[1] + st.b[3] * st.a[1]);
}

template <class Scalar>
__device__ Scalar core_repulsion(const NDDOAtom& pi, const NDDOAtom& pj,
                                 Scalar r, Scalar ss) {
    const Scalar enuc = pi.core_charge * pj.core_charge * ss;
    const Scalar eni = exp(-pi.alp * r), enj = exp(-pj.alp * r);
    Scalar scale = eni + enj;
    const int zsum = pi.atomic_number + pj.atomic_number;
    if (zsum == 8 || zsum == 9) {
        if (pi.atomic_number == 7 || pi.atomic_number == 8)
            scale += (r - 1.0) * eni;
        if (pj.atomic_number == 7 || pj.atomic_number == 8)
            scale += (r - 1.0) * enj;
    }
    Scalar gaussian = 0;
    const Scalar prefactor = pi.core_charge * pj.core_charge / r;
    for (int g = 0; g < 4; ++g) {
        const Scalar ai =
            pi.guess2[g] * (r - pi.guess3[g]) * (r - pi.guess3[g]);
        const Scalar aj =
            pj.guess2[g] * (r - pj.guess3[g]) * (r - pj.guess3[g]);
        if (pi.guess1[g] != 0 && ai <= 25.0)
            gaussian += prefactor * pi.guess1[g] * exp(-ai);
        if (pj.guess1[g] != 0 && aj <= 25.0)
            gaussian += prefactor * pj.guess1[g] * exp(-aj);
    }
    return enuc + fabs(scale * enuc) + gaussian;
}

template <class Scalar>
__device__ Scalar resonance(const NDDOAtom& ai, const NDDOAtom& aj, int a,
                            int b, const Scalar* axes, const Scalar* overlap) {
    Scalar s = 0;
    if (a == 0 && b == 0)
        s = overlap[0];
    else if (a == 0)
        s = -overlap[1] * axes[b - 1];
    else if (b == 0)
        s = overlap[2] * axes[a - 1];
    else
        s = -overlap[3] * axes[a - 1] * axes[b - 1]
            + overlap[4] * ((a == b ? 1.0 : 0.0) - axes[a - 1] * axes[b - 1]);
    return 0.5
           * ((a == 0 ? ai.betas : ai.betap) + (b == 0 ? aj.betas : aj.betap))
           * s;
}

template <class Scalar>
__device__ Scalar attraction(int side, int mu, int nu, const Scalar* radial,
                             const Scalar* axes) {
    Scalar value;
    if (mu == 0 && nu == 0)
        value = radial[0];
    else if (mu == 0 || nu == 0)
        value = radial[side == 0 ? 1 : 4] * axes[(mu == 0 ? nu : mu) - 1];
    else {
        const Scalar direction = axes[mu - 1] * axes[nu - 1];
        value =
            radial[side == 0 ? 2 : 10] * direction
            + radial[side == 0 ? 3 : 11] * ((mu == nu ? 1.0 : 0.0) - direction);
    }
    return value;
}

} // namespace ATV::cuda::semiempirical::internal::detail
