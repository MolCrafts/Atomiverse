/// @file core/angle_cosine.h
/// @brief LAMMPS `angle_style cosine`, one formula for the CPU and CUDA kernels.
///
/// \(U = K\,(1 + \cos\theta)\), minimum at \(\theta = \pi\) (a straight
/// chain) — the Kremer–Grest bending stiffness. \(K\) is an energy [eV].
///
/// Ported from LAMMPS `src/MOLECULE/angle_cosine.cpp`
/// (`AngleCosine::compute`), checkout SHA e8c4c278a8:
///
///   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
///   https://www.lammps.org/, Sandia National Laboratories
///   Copyright (2003) Sandia Corporation. Under the terms of Contract
///   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
///   certain rights in this software. This software is distributed under
///   the GNU General Public License.
///
/// The force decomposition is LAMMPS's: with \(c = \cos\theta\) clamped to
/// \([-1, 1]\), `a11 = K c / r1²`, `a12 = -K / (r1 r2)`, `a22 = K c / r2²`,
/// \(F_1 = a_{11}\,d_1 + a_{12}\,d_2\), \(F_3 = a_{22}\,d_2 + a_{12}\,d_1\),
/// \(F_2 = -(F_1 + F_3)\). There is no \(1/\sin\theta\), so the style has no
/// singularity at \(\theta = 0\) or \(\pi\). The virial (LAMMPS `ev_tally`)
/// is \(d_1 \otimes F_1 + d_2 \otimes F_3\).
#pragma once

#include <cmath>

namespace ATV {

/// @brief Energy and the two end-atom forces of one cosine angle.
template <typename T>
struct AngleCosineTerm {
    T energy; ///< \(K(1+\cos\theta)\) [eV].
    T f1[3];  ///< Force on the first end atom [eV/Å].
    T f3[3];  ///< Force on the third end atom [eV/Å]; centre gets −(f1+f3).
};

#if defined(__CUDACC__)
#define ATV_ANGLE_COSINE_HD __host__ __device__
#else
#define ATV_ANGLE_COSINE_HD
#endif

/// @brief Evaluate one `angle_style cosine` tuple.
/// @param K  Angle stiffness [eV].
/// @param d1 Minimum-image vector end atom 1 − centre [Å].
/// @param d2 Minimum-image vector end atom 3 − centre [Å].
/// @return Energy [eV] and the forces on the two end atoms [eV/Å].
template <typename T>
ATV_ANGLE_COSINE_HD inline AngleCosineTerm<T>
angle_cosine_term(T K, const T d1[3], const T d2[3]) {
    const T rsq1 = d1[0] * d1[0] + d1[1] * d1[1] + d1[2] * d1[2];
    const T rsq2 = d2[0] * d2[0] + d2[1] * d2[1] + d2[2] * d2[2];
#if defined(__CUDA_ARCH__)
    const T r1 = sqrt(rsq1);
    const T r2 = sqrt(rsq2);
#else
    const T r1 = std::sqrt(rsq1);
    const T r2 = std::sqrt(rsq2);
#endif
    T c = (d1[0] * d2[0] + d1[1] * d2[1] + d1[2] * d2[2]) / (r1 * r2);
    if (c > T(1))
        c = T(1);
    if (c < T(-1))
        c = T(-1);

    const T a11 = K * c / rsq1;
    const T a12 = -K / (r1 * r2);
    const T a22 = K * c / rsq2;

    AngleCosineTerm<T> out;
    out.energy = K * (T(1) + c);
    for (int d = 0; d < 3; ++d) {
        out.f1[d] = a11 * d1[d] + a12 * d2[d];
        out.f3[d] = a22 * d2[d] + a12 * d1[d];
    }
    return out;
}

#undef ATV_ANGLE_COSINE_HD

} // namespace ATV
