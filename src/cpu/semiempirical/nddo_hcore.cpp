/// @file src/cpu/semiempirical/nddo_hcore.cpp
/// @brief NDDO reference hcore.F90 one-center initialization port.

#include "atomiverse/cpu/semiempirical/nddo_hcore.h"

#include "atomiverse/cpu/semiempirical/internal/nddo_derivative.h"
#include <array>
#include <cmath>
#include <stdexcept>
#include <string>

#include "atomiverse/core/frame_schema.h"
#include "atomiverse/cpu/semiempirical/nddo_overlap.h"
#include "atomiverse/semiempirical/am1_parameters.h"

namespace ATV {
namespace cpu {
namespace semiempirical {

namespace {
using internal::abs;
using internal::exp;
using internal::sqrt;
using std::abs;
using std::exp;
using std::sqrt;

constexpr double kA0 = 0.529177210903;
constexpr double kEv = 27.211386245988;

double one_center_energy_for_local_ao(const ATV::semiempirical::AM1Parameter& p,
                                      int local_ao) {
    if (local_ao == 0) {
        return p.uss;
    }
    if (local_ao >= 1 && local_ao <= 3) {
        return p.upp;
    }
    if (local_ao >= 4 && local_ao <= 8) {
        return p.udd;
    }
    throw std::runtime_error(
        "AM1 local AO index is outside NDDO reference order");
}

struct KlopmanOhnoTerms {
    double po1 = 0.0;
    double po2 = 0.0;
    double po3 = 0.0;
    double po7 = 0.0;
    double po9 = 0.0;
    double ddp2 = 0.0;
    double ddp3 = 0.0;
};

KlopmanOhnoTerms ko_terms(const ATV::semiempirical::AM1Parameter& p) {
    KlopmanOhnoTerms out;
    double am = p.am;
    if (am < 1.0e-4) {
        am = 1.0;
    }
    out.po1 = 0.5 / am;
    if (p.ad > 1.0e-5) {
        out.po2 = 0.5 / p.ad;
    }
    if (p.aq > 1.0e-5) {
        out.po3 = 0.5 / p.aq;
    }
    out.po7 = out.po1;
    out.po9 = out.po1;
    out.ddp2 = p.dd;
    out.ddp3 = p.qq * sqrt(2.0);
    if (p.atomic_number == 1) {
        out.po2 = 0.0;
        out.po3 = 0.0;
    }
    return out;
}

template <class Scalar>
struct PairRotation {
    std::array<std::array<Scalar, 3>, 3> sp{};
    std::array<std::array<std::array<Scalar, 3>, 3>, 6> pp{};
};
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

template <class Scalar>
using CoreTerms = std::array<std::array<Scalar, 2>, 10>;

template <class Scalar>
CoreTerms<Scalar> sp_core_terms(const ATV::semiempirical::AM1Parameter& pi,
                                const ATV::semiempirical::AM1Parameter& pj,
                                Scalar r_angstrom) {
    const auto ki = ko_terms(pi);
    const auto kj = ko_terms(pj);
    const Scalar r = r_angstrom / kA0;
    const Scalar r2 = r * r;

    CoreTerms<Scalar> core{};
    const Scalar ssi = (ki.po9 + kj.po1) * (ki.po9 + kj.po1);
    const Scalar ssj = (kj.po9 + ki.po1) * (kj.po9 + ki.po1);
    core[0][0] = -pj.core_charge * kEv / sqrt(r2 + ssj);
    core[0][1] = -pi.core_charge * kEv / sqrt(r2 + ssi);

    if (pi.n_orbitals >= 3) {
        const Scalar ppj = (kj.po9 + ki.po7) * (kj.po9 + ki.po7);
        const Scalar da = ki.ddp2;
        const Scalar qa = ki.ddp3 / sqrt(2.0);
        const Scalar twoqa = qa + qa;
        const Scalar adj = (ki.po2 + kj.po9) * (ki.po2 + kj.po9);
        const Scalar aqj = (ki.po3 + kj.po9) * (ki.po3 + kj.po9);
        const Scalar xj1 = 1.0 / sqrt(r2 + ppj);
        const Scalar xj2 = -0.5 / sqrt(r2 + aqj);
        const Scalar xj3 = -0.5 / sqrt((r + da) * (r + da) + adj);
        const Scalar xj4 = 0.5 / sqrt((r - da) * (r - da) + adj);
        const Scalar xj5 = 0.25 / sqrt((r - twoqa) * (r - twoqa) + aqj);
        const Scalar xj6 = 0.25 / sqrt((r + twoqa) * (r + twoqa) + aqj);
        const Scalar xj7 = 0.5 / sqrt(r2 + twoqa * twoqa + aqj);
        const Scalar aj2 = (xj3 + xj4) * kEv;
        const Scalar aj3 = (xj1 + xj2 + xj5 + xj6) * kEv;
        const Scalar aj4 = (xj1 + xj2 + xj7) * kEv;
        core[1][0] = -pj.core_charge * aj2;
        core[2][0] = -pj.core_charge * aj3;
        core[3][0] = -pj.core_charge * aj4;
    }

    if (pj.n_orbitals >= 3) {
        const Scalar ppi = (ki.po9 + kj.po7) * (ki.po9 + kj.po7);
        const Scalar db = kj.ddp2;
        const Scalar qb = kj.ddp3 / sqrt(2.0);
        const Scalar twoqb = qb + qb;
        const Scalar adi = (kj.po2 + ki.po9) * (kj.po2 + ki.po9);
        const Scalar aqi = (kj.po3 + ki.po9) * (kj.po3 + ki.po9);
        const Scalar xi1 = 1.0 / sqrt(r2 + ppi);
        const Scalar xi2 = -0.5 / sqrt(r2 + aqi);
        const Scalar xi3 = -0.5 / sqrt((r + db) * (r + db) + adi);
        const Scalar xi4 = 0.5 / sqrt((r - db) * (r - db) + adi);
        const Scalar xi5 = 0.25 / sqrt((r - twoqb) * (r - twoqb) + aqi);
        const Scalar xi6 = 0.25 / sqrt((r + twoqb) * (r + twoqb) + aqi);
        const Scalar xi7 = 0.5 / sqrt(r2 + twoqb * twoqb + aqi);
        const Scalar ai2 = -(xi3 + xi4) * kEv;
        const Scalar ai3 = (xi1 + xi2 + xi5 + xi6) * kEv;
        const Scalar ai4 = (xi1 + xi2 + xi7) * kEv;
        core[1][1] = -pi.core_charge * ai2;
        core[2][1] = -pi.core_charge * ai3;
        core[3][1] = -pi.core_charge * ai4;
    }

    return core;
}

int indpp(int a, int b) {
    static constexpr int kIndpp[3][3] = {{0, 3, 4}, {3, 1, 5}, {4, 5, 2}};
    return kIndpp[a][b];
}

template <class Scalar>
Scalar electron_nuclear_value(const CoreTerms<Scalar>& core,
                              const PairRotation<Scalar>& rot, int side,
                              int local_i, int local_j) {
    if (local_i == 0 && local_j == 0) {
        return core[0][side];
    }
    if (local_i >= 4 || local_j >= 4) {
        throw std::runtime_error(
            "AM1 d-orbital electron-nuclear integrals are not ported yet");
    }
    if (local_j == 0) {
        return rot.sp[0][local_i - 1] * core[1][side];
    }
    const int ipp = indpp(local_i - 1, local_j - 1);
    return core[2][side] * rot.pp[ipp][0][0]
           + core[3][side] * (rot.pp[ipp][1][1] + rot.pp[ipp][2][2]);
}

void add_electron_nuclear_block(std::vector<double>& hcore,
                                const NDDOAtom& atom,
                                const CoreTerms<double>& core,
                                const PairRotation<double>& rot, int side,
                                double scale) {
    for (int local_i = 0; local_i < atom.n_orbitals; ++local_i) {
        const int ao_i = atom.first_ao + local_i;
        for (int local_j = 0; local_j <= local_i; ++local_j) {
            const int ao_j = atom.first_ao + local_j;
            hcore[packed_index(ao_i, ao_j)] +=
                scale
                * electron_nuclear_value(core, rot, side, local_i, local_j);
        }
    }
}

template <class Scalar>
Scalar core_core_repulsion_ev(const ATV::semiempirical::AM1Parameter& pi,
                              const ATV::semiempirical::AM1Parameter& pj,
                              Scalar r_angstrom) {
    const auto ki = ko_terms(pi);
    const auto kj = ko_terms(pj);
    const Scalar r_bohr = r_angstrom / kA0;
    const Scalar aee = (ki.po9 + kj.po9) * (ki.po9 + kj.po9);
    const Scalar gab = kEv / sqrt(r_bohr * r_bohr + aee);
    const Scalar enuc = pi.core_charge * pj.core_charge * gab;

    Scalar eni = exp(-pi.alp * r_angstrom);
    Scalar enj = exp(-pj.alp * r_angstrom);
    Scalar scale = eni + enj;
    const int nt = pi.atomic_number + pj.atomic_number;
    if (nt == 8 || nt == 9) {
        if (pi.atomic_number == 7 || pi.atomic_number == 8) {
            scale += (r_angstrom - 1.0) * eni;
        }
        if (pj.atomic_number == 7 || pj.atomic_number == 8) {
            scale += (r_angstrom - 1.0) * enj;
        }
    }
    Scalar enuclr = abs(scale * enuc) + enuc;

    Scalar gaussian = 0.0;
    int gaussian_count = 4;
    const int ni = pi.atomic_number;
    const int nj = pj.atomic_number;
    if ((ni == 5 || nj == 5)
        && (ni == 1 || nj == 1 || ni == 6 || nj == 6 || ni == 9 || nj == 9
            || ni == 17 || nj == 17 || ni == 35 || nj == 35 || ni == 53
            || nj == 53)) {
        const Scalar prefactor = pi.core_charge * pj.core_charge / r_angstrom;
        if (ni == 1 || nj == 1) {
            Scalar ax =
                10.0 * (r_angstrom - 0.832586) * (r_angstrom - 0.832586);
            if (ax <= 25.0) {
                gaussian += prefactor * 0.412253 * exp(-ax);
            }
            ax = 6.0 * (r_angstrom - 1.186220) * (r_angstrom - 1.186220);
            if (ax <= 25.0) {
                gaussian += prefactor * (-0.149917) * exp(-ax);
            }
        } else if (ni == 6 || nj == 6) {
            Scalar ax = 8.0 * (r_angstrom - 1.063995) * (r_angstrom - 1.063995);
            if (ax <= 25.0) {
                gaussian += prefactor * 0.261751 * exp(-ax);
            }
            ax = 5.0 * (r_angstrom - 1.936492) * (r_angstrom - 1.936492);
            if (ax <= 25.0) {
                gaussian += prefactor * 0.050275 * exp(-ax);
            }
        } else {
            Scalar ax = 9.0 * (r_angstrom - 0.819351) * (r_angstrom - 0.819351);
            if (ax <= 25.0) {
                gaussian += prefactor * 0.359244 * exp(-ax);
            }
            ax = 9.0 * (r_angstrom - 1.574414) * (r_angstrom - 1.574414);
            if (ax <= 25.0) {
                gaussian += prefactor * 0.074729 * exp(-ax);
            }
        }
        const ATV::semiempirical::AM1Parameter& other = ni == 5 ? pj : pi;
        for (int ig = 0; ig < 4; ++ig) {
            const Scalar ax = other.guess2[ig] * (r_angstrom - other.guess3[ig])
                              * (r_angstrom - other.guess3[ig]);
            if (ax <= 25.0) {
                gaussian += prefactor * other.guess1[ig] * exp(-ax);
            }
        }
        gaussian_count = 0;
    }

    for (int ig = 0; ig < gaussian_count; ++ig) {
        if (abs(pi.guess1[ig]) > 0.0) {
            const Scalar ax = pi.guess2[ig] * (r_angstrom - pi.guess3[ig])
                              * (r_angstrom - pi.guess3[ig]);
            if (ax <= 25.0) {
                gaussian += pi.core_charge * pj.core_charge / r_angstrom
                            * pi.guess1[ig] * exp(-ax);
            }
        }
        if (abs(pj.guess1[ig]) > 0.0) {
            const Scalar ax = pj.guess2[ig] * (r_angstrom - pj.guess3[ig])
                              * (r_angstrom - pj.guess3[ig]);
            if (ax <= 25.0) {
                gaussian += pi.core_charge * pj.core_charge / r_angstrom
                            * pj.guess1[ig] * exp(-ax);
            }
        }
    }
    return enuclr + gaussian;
}

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
    return sqrt(dx * dx + dy * dy + dz * dz);
}

} // namespace

NDDOHCoreResult NDDOHCoreBuilder::build(const NDDOSystem& system,
                                        const cpu::Frame& frame) const {
    if (frame.nlocal() != static_cast<int>(system.atoms.size())) {
        throw std::runtime_error(
            "NDDOHCoreBuilder::build requires matching frame/system atoms");
    }
    return build(system, frame.span(kAtom_x), frame.span(kAtom_y),
                 frame.span(kAtom_z));
}

NDDOHCoreResult NDDOHCoreBuilder::build(const NDDOSystem& system,
                                        const StorageReal* sp_x,
                                        const StorageReal* sp_y,
                                        const StorageReal* sp_z) const {
    if (sp_x == nullptr || sp_y == nullptr || sp_z == nullptr) {
        throw std::runtime_error(
            "NDDOHCoreBuilder::build requires non-null coordinate spans");
    }

    auto result = build_one_center(system);
    result.includes_two_center_terms = true;

    for (std::size_t i = 0; i < system.atoms.size(); ++i) {
        const auto& atom_i = system.atoms[i];
        if (atom_i.n_orbitals > 4) {
            throw std::runtime_error(
                "AM1 d-orbital two-center Hcore integrals are not ported yet");
        }
        const auto* pi = atom_i.params;
        if (pi == nullptr) {
            throw std::runtime_error(
                "AM1 parameters unavailable for Hcore atom");
        }
        const auto xi = atom_xyz(sp_x, sp_y, sp_z, static_cast<int>(i));
        for (std::size_t j = 0; j < i; ++j) {
            const auto& atom_j = system.atoms[j];
            if (atom_j.n_orbitals > 4) {
                throw std::runtime_error(
                    "AM1 d-orbital two-center Hcore integrals are not ported "
                    "yet");
            }
            const auto* pj = atom_j.params;
            if (pj == nullptr) {
                throw std::runtime_error(
                    "AM1 parameters unavailable for Hcore atom");
            }
            const auto xj = atom_xyz(sp_x, sp_y, sp_z, static_cast<int>(j));
            const double rij = distance(xi, xj);
            if (rij < 1.0e-12) {
                continue;
            }

            const auto h1 = one_electron_two_center(*pi, *pj, xi, xj);
            for (int local_i = 0; local_i < atom_i.n_orbitals; ++local_i) {
                const int ao_i = atom_i.first_ao + local_i;
                for (int local_j = 0; local_j < atom_j.n_orbitals; ++local_j) {
                    const int ao_j = atom_j.first_ao + local_j;
                    result.packed_hcore[packed_index(ao_i, ao_j)] +=
                        h1.at(local_i, local_j);
                }
            }

            const auto rot = pair_rotation(xi, xj);
            const auto core = sp_core_terms(*pi, *pj, rij);
            add_electron_nuclear_block(result.packed_hcore, atom_i, core, rot,
                                       0, 1.0);
            add_electron_nuclear_block(result.packed_hcore, atom_j, core, rot,
                                       1, 1.0);
            result.nuclear_repulsion_ev +=
                core_core_repulsion_ev(*pi, *pj, rij);
        }
    }

    return result;
}

std::array<internal::NDDOPairHCoreDerivative, 3>
internal::hcore_pair_derivative(const NDDOSystem& pair,
                                const std::array<double, 3>& xi_value,
                                const std::array<double, 3>& xj_value) {
    // pair contains j at 0 and i at 1, matching the normal Hcore pair walk.
    if (pair.atoms.size() != 2)
        throw std::runtime_error(
            "NDDO Hcore pair derivative requires exactly two atoms");
    const auto& ai = pair.atoms[1];
    const auto& aj = pair.atoms[0];
    if (ai.n_orbitals < 1 || ai.n_orbitals > 4 || aj.n_orbitals < 1
        || aj.n_orbitals > 4 || aj.first_ao != 0 || ai.first_ao != aj.n_orbitals
        || pair.n_orbitals != ai.n_orbitals + aj.n_orbitals || !ai.params
        || !aj.params)
        throw std::runtime_error(
            "NDDO Hcore pair derivative requires a packed s/p pair");
    const auto& pi = *ai.params;
    const auto& pj = *aj.params;
    const auto h1 =
        one_electron_two_center_derivative(pi, pj, xi_value, xj_value);
    std::array<NDDODerivative, 3> xi, xj;
    for (int d = 0; d < 3; ++d) {
        xi[d] = xi_value[d];
        xj[d] = xj_value[d];
        xi[d].gradient[d] = 1.0;
    }
    NDDODerivative r2;
    for (int d = 0; d < 3; ++d)
        r2 += (xi[d] - xj[d]) * (xi[d] - xj[d]);
    const auto r = sqrt(r2);
    const auto core = sp_core_terms(pi, pj, r);
    const auto rot = pair_rotation(xi, xj);
    const auto enuc = core_core_repulsion_ev(pi, pj, r);
    std::array<NDDOPairHCoreDerivative, 3> out{};
    for (int d = 0; d < 3; ++d) {
        auto& h = out[d];
        h.nuclear_repulsion_ev = enuc.gradient[d];
        for (int mu = 0; mu < ai.n_orbitals; ++mu)
            for (int nu = 0; nu < aj.n_orbitals; ++nu)
                h.packed_hcore[packed_index(
                    ai.first_ao + mu, aj.first_ao + nu)] = h1[d].at(mu, nu);
        for (int side = 0; side < 2; ++side) {
            const auto& atom = side == 0 ? ai : aj;
            for (int mu = 0; mu < atom.n_orbitals; ++mu)
                for (int nu = 0; nu <= mu; ++nu)
                    h.packed_hcore[packed_index(atom.first_ao + mu,
                                                atom.first_ao + nu)] =
                        electron_nuclear_value(core, rot, side, mu, nu)
                            .gradient[d];
        }
    }
    return out;
}

NDDOHCoreResult
NDDOHCoreBuilder::build_one_center(const NDDOSystem& system) const {
    NDDOHCoreResult result;
    result.packed_hcore.assign(
        static_cast<std::size_t>(system.n_orbitals)
            * static_cast<std::size_t>(system.n_orbitals + 1) / 2u,
        0.0);

    for (const auto& atom : system.atoms) {
        const auto* p = atom.params;
        if (p == nullptr) {
            throw std::runtime_error(
                "AM1 parameters unavailable for Hcore atom");
        }
        for (int ao = atom.first_ao; ao <= atom.last_ao; ++ao) {
            const int local = ao - atom.first_ao;
            result.packed_hcore[packed_index(ao, ao)] =
                one_center_energy_for_local_ao(*p, local);
        }
    }
    return result;
}

} // namespace semiempirical
} // namespace cpu
} // namespace ATV
