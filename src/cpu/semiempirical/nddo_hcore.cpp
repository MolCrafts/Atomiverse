/// @file src/cpu/semiempirical/nddo_hcore.cpp
/// @brief NDDO reference hcore.F90 one-center initialization port.

#include "atomiverse/cpu/semiempirical/nddo_hcore.h"

#include <array>
#include <cmath>
#include <stdexcept>
#include <string>

#include "atomiverse/core/frame_schema.h"
#include "atomiverse/cpu/semiempirical/am1_parameters.h"
#include "atomiverse/cpu/semiempirical/nddo_overlap.h"

namespace ATV {
namespace cpu {
namespace semiempirical {

namespace {

constexpr double kA0 = 0.529177210903;
constexpr double kEv = 27.211386245988;

double one_center_energy_for_local_ao(const AM1Parameter& p, int local_ao) {
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

KlopmanOhnoTerms ko_terms(const AM1Parameter& p) {
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
    out.ddp3 = p.qq * std::sqrt(2.0);
    if (p.atomic_number == 1) {
        out.po2 = 0.0;
        out.po3 = 0.0;
    }
    return out;
}

struct PairRotation {
    std::array<std::array<double, 3>, 3> sp{};
    std::array<std::array<std::array<double, 3>, 3>, 6> pp{};
};

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
        sa = 0.0;
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

using CoreTerms = std::array<std::array<double, 2>, 10>;

CoreTerms sp_core_terms(const AM1Parameter& pi, const AM1Parameter& pj,
                        double r_angstrom) {
    const auto ki = ko_terms(pi);
    const auto kj = ko_terms(pj);
    const double r = r_angstrom / kA0;
    const double r2 = r * r;

    CoreTerms core{};
    const double ssi = (ki.po9 + kj.po1) * (ki.po9 + kj.po1);
    const double ssj = (kj.po9 + ki.po1) * (kj.po9 + ki.po1);
    core[0][0] = -pj.core_charge * kEv / std::sqrt(r2 + ssj);
    core[0][1] = -pi.core_charge * kEv / std::sqrt(r2 + ssi);

    if (pi.n_orbitals >= 3) {
        const double ppj = (kj.po9 + ki.po7) * (kj.po9 + ki.po7);
        const double da = ki.ddp2;
        const double qa = ki.ddp3 / std::sqrt(2.0);
        const double twoqa = qa + qa;
        const double adj = (ki.po2 + kj.po9) * (ki.po2 + kj.po9);
        const double aqj = (ki.po3 + kj.po9) * (ki.po3 + kj.po9);
        const double xj1 = 1.0 / std::sqrt(r2 + ppj);
        const double xj2 = -0.5 / std::sqrt(r2 + aqj);
        const double xj3 = -0.5 / std::sqrt((r + da) * (r + da) + adj);
        const double xj4 = 0.5 / std::sqrt((r - da) * (r - da) + adj);
        const double xj5 = 0.25 / std::sqrt((r - twoqa) * (r - twoqa) + aqj);
        const double xj6 = 0.25 / std::sqrt((r + twoqa) * (r + twoqa) + aqj);
        const double xj7 = 0.5 / std::sqrt(r2 + twoqa * twoqa + aqj);
        const double aj2 = (xj3 + xj4) * kEv;
        const double aj3 = (xj1 + xj2 + xj5 + xj6) * kEv;
        const double aj4 = (xj1 + xj2 + xj7) * kEv;
        core[1][0] = -pj.core_charge * aj2;
        core[2][0] = -pj.core_charge * aj3;
        core[3][0] = -pj.core_charge * aj4;
    }

    if (pj.n_orbitals >= 3) {
        const double ppi = (ki.po9 + kj.po7) * (ki.po9 + kj.po7);
        const double db = kj.ddp2;
        const double qb = kj.ddp3 / std::sqrt(2.0);
        const double twoqb = qb + qb;
        const double adi = (kj.po2 + ki.po9) * (kj.po2 + ki.po9);
        const double aqi = (kj.po3 + ki.po9) * (kj.po3 + ki.po9);
        const double xi1 = 1.0 / std::sqrt(r2 + ppi);
        const double xi2 = -0.5 / std::sqrt(r2 + aqi);
        const double xi3 = -0.5 / std::sqrt((r + db) * (r + db) + adi);
        const double xi4 = 0.5 / std::sqrt((r - db) * (r - db) + adi);
        const double xi5 = 0.25 / std::sqrt((r - twoqb) * (r - twoqb) + aqi);
        const double xi6 = 0.25 / std::sqrt((r + twoqb) * (r + twoqb) + aqi);
        const double xi7 = 0.5 / std::sqrt(r2 + twoqb * twoqb + aqi);
        const double ai2 = -(xi3 + xi4) * kEv;
        const double ai3 = (xi1 + xi2 + xi5 + xi6) * kEv;
        const double ai4 = (xi1 + xi2 + xi7) * kEv;
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

double electron_nuclear_value(const CoreTerms& core, const PairRotation& rot,
                              int side, int local_i, int local_j) {
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
                                const NDDOAtom& atom, const CoreTerms& core,
                                const PairRotation& rot, int side,
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

double core_core_repulsion_ev(const AM1Parameter& pi, const AM1Parameter& pj,
                              double r_angstrom) {
    const auto ki = ko_terms(pi);
    const auto kj = ko_terms(pj);
    const double r_bohr = r_angstrom / kA0;
    const double aee = (ki.po9 + kj.po9) * (ki.po9 + kj.po9);
    const double gab = kEv / std::sqrt(r_bohr * r_bohr + aee);
    const double enuc = pi.core_charge * pj.core_charge * gab;

    double eni = std::exp(-pi.alp * r_angstrom);
    double enj = std::exp(-pj.alp * r_angstrom);
    double scale = eni + enj;
    const int nt = pi.atomic_number + pj.atomic_number;
    if (nt == 8 || nt == 9) {
        if (pi.atomic_number == 7 || pi.atomic_number == 8) {
            scale += (r_angstrom - 1.0) * eni;
        }
        if (pj.atomic_number == 7 || pj.atomic_number == 8) {
            scale += (r_angstrom - 1.0) * enj;
        }
    }
    double enuclr = std::abs(scale * enuc) + enuc;

    double gaussian = 0.0;
    int gaussian_count = 4;
    const int ni = pi.atomic_number;
    const int nj = pj.atomic_number;
    if ((ni == 5 || nj == 5)
        && (ni == 1 || nj == 1 || ni == 6 || nj == 6 || ni == 9 || nj == 9
            || ni == 17 || nj == 17 || ni == 35 || nj == 35 || ni == 53
            || nj == 53)) {
        const double prefactor = pi.core_charge * pj.core_charge / r_angstrom;
        if (ni == 1 || nj == 1) {
            double ax =
                10.0 * (r_angstrom - 0.832586) * (r_angstrom - 0.832586);
            if (ax <= 25.0) {
                gaussian += prefactor * 0.412253 * std::exp(-ax);
            }
            ax = 6.0 * (r_angstrom - 1.186220) * (r_angstrom - 1.186220);
            if (ax <= 25.0) {
                gaussian += prefactor * (-0.149917) * std::exp(-ax);
            }
        } else if (ni == 6 || nj == 6) {
            double ax = 8.0 * (r_angstrom - 1.063995) * (r_angstrom - 1.063995);
            if (ax <= 25.0) {
                gaussian += prefactor * 0.261751 * std::exp(-ax);
            }
            ax = 5.0 * (r_angstrom - 1.936492) * (r_angstrom - 1.936492);
            if (ax <= 25.0) {
                gaussian += prefactor * 0.050275 * std::exp(-ax);
            }
        } else {
            double ax = 9.0 * (r_angstrom - 0.819351) * (r_angstrom - 0.819351);
            if (ax <= 25.0) {
                gaussian += prefactor * 0.359244 * std::exp(-ax);
            }
            ax = 9.0 * (r_angstrom - 1.574414) * (r_angstrom - 1.574414);
            if (ax <= 25.0) {
                gaussian += prefactor * 0.074729 * std::exp(-ax);
            }
        }
        const AM1Parameter& other = ni == 5 ? pj : pi;
        for (int ig = 0; ig < 4; ++ig) {
            const double ax = other.guess2[ig] * (r_angstrom - other.guess3[ig])
                              * (r_angstrom - other.guess3[ig]);
            if (ax <= 25.0) {
                gaussian += prefactor * other.guess1[ig] * std::exp(-ax);
            }
        }
        gaussian_count = 0;
    }

    for (int ig = 0; ig < gaussian_count; ++ig) {
        if (std::abs(pi.guess1[ig]) > 0.0) {
            const double ax = pi.guess2[ig] * (r_angstrom - pi.guess3[ig])
                              * (r_angstrom - pi.guess3[ig]);
            if (ax <= 25.0) {
                gaussian += pi.core_charge * pj.core_charge / r_angstrom
                            * pi.guess1[ig] * std::exp(-ax);
            }
        }
        if (std::abs(pj.guess1[ig]) > 0.0) {
            const double ax = pj.guess2[ig] * (r_angstrom - pj.guess3[ig])
                              * (r_angstrom - pj.guess3[ig]);
            if (ax <= 25.0) {
                gaussian += pi.core_charge * pj.core_charge / r_angstrom
                            * pj.guess1[ig] * std::exp(-ax);
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
    return std::sqrt(dx * dx + dy * dy + dz * dz);
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
