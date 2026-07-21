/// @file src/cpu/semiempirical/nddo_fock.cpp
/// @brief NDDO reference fock1/wstore/helect ports for NDDO methods.

#include "atomiverse/cpu/semiempirical/nddo_fock.h"

#include <algorithm>
#include <stdexcept>
#include <utility>

#include "atomiverse/core/frame_schema.h"
#include "atomiverse/cpu/semiempirical/am1_parameters.h"
#include "atomiverse/cpu/semiempirical/nddo_two_electron.h"

namespace ATV {
namespace cpu {
namespace semiempirical {

namespace {

std::size_t square_index(int row0, int col0, int dim) {
    return static_cast<std::size_t>(row0) * static_cast<std::size_t>(dim)
           + static_cast<std::size_t>(col0);
}

int local_pair_index0(int a1, int b1) {
    const int hi = std::max(a1, b1);
    const int lo = std::min(a1, b1);
    return (hi * (hi - 1)) / 2 + lo - 1;
}

void set_matrix(std::vector<double>& values, int dim, int row1, int col1,
                double value) {
    values[square_index(row1 - 1, col1 - 1, dim)] = value;
}

double matrix_value(const std::vector<double>& values, int dim, int row0,
                    int col0) {
    return values[square_index(row0, col0, dim)];
}

std::vector<std::pair<int, int>> local_pairs(int n_orbitals) {
    std::vector<std::pair<int, int>> pairs;
    pairs.reserve(static_cast<std::size_t>(n_orbitals)
                  * static_cast<std::size_t>(n_orbitals + 1) / 2u);
    for (int i = 0; i < n_orbitals; ++i) {
        for (int j = 0; j <= i; ++j) {
            pairs.emplace_back(i, j);
        }
    }
    return pairs;
}

void validate_packed_size(const char* name, int n_orbitals,
                          const std::vector<double>& packed) {
    const std::size_t expected = static_cast<std::size_t>(n_orbitals)
                                 * static_cast<std::size_t>(n_orbitals + 1)
                                 / 2u;
    if (packed.size() != expected) {
        throw std::runtime_error(std::string(name)
                                 + " size does not match NDDO orbital count");
    }
}

} // namespace

double NDDOOneCenterIntegrals::at(int row, int col) const {
    if (row < 0 || col < 0 || row >= dimension || col >= dimension) {
        throw std::runtime_error("NDDO one-center integral index out of range");
    }
    return values[square_index(row, col, dimension)];
}

NDDOOneCenterIntegrals
NDDOOneCenterIntegralBuilder::build(const NDDOAtom& atom) const {
    const auto* p = atom.params;
    if (p == nullptr) {
        throw std::runtime_error(
            "AM1 parameters unavailable for one-center integrals");
    }
    if (atom.n_orbitals > 4) {
        throw std::runtime_error(
            "AM1 d-orbital one-center integrals are not ported yet");
    }

    NDDOOneCenterIntegrals out;
    out.atomic_number = atom.atomic_number;
    out.dimension = atom.n_orbitals * (atom.n_orbitals + 1) / 2;
    out.values.assign(static_cast<std::size_t>(out.dimension)
                          * static_cast<std::size_t>(out.dimension),
                      0.0);

    set_matrix(out.values, out.dimension, 1, 1, p->gss);
    if (atom.n_orbitals > 2) {
        const int ip = 1;
        const int ipx = ip + 2;
        const int ipy = ip + 5;
        const int ipz = ip + 9;
        set_matrix(out.values, out.dimension, ipx, ip, p->gsp);
        set_matrix(out.values, out.dimension, ipy, ip, p->gsp);
        set_matrix(out.values, out.dimension, ipz, ip, p->gsp);
        set_matrix(out.values, out.dimension, ip, ipx, p->gsp);
        set_matrix(out.values, out.dimension, ip, ipy, p->gsp);
        set_matrix(out.values, out.dimension, ip, ipz, p->gsp);
        set_matrix(out.values, out.dimension, ipx, ipx, p->gpp);
        set_matrix(out.values, out.dimension, ipy, ipy, p->gpp);
        set_matrix(out.values, out.dimension, ipz, ipz, p->gpp);
        set_matrix(out.values, out.dimension, ipy, ipx, p->gp2);
        set_matrix(out.values, out.dimension, ipz, ipx, p->gp2);
        set_matrix(out.values, out.dimension, ipz, ipy, p->gp2);
        set_matrix(out.values, out.dimension, ipx, ipy, p->gp2);
        set_matrix(out.values, out.dimension, ipx, ipz, p->gp2);
        set_matrix(out.values, out.dimension, ipy, ipz, p->gp2);
        set_matrix(out.values, out.dimension, ip + 1, ip + 1, p->hsp);
        set_matrix(out.values, out.dimension, ip + 3, ip + 3, p->hsp);
        set_matrix(out.values, out.dimension, ip + 6, ip + 6, p->hsp);
        const double hpp = 0.5 * (p->gpp - p->gp2);
        set_matrix(out.values, out.dimension, ip + 4, ip + 4, hpp);
        set_matrix(out.values, out.dimension, ip + 7, ip + 7, hpp);
        set_matrix(out.values, out.dimension, ip + 8, ip + 8, hpp);
    }
    return out;
}

std::vector<double>
NDDOFockBuilder::add_one_center(const NDDOSystem& system,
                                const std::vector<double>& packed_hcore,
                                const std::vector<double>& total_density,
                                const std::vector<double>& spin_density) const {
    validate_packed_size("packed_hcore", system.n_orbitals, packed_hcore);
    validate_packed_size("total_density", system.n_orbitals, total_density);
    validate_packed_size("spin_density", system.n_orbitals, spin_density);

    std::vector<double> fock = packed_hcore;
    NDDOOneCenterIntegralBuilder integral_builder;

    for (const auto& atom : system.atoms) {
        const auto w = integral_builder.build(atom);
        const int ia = atom.first_ao;
        const int ib = atom.last_ao;
        for (int i = ia; i <= ib; ++i) {
            const int iw = i - ia + 1;
            for (int j = ia; j <= i; ++j) {
                const int jw = j - ia + 1;
                const std::size_t ij = packed_index(i, j);
                const int ijw = local_pair_index0(iw, jw);
                double sum = 0.0;

                for (int k = ia; k <= ib; ++k) {
                    const int kw = k - ia + 1;
                    for (int l = ia; l <= ib; ++l) {
                        const int lw = l - ia + 1;
                        const std::size_t ijp = packed_index(k, l);
                        const int klw = local_pair_index0(kw, lw);
                        const int ikw = local_pair_index0(kw, jw);
                        const int jlw = local_pair_index0(lw, iw);
                        sum +=
                            total_density[ijp]
                                * matrix_value(w.values, w.dimension, ijw, klw)
                            - spin_density[ijp]
                                  * matrix_value(w.values, w.dimension, ikw,
                                                 jlw);
                    }
                }
                fock[ij] += sum;
            }
        }
    }
    return fock;
}

std::vector<double>
NDDOFockBuilder::add_two_center(const NDDOSystem& system,
                                const cpu::Frame& frame,
                                const std::vector<double>& one_center_fock,
                                const std::vector<double>& total_density,
                                const std::vector<double>& spin_density) const {
    return add_two_center(system, frame.span(kAtom_x), frame.span(kAtom_y),
                          frame.span(kAtom_z), one_center_fock, total_density,
                          spin_density);
}

std::vector<double> NDDOFockBuilder::add_two_center(
    const NDDOSystem& system, const StorageReal* sp_x, const StorageReal* sp_y,
    const StorageReal* sp_z, const std::vector<double>& one_center_fock,
    const std::vector<double>& total_density,
    const std::vector<double>& spin_density) const {
    validate_packed_size("one_center_fock", system.n_orbitals, one_center_fock);
    validate_packed_size("total_density", system.n_orbitals, total_density);
    validate_packed_size("spin_density", system.n_orbitals, spin_density);
    if (sp_x == nullptr || sp_y == nullptr || sp_z == nullptr) {
        throw std::runtime_error(
            "NDDOFockBuilder::add_two_center requires non-null coordinates");
    }

    std::vector<double> fock = one_center_fock;
    // Coordinates are hoisted by the caller (run / force) so the per-pair
    // two-center block path never re-resolves Frame keys (13.7× rule).
    for (std::size_t i = 0; i < system.atoms.size(); ++i) {
        const auto& atom_i = system.atoms[i];
        const auto pairs_i = local_pairs(atom_i.n_orbitals);
        for (std::size_t j = 0; j < i; ++j) {
            const auto& atom_j = system.atoms[j];
            const auto pairs_j = local_pairs(atom_j.n_orbitals);
            const auto w = build_two_center_integral_block(
                system, sp_x, sp_y, sp_z, static_cast<int>(i),
                static_cast<int>(j));

            for (std::size_t pi = 0; pi < pairs_i.size(); ++pi) {
                const auto [li_a, li_b] = pairs_i[pi];
                const int ao_a = atom_i.first_ao + li_a;
                const int ao_b = atom_i.first_ao + li_b;
                const auto ab = packed_index(ao_a, ao_b);
                const double mult_i = li_a == li_b ? 1.0 : 2.0;

                for (std::size_t pj = 0; pj < pairs_j.size(); ++pj) {
                    const auto [lj_c, lj_d] = pairs_j[pj];
                    const int ao_c = atom_j.first_ao + lj_c;
                    const int ao_d = atom_j.first_ao + lj_d;
                    const auto cd = packed_index(ao_c, ao_d);
                    const double mult_j = lj_c == lj_d ? 1.0 : 2.0;
                    const double wij =
                        w.at(static_cast<int>(pi), static_cast<int>(pj));

                    fock[ab] += mult_j * total_density[cd] * wij;
                    fock[cd] += mult_i * total_density[ab] * wij;

                    const double exchange = 0.25 * mult_i * mult_j * wij;
                    fock[packed_index(ao_a, ao_c)] -=
                        exchange * spin_density[packed_index(ao_b, ao_d)];
                    fock[packed_index(ao_a, ao_d)] -=
                        exchange * spin_density[packed_index(ao_b, ao_c)];
                    fock[packed_index(ao_b, ao_c)] -=
                        exchange * spin_density[packed_index(ao_a, ao_d)];
                    fock[packed_index(ao_b, ao_d)] -=
                        exchange * spin_density[packed_index(ao_a, ao_c)];
                }
            }
        }
    }
    return fock;
}

std::vector<double>
NDDOFockBuilder::build(const NDDOSystem& system, const cpu::Frame& frame,
                       const std::vector<double>& packed_hcore,
                       const std::vector<double>& total_density,
                       const std::vector<double>& spin_density) const {
    return build(system, frame.span(kAtom_x), frame.span(kAtom_y),
                 frame.span(kAtom_z), packed_hcore, total_density,
                 spin_density);
}

std::vector<double>
NDDOFockBuilder::build(const NDDOSystem& system, const StorageReal* sp_x,
                       const StorageReal* sp_y, const StorageReal* sp_z,
                       const std::vector<double>& packed_hcore,
                       const std::vector<double>& total_density,
                       const std::vector<double>& spin_density) const {
    return add_two_center(
        system, sp_x, sp_y, sp_z,
        add_one_center(system, packed_hcore, total_density, spin_density),
        total_density, spin_density);
}

double electronic_energy_ev(int n_orbitals, const std::vector<double>& density,
                            const std::vector<double>& hcore,
                            const std::vector<double>& fock) {
    validate_packed_size("density", n_orbitals, density);
    validate_packed_size("hcore", n_orbitals, hcore);
    validate_packed_size("fock", n_orbitals, fock);

    double diagonal = 0.0;
    double off_diagonal = 0.0;
    int k = 0;
    const int nn = n_orbitals + 1;
    for (int i = 2; i <= nn; ++i) {
        diagonal += density[static_cast<std::size_t>(k)]
                    * (hcore[static_cast<std::size_t>(k)]
                       + fock[static_cast<std::size_t>(k)]);
        const int jj = i - 1;
        ++k;
        if (i == nn) {
            continue;
        }
        for (int off = 0; off < jj; ++off) {
            off_diagonal += density[static_cast<std::size_t>(k)]
                            * (hcore[static_cast<std::size_t>(k)]
                               + fock[static_cast<std::size_t>(k)]);
            ++k;
        }
    }
    return off_diagonal + 0.5 * diagonal;
}

} // namespace semiempirical
} // namespace cpu
} // namespace ATV
