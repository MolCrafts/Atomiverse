/// @file src/cpu/semiempirical/am1_bcc_charges.cpp
/// @brief AM1-BCC / ABCG2 charges: Atomiverse AM1 base charges, molrs BCC.

#include "atomiverse/cpu/semiempirical/am1_bcc_charges.h"

#include <algorithm>
#include <cstdint>
#include <exception>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>

#include "atomiverse/core/frame_schema.h"
#include "atomiverse/cpu/semiempirical/internal/bcc_graph.h"
#include "atomiverse/molpy.hpp"

namespace ATV {
namespace cpu {
namespace semiempirical {

namespace internal {

std::vector<std::array<int, 2>> local_bonds(const Frame& frame) {
    std::vector<std::array<int, 2>> bonds;
    if (!frame.has_bonded())
        return bonds;

    const int n = frame.nlocal();
    const auto tag = frame.span(kAtom_tag);
    std::unordered_map<TagInt, int> row_of;
    row_of.reserve(static_cast<std::size_t>(n));
    for (int i = 0; i < n; ++i)
        row_of.emplace(tag[static_cast<std::size_t>(i)], i);

    const auto num_bond = frame.span(kAtom_num_bond);
    const auto bond_atom = frame.span(kAtom_bond_atom);
    for (int i = 0; i < n; ++i) {
        for (int k = 0; k < num_bond[static_cast<std::size_t>(i)]; ++k) {
            const TagInt partner = bond_atom(i, k);
            const auto it = row_of.find(partner);
            if (it == row_of.end()) {
                throw std::invalid_argument(
                    "AM1BCCCharges: atom row " + std::to_string(i)
                    + " is bonded to tag " + std::to_string(partner)
                    + ", which is not a local atom; the Frame must hold "
                      "whole molecules");
            }
            const int j = it->second;
            bonds.push_back({std::min(i, j), std::max(i, j)});
        }
    }
    std::sort(bonds.begin(), bonds.end());
    bonds.erase(std::unique(bonds.begin(), bonds.end()), bonds.end());
    return bonds;
}

std::string_view molrs_name(BccParameterSet set) {
    switch (set) {
    case BccParameterSet::Bcc:
        return "bcc";
    case BccParameterSet::Abcg2:
        return "abcg2";
    }
    throw std::invalid_argument("AM1BCCCharges: unknown BccParameterSet");
}

} // namespace internal

AM1BCCCharges::AM1BCCCharges(AM1 am1, BccParameterSet parameter_set)
    : am1_(std::move(am1)), parameter_set_(parameter_set) {}

void AM1BCCCharges::assign(Frame& frame) {
    const int n = frame.nlocal();
    const auto bonds = internal::local_bonds(frame);
    if (n > 1 && bonds.empty()) {
        throw std::invalid_argument(
            "AM1BCCCharges: the Frame has no bonds. BCC corrections are "
            "looked up per bond, so the molecule's connectivity is required: "
            "read it from a file that carries bonds (mol2, sdf, LAMMPS data) "
            "or add them with Frame::add_bond");
    }

    am1_.build(frame);
    const AM1Result& am1 = am1_.run(static_cast<const Frame&>(frame));

    // The molecular graph molrs types: Z, positions, bonds (local indices).
    const auto type = frame.span(kAtom_type);
    const auto x = frame.span(kAtom_x);
    const auto y = frame.span(kAtom_y);
    const auto z = frame.span(kAtom_z);
    std::vector<std::uint64_t> atomic_number(static_cast<std::size_t>(n));
    std::vector<double> px(static_cast<std::size_t>(n));
    std::vector<double> py(static_cast<std::size_t>(n));
    std::vector<double> pz(static_cast<std::size_t>(n));
    for (std::size_t i = 0; i < static_cast<std::size_t>(n); ++i) {
        atomic_number[i] = static_cast<std::uint64_t>(type[i]);
        px[i] = x[i];
        py[i] = y[i];
        pz[i] = z[i];
    }
    std::vector<std::uint64_t> atomi;
    std::vector<std::uint64_t> atomj;
    atomi.reserve(bonds.size());
    atomj.reserve(bonds.size());
    for (const auto& b : bonds) {
        atomi.push_back(static_cast<std::uint64_t>(b[0]));
        atomj.push_back(static_cast<std::uint64_t>(b[1]));
    }

    ::rust::Vec<double> bcc;
    try {
        auto graph = molrs::frame_new();
        auto f64 = [](const std::vector<double>& v) {
            return ::rust::Slice<const double>(v.data(), v.size());
        };
        auto u64 = [](const std::vector<std::uint64_t>& v) {
            return ::rust::Slice<const std::uint64_t>(v.data(), v.size());
        };
        molrs::frame_set_column_u64(*graph, "atoms", "atomic_number",
                                    u64(atomic_number));
        molrs::frame_set_column_f64(*graph, "atoms", "x", f64(px));
        molrs::frame_set_column_f64(*graph, "atoms", "y", f64(py));
        molrs::frame_set_column_f64(*graph, "atoms", "z", f64(pz));
        if (!bonds.empty()) {
            molrs::frame_set_column_u64(*graph, "bonds", "atomi", u64(atomi));
            molrs::frame_set_column_u64(*graph, "bonds", "atomj", u64(atomj));
        }
        const std::string_view name = internal::molrs_name(parameter_set_);
        bcc = molrs::bcc_model_new({name.data(), name.size()})
                  ->assign(*graph, ::rust::Slice<const double>(
                                       am1.mulliken_charges.data(),
                                       am1.mulliken_charges.size()));
    } catch (const std::exception& err) { // rust::Error: molrs chemistry
        throw std::runtime_error(std::string("AM1BCCCharges: molrs BccModel: ")
                                 + err.what());
    }
    if (bcc.size() != static_cast<std::size_t>(n)) {
        throw std::runtime_error("AM1BCCCharges: molrs returned "
                                 + std::to_string(bcc.size()) + " charges for "
                                 + std::to_string(n) + " atoms");
    }

    auto charge = frame.span(kAtom_charge);
    for (std::size_t i = 0; i < static_cast<std::size_t>(n); ++i)
        charge[i] = static_cast<StorageReal>(bcc[i]);
}

} // namespace semiempirical
} // namespace cpu
} // namespace ATV
