/// @file cpu/semiempirical/am1_bcc_charges.cpp
/// @brief AM1 on an internal QM frame; BCC on the complete chemical graph.
#include "atomiverse/cpu/semiempirical/am1_bcc_charges.h"

#include <limits>
#include <stdexcept>
#include <utility>

#include "atomiverse/core/frame_schema.h"
#include "rust/cxx.h"

namespace ATV::cpu::semiempirical {

AM1BCCCharges::AM1BCCCharges(AM1 method, std::string_view correction)
    : am1_(std::move(method)), correction_(correction) {
    if (correction_ != "bcc" && correction_ != "abcg2")
        throw std::invalid_argument("unknown BCC correction '" + correction_
                                    + "'; expected bcc or abcg2");
}

std::vector<double> AM1BCCCharges::assign(molrs::FrameRef& molecule) {
    ::rust::Vec<double> x, y, z;
    ::rust::Vec<std::uint64_t> atomic_numbers, bond_i, bond_j;
    try {
        x = molrs::frame_column_f64(molecule, "atoms", "x");
        y = molrs::frame_column_f64(molecule, "atoms", "y");
        z = molrs::frame_column_f64(molecule, "atoms", "z");
        try {
            atomic_numbers =
                molrs::frame_column_u64(molecule, "atoms", "atomic_number");
        } catch (const ::rust::Error&) {
            molrs::frame_fill_atomic_numbers(molecule);
            atomic_numbers =
                molrs::frame_column_u64(molecule, "atoms", "atomic_number");
        }
        // BCC corrections are looked up per bond: refuse a graph without
        // connectivity by name rather than as a missing frame key.
        if (!molrs::frame_has_block(molecule, "bonds"))
            throw std::invalid_argument(
                "AM1BCCCharges: the molecule has no bonds; BCC corrections "
                "are looked up per bond, so connectivity is required");
        bond_i = molrs::frame_column_u64(molecule, "bonds", "atomi");
        bond_j = molrs::frame_column_u64(molecule, "bonds", "atomj");
    } catch (const ::rust::Error& error) {
        throw std::invalid_argument(std::string("AM1BCCCharges: ")
                                    + error.what());
    }
    const std::size_t n = x.size();
    if (n == 0 || y.size() != n || z.size() != n || atomic_numbers.size() != n
        || bond_i.empty() || bond_j.size() != bond_i.size())
        throw std::invalid_argument(
            "AM1BCCCharges: molecule is missing coordinates or connectivity");
    if (n > static_cast<std::size_t>(std::numeric_limits<int>::max()))
        throw std::invalid_argument(
            "AM1BCCCharges: molecule has too many atoms");
    Frame work(static_cast<int>(n));
    for (std::size_t i = 0; i < n; ++i) {
        work.span(kAtom_type)[i] = static_cast<int>(atomic_numbers[i]);
        work.span(kAtom_x)[i] = static_cast<StorageReal>(x[i]);
        work.span(kAtom_y)[i] = static_cast<StorageReal>(y[i]);
        work.span(kAtom_z)[i] = static_cast<StorageReal>(z[i]);
    }
    am1_.build(work);
    const auto& base =
        am1_.run(static_cast<const Frame&>(work)).mulliken_charges;
    auto result =
        molrs::bcc_model_new({correction_.data(), correction_.size()})
            ->assign(molecule,
                     ::rust::Slice<const double>(base.data(), base.size()));
    if (result.size() != n)
        throw std::runtime_error(
            "AM1BCCCharges: BCC charge count differs from atom count");
    return {result.begin(), result.end()};
}

} // namespace ATV::cpu::semiempirical
