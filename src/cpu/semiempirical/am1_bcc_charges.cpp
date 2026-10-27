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

std::vector<double> AM1BCCCharges::assign(const molrs::Atomistic& molecule) {
    molrs::MolecularGeometry geometry;
    try {
        geometry = molecule.geometry();
    } catch (const ::rust::Error& error) {
        throw std::invalid_argument(std::string("AM1BCCCharges: ")
                                    + error.what());
    }
    const std::size_t n = geometry.atomic_numbers.size();
    if (n > static_cast<std::size_t>(std::numeric_limits<int>::max()))
        throw std::invalid_argument(
            "AM1BCCCharges: molecule has too many atoms");
    Frame work(static_cast<int>(n));
    for (std::size_t i = 0; i < n; ++i) {
        work.span(kAtom_type)[i] = static_cast<int>(geometry.atomic_numbers[i]);
        work.span(kAtom_x)[i] =
            static_cast<StorageReal>(geometry.positions[3 * i]);
        work.span(kAtom_y)[i] =
            static_cast<StorageReal>(geometry.positions[3 * i + 1]);
        work.span(kAtom_z)[i] =
            static_cast<StorageReal>(geometry.positions[3 * i + 2]);
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
