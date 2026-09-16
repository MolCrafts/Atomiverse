/// @file src/cpu/semiempirical/am1_bcc_charge_assigner.cpp
/// @brief AM1-BCC charge assignment bridge implementation.

#include "atomiverse/cpu/semiempirical/am1_bcc_charge_assigner.h"

#include <cstdint>
#include <stdexcept>
#include <string>

#include "atomiverse/core/frame_schema.h"
#include "atomiverse/core/mat3.h"
#include "atomiverse/molrs.hpp"

namespace ATV {
namespace cpu {
namespace semiempirical {

namespace {

::rust::Vec<double> require_f64_column(const molrs::FrameRef& frame,
                                       const char* block, const char* col) {
    auto values = molrs::frame_column_f64(frame, block, col);
    if (values.empty()) {
        throw std::runtime_error(std::string("AM1BCCChargeAssigner: required ")
                                 + block + "." + col + " column is missing");
    }
    return values;
}

::rust::Vec<std::uint64_t> require_u32_column(const molrs::FrameRef& frame,
                                              const char* block,
                                              const char* col) {
    auto values = molrs::frame_column_u32(frame, block, col);
    if (values.empty()) {
        throw std::runtime_error(std::string("AM1BCCChargeAssigner: required ")
                                 + block + "." + col
                                 + " uint column is missing");
    }
    return values;
}

cpu::Frame build_am1_frame(const molrs::FrameRef& frame) {
    auto x = require_f64_column(frame, "atoms", "x");
    auto y = require_f64_column(frame, "atoms", "y");
    auto z = require_f64_column(frame, "atoms", "z");
    // molrs 0.13: Z is `atomic_number` (UInt). `type` is a String
    // force-field label and no longer carries the element.
    auto atomic_numbers = require_u32_column(frame, "atoms", "atomic_number");

    const auto n = static_cast<int>(x.size());
    if (static_cast<int>(y.size()) != n || static_cast<int>(z.size()) != n) {
        throw std::runtime_error(
            "AM1BCCChargeAssigner: atoms.x/y/z row counts differ");
    }
    if (static_cast<int>(atomic_numbers.size()) != n) {
        throw std::runtime_error("AM1BCCChargeAssigner: atomic-number row "
                                 "count differs from atoms.x");
    }
    const int first_element = static_cast<int>(molrs::Element::H);
    const int last_element = static_cast<int>(molrs::Element::Og);
    for (int i = 0; i < n; ++i) {
        const int atomic_number = atomic_numbers[static_cast<std::size_t>(i)];
        if (atomic_number < first_element || atomic_number > last_element) {
            throw std::runtime_error(
                "AM1BCCChargeAssigner: type column row " + std::to_string(i)
                + " is not a molrs::Element atomic number");
        }
    }

    cpu::Frame work(n);
    int* __restrict sp_type = work.span(kAtom_type);
    StorageReal* __restrict sp_x = work.span(kAtom_x);
    StorageReal* __restrict sp_y = work.span(kAtom_y);
    StorageReal* __restrict sp_z = work.span(kAtom_z);
    for (int i = 0; i < n; ++i) {
        sp_type[i] = atomic_numbers[static_cast<std::size_t>(i)];
        sp_x[i] = x[static_cast<std::size_t>(i)];
        sp_y[i] = y[static_cast<std::size_t>(i)];
        sp_z[i] = z[static_cast<std::size_t>(i)];
    }

    auto h = molrs::frame_box(frame);
    if (h.size() == 9u) {
        Mat3 cell;
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) {
                cell(i, j) = h[static_cast<std::size_t>(3 * i + j)];
            }
        }
        work.box.set_cell(cell);
    }
    return work;
}

std::vector<double> to_std_vector(const ::rust::Vec<double>& values) {
    std::vector<double> out;
    out.reserve(values.size());
    for (std::size_t i = 0; i < values.size(); ++i)
        out.push_back(values[i]);
    return out;
}

} // namespace

AM1BCCChargeAssigner::AM1BCCChargeAssigner(AM1 am1, std::string parameter_set)
    : solver_(std::move(am1)), parameter_set_(std::move(parameter_set)) {}

std::vector<double> AM1BCCChargeAssigner::assign(molrs::FrameRef& frame) {
    cpu::Frame work = build_am1_frame(frame);
    solver_.build(work);
    const auto& am1 = solver_.run(work);
    if (am1.mulliken_charges.size()
        != static_cast<std::size_t>(work.nlocal())) {
        throw std::runtime_error("AM1BCCChargeAssigner: AM1 result charge "
                                 "count does not match atom count");
    }

    auto bcc = molrs::am1_bcc_assign_frame_from_base(
        frame,
        ::rust::Slice<const double>(am1.mulliken_charges.data(),
                                    am1.mulliken_charges.size()),
        {parameter_set_.data(), parameter_set_.size()});
    last_charges_ = to_std_vector(bcc);
    return last_charges_;
}

} // namespace semiempirical
} // namespace cpu
} // namespace ATV
