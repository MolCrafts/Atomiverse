/// @file core/molrs_contract.cpp
/// @brief Exact Atomiverse/molrs CXX API contract enforcement.

#include "atomiverse/facade/molrs_contract.h"

#include "atomiverse/molrs.hpp"

#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <string>

#ifndef ATV_MOLRS_CXX_API_VERSION
#error "ATV_MOLRS_CXX_API_VERSION must come from cmake/MolrsContract.cmake"
#endif

#ifndef ATV_MOLRS_CXX_REQUIRED_CAPABILITIES
#error                                                                         \
    "ATV_MOLRS_CXX_REQUIRED_CAPABILITIES must come from cmake/MolrsContract.cmake"
#endif

namespace ATV::molrs_contract {
namespace {

std::string hex_mask(std::uint64_t value) {
    std::ostringstream out;
    out << "0x" << std::hex << std::uppercase << value;
    return out.str();
}

} // namespace

std::uint32_t expected_api_version() noexcept {
    return ATV_MOLRS_CXX_API_VERSION;
}

std::uint64_t required_capabilities() noexcept {
    return ATV_MOLRS_CXX_REQUIRED_CAPABILITIES;
}

void validate(std::uint32_t actual_version, std::uint64_t actual_capabilities) {
    if (actual_version != expected_api_version()) {
        throw std::runtime_error("molrs CXX API mismatch: Atomiverse requires v"
                                 + std::to_string(expected_api_version())
                                 + "; found v"
                                 + std::to_string(actual_version));
    }

    const std::uint64_t missing =
        required_capabilities() & ~actual_capabilities;
    if (missing != 0) {
        throw std::runtime_error(
            "molrs CXX capability mismatch: Atomiverse requires "
            + hex_mask(required_capabilities()) + "; found "
            + hex_mask(actual_capabilities) + "; missing " + hex_mask(missing));
    }
}

void require_compatible() {
    validate(molrs::cxx_api_version(), molrs::cxx_api_capabilities());
}

} // namespace ATV::molrs_contract
