/// @file core/molrs_contract.h
/// @brief Exact Atomiverse/molrs CXX API and capability contract.
#pragma once

#include <cstdint>

namespace ATV::molrs_contract {

/// Atomiverse's compiled expected CXX API version.
std::uint32_t expected_api_version() noexcept;

/// Capability bits Atomiverse requires from the loaded molrs bridge.
std::uint64_t required_capabilities() noexcept;

/// Validate explicit values. Public so contract-failure tests do not need a
/// second incompatible molrs library in the same process.
/// @throws std::runtime_error with expected/found values on incompatibility.
void validate(std::uint32_t actual_version, std::uint64_t actual_capabilities);

/// Query the linked molrs bridge and enforce the exact contract.
/// Called by every CPU Backend before it initializes execution services.
void require_compatible();

} // namespace ATV::molrs_contract
