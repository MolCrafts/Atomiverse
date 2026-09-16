/// @file tests/cpu/test_molrs_contract.cpp
/// @brief Exact CXX API/capability contract against the linked molrs bridge.

#include <cstdint>
#include <stdexcept>
#include <string>

#include <gtest/gtest.h>

#include "atomiverse/core/molrs_contract.h"
#include "atomiverse/molrs.hpp"

namespace {

TEST(MolrsContract, LinkedBridgeMatchesExactly) {
    EXPECT_EQ(molrs::cxx_api_version(),
              ATV::molrs_contract::expected_api_version());
    EXPECT_EQ(molrs::cxx_api_capabilities()
                  & ATV::molrs_contract::required_capabilities(),
              ATV::molrs_contract::required_capabilities());
    EXPECT_NO_THROW(ATV::molrs_contract::require_compatible());
}

TEST(MolrsContract, VersionMismatchNamesExpectedAndFound) {
    try {
        ATV::molrs_contract::validate(
            ATV::molrs_contract::expected_api_version() + 1,
            ATV::molrs_contract::required_capabilities());
        FAIL() << "version mismatch was accepted";
    } catch (const std::runtime_error& err) {
        const std::string message = err.what();
        EXPECT_NE(message.find("requires v1"), std::string::npos);
        EXPECT_NE(message.find("found v2"), std::string::npos);
    }
}

TEST(MolrsContract, MissingCapabilityNamesAllMasks) {
    try {
        ATV::molrs_contract::validate(
            ATV::molrs_contract::expected_api_version(), 0);
        FAIL() << "missing capabilities were accepted";
    } catch (const std::runtime_error& err) {
        const std::string message = err.what();
        EXPECT_NE(message.find("requires 0x7"), std::string::npos);
        EXPECT_NE(message.find("found 0x0"), std::string::npos);
        EXPECT_NE(message.find("missing 0x7"), std::string::npos);
    }
}

} // namespace
