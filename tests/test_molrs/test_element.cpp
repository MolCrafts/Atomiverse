/// @file tests/test_molrs/test_element.cpp
/// @brief Drift fence for molrs::Element — the type key ATV lookup tables use.

#include <cstdint>

#include <gtest/gtest.h>

#include "atomiverse/molpy.hpp"

namespace {

TEST(MolrsElement, HydrogenIsOne) {
    EXPECT_EQ(static_cast<int>(molrs::Element::H), 1);
}

TEST(MolrsElement, OxygenIsEight) {
    EXPECT_EQ(static_cast<int>(molrs::Element::O), 8);
}

TEST(MolrsElement, SiliconIsFourteen) {
    EXPECT_EQ(static_cast<int>(molrs::Element::Si), 14);
}

TEST(MolrsElement, OganessonIsLast) {
    EXPECT_EQ(static_cast<int>(molrs::Element::Og), 118);
}

TEST(MolrsElement, CastFromAtomicNumberRoundTrips) {
    const auto el = static_cast<molrs::Element>(8);
    EXPECT_EQ(el, molrs::Element::O);
}

} // namespace
