/// @file tests/test_molpy/test_element.cpp
/// @brief Drift fence for molrs::Element — the type key ATV lookup tables use.

#include <cstdint>

#include <gtest/gtest.h>

#include "atomiverse/molpy.hpp"

namespace {

TEST(MolpyElement, HydrogenIsOne) {
    EXPECT_EQ(static_cast<int>(molrs::Element::H), 1);
}

TEST(MolpyElement, OxygenIsEight) {
    EXPECT_EQ(static_cast<int>(molrs::Element::O), 8);
}

TEST(MolpyElement, SiliconIsFourteen) {
    EXPECT_EQ(static_cast<int>(molrs::Element::Si), 14);
}

TEST(MolpyElement, OganessonIsLast) {
    EXPECT_EQ(static_cast<int>(molrs::Element::Og), 118);
}

TEST(MolpyElement, CastFromAtomicNumberRoundTrips) {
    const auto el = static_cast<molrs::Element>(8);
    EXPECT_EQ(el, molrs::Element::O);
}

} // namespace
