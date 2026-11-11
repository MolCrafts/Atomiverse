/// @file tests/cpu/md/test_frame_tag_image.cpp
/// @brief `cpu::Frame` identity columns: `tag` (`TagInt`) and `image`
///        (`ImageInt` = int32 per component, shape [n, 3]), with the
///        default-construction invariant `tag[i] = i` and `image = 0`.

#include <gtest/gtest.h>

#include <cstdint>
#include <type_traits>

#include "atomiverse/cpu/frame/frame.h"

using namespace ATV;

// ── identity tag defaults ────────────────────────────────────────────

TEST(FrameTagImage, DefaultConstructSetsIdentityTags) {
    constexpr int n = 100;
    cpu::Frame frame(n);
    for (int i = 0; i < n; ++i) {
        EXPECT_EQ(frame.span(ATV::kAtom_tag)[i], static_cast<std::int64_t>(i))
            << "tag at index " << i << " should equal i by default";
    }
}

// ── image defaults to zero ───────────────────────────────────────────

TEST(FrameTagImage, DefaultConstructZerosImage) {
    constexpr int n = 100;
    cpu::Frame frame(n);
    for (int i = 0; i < n; ++i) {
        for (int d = 0; d < 3; ++d) {
            EXPECT_EQ(frame.span(ATV::kAtom_image)(i, d),
                      static_cast<ATV::ImageInt>(0))
                << "image(" << i << "," << d << ") should default to 0";
        }
    }
}

// ── tag value_type is int64_t ────────────────────────────────────────

TEST(FrameTagImage, TagDtypeIsTagInt) {
    cpu::Frame frame(4);
    using tag_t = typename std::decay_t<decltype(frame.span(
        ATV::kAtom_tag))>::element_type;
    // tag is the TagInt tier (int64 under bigbig, int32 under smallbig), not a
    // hardcoded int64 — see core/precision.h.
    static_assert(std::is_same_v<tag_t, ATV::TagInt>,
                  "Frame::Atoms::tag must be xtensor<TagInt, 1>");
    SUCCEED();
}

// ── image dtype is the int32 ImageInt tier ───────────────────────────

TEST(FrameTagImage, ImageDtypeIsInt32ImageInt) {
    cpu::Frame frame(4);
    using image_t = typename std::decay_t<decltype(frame.span(
        ATV::kAtom_image))>::element_type;
    // One signed 32-bit counter per component in every ID build: an int8
    // counter overflowed after ±127 box crossings (core/precision.h).
    static_assert(std::is_same_v<image_t, ATV::ImageInt>);
    static_assert(std::is_same_v<ATV::ImageInt, std::int32_t>);
    frame.span(ATV::kAtom_image)(0, 0) = 1'000'000;
    EXPECT_EQ(frame.span(ATV::kAtom_image)(0, 0), 1'000'000);
}

// ── image shape [n, 3] ───────────────────────────────────────────────

TEST(FrameTagImage, ImageShapeIsNx3) {
    constexpr int n = 37;
    cpu::Frame frame(n);
    // Per-atom SoA fields are sized to capacity `nmax` (LAMMPS DELTA growth,
    // floored at 64), not exactly `n`. The invariant is that `image` is a
    // `[nmax, 3]` array with the same row count as the other per-atom fields.
    EXPECT_GE(frame.capacity(), n);
    EXPECT_EQ(frame.block("atoms").rows(), frame.capacity());
    EXPECT_EQ(frame.span(ATV::kAtom_image).cols(), 3);
}
