/// @file tests/cpu/md/test_frame_tag_image.cpp
/// @brief RED tests for Phase 1 `cpu::Frame::Atoms` additions: `tag` and `image`.
///
/// Spec: .claude/specs/parallel-md-engine.md § A.2 (Frame additions).
/// These tests fail to compile until the production code adds `tag` (xtensor<int64_t,1>)
/// and `image` (xtensor<int8_t,2>, shape [n,3]) to `cpu::Frame::Atoms`, with the
/// default-construction invariant `tag[i]=i` and `image=0`.

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
                      static_cast<std::int8_t>(0))
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
