#include "../numx/foundation/shape.h"
#include <gtest/gtest.h>

using namespace nx::foundation;

TEST(TestShape, TestBroadcastableSameRank) {
    Shape shape = {1, 2, 3, 1, 3};
    EXPECT_EQ(shape.broadcastable({2, 2, 1, 2, 1}), true);
}

TEST(TestShape, TestBroadcastableDiffRanks) {
    Shape shape = {1, 2, 3, 1, 3};
    EXPECT_EQ(shape.broadcastable({1, 2, 1}), true);
}

TEST(TestShape, TestBroadcastableToSameRank) {
    Shape shape = {1, 1, 1, 4, 3, 1};
    EXPECT_EQ(shape.broadcastable_to({2, 1, 2, 4, 3, 3}), true);
}

TEST(TestShape, TestBroadcastableToDiffRanks) {
    Shape shape = {1, 4, 1, 3};
    EXPECT_EQ(shape.broadcastable_to({2, 1, 2, 4, 3, 3}), true);
}

TEST(TestShape, TestNotBroadcastableToDiffRanks) {
    Shape shape = {1, 4, 1, 3, 1, 3, 3};
    EXPECT_EQ(shape.broadcastable_to({2, 1, 2, 4, 3, 3}), false);
}

TEST(TestShape, TestNotBroadcastable) {
    Shape shape = {1, 2, 3, 1, 3};
    EXPECT_EQ(shape.broadcastable({1, 2, 1, 1}), false);
}

TEST(TestShape, TestBroadcastSameRank) {
    Shape shape = {1, 2, 3, 1, 3};
    auto [broadcast_shape, broadcast_dims] = shape.broadcast({2, 2, 1, 2, 1});
    ShapeView broadcast_view = {2, 2, 3, 2, 3};
    ShapeStride broadcast_stride = {0, 9, 3, 0, 1};
    EXPECT_EQ(broadcast_shape.get_view(), broadcast_view);
    EXPECT_EQ(broadcast_shape.get_stride(), broadcast_stride);
}

TEST(TestShape, TestBroadcastDiffRanksV1) {
    Shape shape = {1, 2, 3, 1, 3};
    auto [broadcast_shape, broadcast_dims] = shape.broadcast({1, 2, 1});
    ShapeView broadcast_view = {1, 2, 3, 2, 3};
    ShapeStride broadcast_stride = {18, 9, 3, 0, 1};
    EXPECT_EQ(broadcast_shape.get_view(), broadcast_view);
    EXPECT_EQ(broadcast_shape.get_stride(), broadcast_stride);
}

TEST(TestShape, TestBroadcastDiffRanksV2) {
    Shape shape = {1, 4, 1, 3};
    auto [broadcast_shape, broadcast_dims] = shape.broadcast({2, 1, 2, 4, 3, 3});
    ShapeView broadcast_view = {2, 1, 2, 4, 3, 3};
    ShapeStride broadcast_stride = {0, 0, 0, 3, 0, 1};
    EXPECT_EQ(broadcast_shape.get_view(), broadcast_view);
    EXPECT_EQ(broadcast_shape.get_stride(), broadcast_stride);
}

TEST(TestShape, TestBroadcastScalarToDims) {
    Shape shape = {1};
    auto [broadcast_shape, broadcast_dims] = shape.broadcast({2, 3, 4});
    ShapeView broadcast_view = {2, 3, 4};
    ShapeStride broadcast_stride = {0, 0, 0};
    EXPECT_EQ(broadcast_shape.get_view(), broadcast_view);
    EXPECT_EQ(broadcast_shape.get_stride(), broadcast_stride);
}

TEST(TestShape, TestBroadcastToSameRank) {
    Shape shape = {1, 1, 1, 4, 3, 1};
    auto [broadcast_shape, broadcast_dims] = shape.broadcast_to({2, 1, 2, 4, 3, 3});
    ShapeView broadcast_view = {2, 1, 2, 4, 3, 3};
    ShapeStride broadcast_stride = {0, 12, 0, 3, 1, 0};
    EXPECT_EQ(broadcast_shape.get_view(), broadcast_view);
    EXPECT_EQ(broadcast_shape.get_stride(), broadcast_stride);
}

TEST(TestShape, TestBroadcastToDiffRanks) {
    Shape shape = {1, 4, 1, 3};
    auto [broadcast_shape, broadcast_dims] = shape.broadcast_to({2, 1, 2, 4, 3, 3});
    ShapeView broadcast_view = {2, 1, 2, 4, 3, 3};
    ShapeStride broadcast_stride = {0, 0, 0, 3, 0, 1};
    EXPECT_EQ(broadcast_shape.get_view(), broadcast_view);
    EXPECT_EQ(broadcast_shape.get_stride(), broadcast_stride);
}

TEST(TestShape, TestPermuteV1) {
    Shape shape = {2, 3, 4};
    Shape permute_shape = shape.permute({2, 0, 1});
    ShapeView permute_view = {4, 2, 3};
    ShapeStride permute_stride = {1, 12, 4};
    EXPECT_EQ(permute_shape.get_view(), permute_view);
    EXPECT_EQ(permute_shape.get_stride(), permute_stride);
}

TEST(TestShape, TestPermuteV2) {
    Shape shape = {2, 3, 4, 5};
    Shape permute_shape = shape.permute({3, 1, 2, 0});
    ShapeView permute_view = {5, 3, 4, 2};
    ShapeStride permute_stride = {1, 20, 5, 60};
    EXPECT_EQ(permute_shape.get_view(), permute_view);
    EXPECT_EQ(permute_shape.get_stride(), permute_stride);
}

TEST(TestShape, TestUnsqueezeInner) {
    Shape shape = {2, 3, 4, 5};
    ShapeDims dims = {1, 2};
    Shape unsqueeze_shape = shape.unsqueeze(dims);
    ShapeView unsqueeze_view = {2, 1, 3, 1, 4, 5};
    ShapeStride unsqueeze_stride = {60, 0, 20, 0, 5, 1};
    EXPECT_EQ(unsqueeze_shape.get_view(), unsqueeze_view);
    EXPECT_EQ(unsqueeze_shape.get_stride(), unsqueeze_stride);
}

TEST(TestShape, TestUnsqueezeOuter) {
    Shape shape = {2, 3, 4, 5};
    ShapeDims dims = {0, 4};
    Shape unsqueeze_shape = shape.unsqueeze(dims);
    ShapeView unsqueeze_view = {1, 2, 3, 4, 5, 1};
    ShapeStride unsqueeze_stride = {0, 60, 20, 5, 1, 0};
    EXPECT_EQ(unsqueeze_shape.get_view(), unsqueeze_view);
    EXPECT_EQ(unsqueeze_shape.get_stride(), unsqueeze_stride);
}

TEST(TestShape, TestSqueeze) {
    Shape shape = {1, 2, 1, 3, 4, 5, 1};
    ShapeDims dims = {0, 6};
    Shape squeeze_shape = shape.squeeze(dims);
    ShapeView squeeze_view = {2, 1, 3, 4, 5};
    ShapeStride squeeze_stride = {60, 60, 20, 5, 1};
    EXPECT_EQ(squeeze_shape.get_view(), squeeze_view);
    EXPECT_EQ(squeeze_shape.get_stride(), squeeze_stride);
}

TEST(TestShape, TestSqueezeAll) {
    Shape shape = {1, 2, 1, 3, 4, 5, 1};
    ShapeDims dims = {};
    Shape squeeze_shape = shape.squeeze(dims);
    ShapeView squeeze_view = {2, 3, 4, 5};
    ShapeStride squeeze_stride = {60, 20, 5, 1};
    EXPECT_EQ(squeeze_shape.get_view(), squeeze_view);
    EXPECT_EQ(squeeze_shape.get_stride(), squeeze_stride);
}
