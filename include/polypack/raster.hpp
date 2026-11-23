#pragma once

// Rasterisation for the neural network's *perception* only.
//
// The network consumes images, so shapes are drawn into small occupancy grids
// using floating point. This never influences geometric decisions: whether a
// placement is legal is always decided by the exact predicates.

#include "polypack/geometry.hpp"
#include "polypack/polygon.hpp"

#include <cstdint>
#include <vector>

namespace polypack {

// Adds the coverage of `polygon` (as seen inside `frame`) to a size x size
// image (row 0 = top of the frame, column 0 = left). Each pixel stores
// coverage 0..255, estimated with supersample^2 samples, saturating at 255.
void rasterizeInto(const Polygon& polygon, const Box& frame, int size,
                   std::vector<std::uint8_t>& image, int supersample = 2);

}  // namespace polypack
