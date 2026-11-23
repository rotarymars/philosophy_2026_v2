#pragma once

// Exact sliding ("gravity") of a polygon among obstacles.
//
// Used to turn a coarse action of the agent ("drop this item into column g")
// into a tight placement: the item falls straight down until it rests on
// something, then slides left, and so on (the classic bottom-left rule).
// All distances are exact rationals, so items end up exactly touching,
// never overlapping and never separated by a rounding gap.

#include "polypack/geometry.hpp"
#include "polypack/polygon.hpp"

#include <vector>

namespace polypack {

enum class Direction { Down, Left };

// Largest s >= 0 such that `moving` translated by s in `direction` stays in
// `container` and its interior never meets an obstacle's interior on the way.
// Precondition: `moving` is a valid placement already.
Rational maxSlide(const Polygon& moving, Direction direction,
                  const std::vector<Polygon>& obstacles, const Box& container);

// Alternately slides down and left until neither moves (or maxRounds is hit).
// Precondition: `start` is a valid placement already.
Polygon settleBottomLeft(const Polygon& start, const std::vector<Polygon>& obstacles,
                         const Box& container, int maxRounds = 32);

}  // namespace polypack
