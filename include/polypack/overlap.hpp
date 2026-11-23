#pragma once

// Exact overlap tests.
//
// Two pieces "overlap" when their interiors share any area. Touching along an
// edge or at a point is allowed, since tight packings rely on it.

#include "polypack/geometry.hpp"
#include "polypack/polygon.hpp"

namespace polypack {

// True if the interiors of `a` and `b` intersect. Works for any simple
// polygons, convex or not.
bool interiorsOverlap(const Polygon& a, const Polygon& b);

// True if `polygon` lies entirely inside the closed box.
bool isContainedIn(const Polygon& polygon, const Box& box);

}  // namespace polypack
