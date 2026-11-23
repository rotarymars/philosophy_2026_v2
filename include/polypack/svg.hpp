#pragma once

// SVG rendering for inspection. Coordinates are converted to floating point
// here only, and the y axis is flipped so the picture matches the maths.

#include "polypack/geometry.hpp"
#include "polypack/polygon.hpp"

#include <string>
#include <vector>

namespace polypack {

std::string renderSvg(const std::vector<Polygon>& polygons, const Box& frame, int sizePx = 512);

}  // namespace polypack
