#pragma once

// Random polygon items for the online packing problem.
//
// Items are star-shaped (often non-convex) polygons. Floating point is used
// only to *sample* the shape. Every vertex is then snapped to the exact grid
// 1/gridResolution, so the item itself is an exact rational polygon.

#include "polypack/polygon.hpp"

#include <random>

namespace polypack {

struct GeneratorConfig {
  int minVertices = 3;
  int maxVertices = 8;
  // The circumradius of an item is drawn uniformly from [minRadius, maxRadius].
  // maxRadius <= 0.5 guarantees that every rotation of an item fits in a 1x1 bin.
  double minRadius = 0.08;
  double maxRadius = 0.22;
  // Each vertex sits at a distance in [innerRatio * R, R] from the centre.
  // Lower values give spikier, more non-convex shapes.
  double innerRatio = 0.45;
  int gridResolution = 1024;
};

// Throws std::invalid_argument if the configuration is unusable.
void validate(const GeneratorConfig& config);

// A random simple polygon whose bounding box starts at (0, 0).
Polygon randomPolygon(std::mt19937_64& rng, const GeneratorConfig& config);

}  // namespace polypack
