#include "polypack/generator.hpp"

#include <algorithm>
#include <cmath>
#include <numbers>
#include <stdexcept>
#include <vector>

namespace polypack {

void validate(const GeneratorConfig& c) {
  if (c.minVertices < 3 || c.maxVertices < c.minVertices) {
    throw std::invalid_argument("GeneratorConfig: need 3 <= minVertices <= maxVertices");
  }
  if (!(c.minRadius > 0) || c.maxRadius < c.minRadius || c.maxRadius > 0.5) {
    throw std::invalid_argument("GeneratorConfig: need 0 < minRadius <= maxRadius <= 0.5");
  }
  if (!(c.innerRatio > 0) || c.innerRatio > 1) {
    throw std::invalid_argument("GeneratorConfig: need 0 < innerRatio <= 1");
  }
  if (c.gridResolution < 16) throw std::invalid_argument("GeneratorConfig: gridResolution too small");
}

Polygon randomPolygon(std::mt19937_64& rng, const GeneratorConfig& c) {
  validate(c);
  std::uniform_int_distribution<int> vertexCount(c.minVertices, c.maxVertices);
  std::uniform_real_distribution<double> unit(0.0, 1.0);

  while (true) {
    const int n = vertexCount(rng);
    const double radius = c.minRadius + (c.maxRadius - c.minRadius) * unit(rng);

    std::vector<double> angles(static_cast<std::size_t>(n));
    for (double& a : angles) a = 2.0 * std::numbers::pi * unit(rng);
    std::sort(angles.begin(), angles.end());

    std::vector<Point> vertices;
    vertices.reserve(angles.size());
    for (const double a : angles) {
      const double r = radius * (c.innerRatio + (1.0 - c.innerRatio) * unit(rng));
      // Snap to the exact grid; the offset keeps coordinates positive.
      const long gx = std::lround((0.5 + r * std::cos(a)) * c.gridResolution);
      const long gy = std::lround((0.5 + r * std::sin(a)) * c.gridResolution);
      vertices.emplace_back(Rational(gx, c.gridResolution), Rational(gy, c.gridResolution));
    }
    for (Point& p : vertices) {
      p.x.canonicalize();
      p.y.canonicalize();
    }

    try {
      Polygon polygon(std::move(vertices));
      // Reject slivers (a sampling-quality heuristic, not a geometric test).
      if (toDouble(polygon.area()) * 40.0 < radius * radius) continue;
      return polygon.translated(-polygon.bounds().min);
    } catch (const std::invalid_argument&) {
      // Snapping made it degenerate or self-touching: draw again.
    }
  }
}

}  // namespace polypack
