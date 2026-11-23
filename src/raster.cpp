#include "polypack/raster.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace polypack {

void rasterizeInto(const Polygon& polygon, const Box& frame, int size,
                   std::vector<std::uint8_t>& image, int supersample) {
  if (image.size() != static_cast<std::size_t>(size) * size) {
    throw std::invalid_argument("rasterizeInto: image has the wrong size");
  }
  const double fx = toDouble(frame.min.x);
  const double fy = toDouble(frame.min.y);
  const double fw = toDouble(frame.max.x) - fx;
  const double fh = toDouble(frame.max.y) - fy;

  // Vertices in pixel units: u to the right, v downwards.
  std::vector<double> us;
  std::vector<double> vs;
  for (const Point& p : polygon.vertices()) {
    us.push_back((toDouble(p.x) - fx) / fw * size);
    vs.push_back((fh - (toDouble(p.y) - fy)) / fh * size);
  }
  const auto [uMin, uMax] = std::minmax_element(us.begin(), us.end());
  const auto [vMin, vMax] = std::minmax_element(vs.begin(), vs.end());
  const int c0 = std::max(0, static_cast<int>(std::floor(*uMin)));
  const int c1 = std::min(size - 1, static_cast<int>(std::floor(*uMax)));
  const int r0 = std::max(0, static_cast<int>(std::floor(*vMin)));
  const int r1 = std::min(size - 1, static_cast<int>(std::floor(*vMax)));

  const std::size_t n = us.size();
  auto inside = [&](double u, double v) {
    bool in = false;
    for (std::size_t i = 0, j = n - 1; i < n; j = i++) {
      if ((vs[i] > v) != (vs[j] > v) &&
          u < (us[j] - us[i]) * (v - vs[i]) / (vs[j] - vs[i]) + us[i]) {
        in = !in;
      }
    }
    return in;
  };

  const int samples = supersample * supersample;
  for (int r = r0; r <= r1; ++r) {
    for (int c = c0; c <= c1; ++c) {
      int hits = 0;
      for (int sy = 0; sy < supersample; ++sy) {
        for (int sx = 0; sx < supersample; ++sx) {
          hits += inside(c + (sx + 0.5) / supersample, r + (sy + 0.5) / supersample);
        }
      }
      if (hits == 0) continue;
      std::uint8_t& pixel = image[static_cast<std::size_t>(r) * size + c];
      pixel = static_cast<std::uint8_t>(std::min(255, pixel + (255 * hits + samples / 2) / samples));
    }
  }
}

}  // namespace polypack
