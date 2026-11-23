#pragma once

// A container (by default the 1x1 square) holding non-overlapping polygons.

#include "polypack/geometry.hpp"
#include "polypack/polygon.hpp"

#include <string>
#include <vector>

namespace polypack {

class Bin {
 public:
  explicit Bin(Box bounds = unitSquare());

  const Box& bounds() const noexcept { return bounds_; }
  const std::vector<Polygon>& items() const noexcept { return items_; }

  // True if `polygon` lies inside the bin and overlaps no placed item.
  bool canPlace(const Polygon& polygon) const;

  // Places `polygon` if canPlace(); returns whether it was placed.
  bool tryPlace(const Polygon& polygon);

  const Rational& usedArea() const noexcept { return usedArea_; }
  Rational fillRatio() const;

  // SVG document for viewing (floating point is used only for drawing).
  std::string toSvg(int sizePx = 512) const;

 private:
  Box bounds_;
  std::vector<Polygon> items_;
  Rational usedArea_;
};

}  // namespace polypack
