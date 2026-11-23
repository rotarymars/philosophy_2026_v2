#include "polypack/bin.hpp"

#include "polypack/overlap.hpp"
#include "polypack/svg.hpp"

#include <utility>

namespace polypack {

Bin::Bin(Box bounds) : bounds_(std::move(bounds)), usedArea_(0) {}

bool Bin::canPlace(const Polygon& polygon) const {
  if (!isContainedIn(polygon, bounds_)) return false;
  for (const Polygon& item : items_) {
    if (interiorsOverlap(item, polygon)) return false;
  }
  return true;
}

bool Bin::tryPlace(const Polygon& polygon) {
  if (!canPlace(polygon)) return false;
  items_.push_back(polygon);
  usedArea_ += polygon.area();
  return true;
}

Rational Bin::fillRatio() const {
  const Rational binArea = (bounds_.max.x - bounds_.min.x) * (bounds_.max.y - bounds_.min.y);
  return Rational(usedArea_ / binArea);
}

std::string Bin::toSvg(int sizePx) const { return renderSvg(items_, bounds_, sizePx); }

}  // namespace polypack
