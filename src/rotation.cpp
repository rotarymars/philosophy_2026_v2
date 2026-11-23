#include "polypack/rotation.hpp"

#include <stdexcept>
#include <utility>

namespace polypack {

Rotation::Rotation() : cos_(1), sin_(0) {}

Rotation::Rotation(Rational cos, Rational sin) : cos_(std::move(cos)), sin_(std::move(sin)) {}

Rotation Rotation::fromHalfAngleTangent(const Rational& t) {
  const Rational t2 = t * t;
  const Rational denominator = 1 + t2;
  return Rotation(Rational((1 - t2) / denominator), Rational((2 * t) / denominator));
}

Rotation Rotation::quarterTurns(int k) {
  switch (((k % 4) + 4) % 4) {
    case 0: return Rotation(1, 0);
    case 1: return Rotation(0, 1);
    case 2: return Rotation(-1, 0);
    default: return Rotation(0, -1);
  }
}

Rotation Rotation::fromCosSin(const Rational& cos, const Rational& sin) {
  if (cos * cos + sin * sin != 1) {
    throw std::invalid_argument("Rotation::fromCosSin: cos^2 + sin^2 must equal 1 exactly");
  }
  return Rotation(cos, sin);
}

Point Rotation::apply(const Point& v) const {
  return Point(cos_ * v.x - sin_ * v.y, sin_ * v.x + cos_ * v.y);
}

Point Rotation::apply(const Point& p, const Point& pivot) const {
  return pivot + apply(p - pivot);
}

Rotation Rotation::inverse() const { return Rotation(cos_, -sin_); }

Rotation operator*(const Rotation& a, const Rotation& b) {
  return Rotation(Rational(a.cos_ * b.cos_ - a.sin_ * b.sin_),
                  Rational(a.sin_ * b.cos_ + a.cos_ * b.sin_));
}

}  // namespace polypack
