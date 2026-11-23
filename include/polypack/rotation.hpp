#pragma once

// Exact rational rotations.
//
// A rotation by an arbitrary angle generally produces irrational coordinates
// (e.g. 45 degrees needs sqrt(2)/2). Instead of approximating, polypack only
// uses rotations whose cosine and sine are both rational. These are exactly
// the rotations described by the tangent of the half angle, t = tan(theta/2):
//
//     cos(theta) = (1 - t^2) / (1 + t^2),   sin(theta) = 2t / (1 + t^2)
//
// Every rational t gives an exact rotation (t = 1/2 gives the 3-4-5 rotation),
// and such rotations are dense: any angle can be approached as closely as you
// like. The only angle not reachable through t is 180 degrees, which is
// available via quarterTurns(2).

#include "polypack/geometry.hpp"

namespace polypack {

class Rotation {
 public:
  // Identity rotation.
  Rotation();

  // Rotation by theta where t = tan(theta / 2). Positive t is counter-clockwise.
  static Rotation fromHalfAngleTangent(const Rational& t);

  // Counter-clockwise rotation by k * 90 degrees (k may be negative).
  static Rotation quarterTurns(int k);

  // Throws std::invalid_argument unless cos^2 + sin^2 == 1 exactly.
  static Rotation fromCosSin(const Rational& cos, const Rational& sin);

  const Rational& cos() const noexcept { return cos_; }
  const Rational& sin() const noexcept { return sin_; }

  // Rotates a vector about the origin.
  Point apply(const Point& v) const;
  // Rotates a point about `pivot`.
  Point apply(const Point& p, const Point& pivot) const;

  Rotation inverse() const;

  // (a * b) applies b first, then a.
  friend Rotation operator*(const Rotation& a, const Rotation& b);
  friend bool operator==(const Rotation& a, const Rotation& b) {
    return a.cos_ == b.cos_ && a.sin_ == b.sin_;
  }

 private:
  Rotation(Rational cos, Rational sin);

  Rational cos_;
  Rational sin_;
};

}  // namespace polypack
