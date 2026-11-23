#pragma once

// A simple (non-self-intersecting) polygon with exact rational vertices.
//
// Polygons are drawn the way an SVG path is drawn:
//
//     auto tri = Polygon::fromSvgPath("M 0 0 L 1/2 0 L 1/4 1/3 Z");
//     auto sq  = Polygon::fromSvgPath("m 1/4 1/4 h 1/2 v 1/2 h -1/2 z");
//
// Supported commands: M/m, L/l, H/h, V/v, Z/z (upper case = absolute,
// lower case = relative), with SVG's implicit "repeat as lineto" rule.
// Numbers may be integers, fractions ("1/3") or exact decimals ("0.25").
// Unlike SVG, the y axis points up.
//
// Invariants enforced by the constructor (std::invalid_argument otherwise):
//   * at least 3 vertices, no zero-length edges, no zero-width spikes;
//   * the boundary does not touch or cross itself;
//   * vertices are stored counter-clockwise, with redundant collinear
//     vertices removed.

#include "polypack/geometry.hpp"
#include "polypack/rotation.hpp"

#include <cstddef>
#include <string>
#include <string_view>
#include <vector>

namespace polypack {

enum class Location { Outside, Boundary, Inside };

class Polygon {
 public:
  explicit Polygon(std::vector<Point> vertices);

  static Polygon fromSvgPath(std::string_view path);

  const std::vector<Point>& vertices() const noexcept { return vertices_; }
  std::size_t size() const noexcept { return vertices_.size(); }
  const Rational& area() const noexcept { return area_; }
  const Box& bounds() const noexcept { return bounds_; }
  Point centroid() const;

  // Exact point classification.
  Location locate(const Point& p) const;

  // Rigid transforms (and uniform scaling). They return new polygons.
  Polygon translated(const Point& offset) const;
  Polygon rotated(const Rotation& rotation, const Point& pivot) const;
  Polygon rotated(const Rotation& rotation) const;  // about the centroid
  Polygon scaled(const Rational& factor, const Point& pivot) const;

  // Exact path string, e.g. "M 0 0 L 1/2 0 L 1/4 1/3 Z". Round-trips
  // through fromSvgPath.
  std::string toPathString() const;

  // Same vertex sequence (same starting vertex).
  friend bool operator==(const Polygon& a, const Polygon& b) { return a.vertices_ == b.vertices_; }

 private:
  struct Trusted {};
  // For transforms that preserve validity and orientation: skips validation.
  Polygon(Trusted, std::vector<Point> vertices);

  void removeRedundantVertices();
  void ensureSimple() const;
  void computeCache();

  std::vector<Point> vertices_;
  Rational area_;
  Box bounds_;
};

}  // namespace polypack
