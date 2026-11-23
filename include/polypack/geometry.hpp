#pragma once

// Exact 2D primitives: points/vectors, axis-aligned boxes and segment
// predicates. All results are exact (no epsilon anywhere).

#include "polypack/rational.hpp"

#include <utility>

namespace polypack {

// A point or a vector in the plane (y axis points up, as in mathematics).
struct Point {
  Rational x;
  Rational y;

  Point() = default;
  Point(Rational x_, Rational y_) : x(std::move(x_)), y(std::move(y_)) {}

  friend bool operator==(const Point& a, const Point& b) { return a.x == b.x && a.y == b.y; }
  friend bool operator!=(const Point& a, const Point& b) { return !(a == b); }
};

inline Point operator+(const Point& a, const Point& b) { return {a.x + b.x, a.y + b.y}; }
inline Point operator-(const Point& a, const Point& b) { return {a.x - b.x, a.y - b.y}; }
inline Point operator-(const Point& a) { return {-a.x, -a.y}; }
inline Point operator*(const Point& a, const Rational& k) { return {a.x * k, a.y * k}; }
inline Point operator*(const Rational& k, const Point& a) { return a * k; }

inline Rational dot(const Point& a, const Point& b) { return a.x * b.x + a.y * b.y; }
inline Rational cross(const Point& a, const Point& b) { return a.x * b.y - a.y * b.x; }

// +1 if a->b->c turns left (counter-clockwise), -1 if right, 0 if collinear.
int orientation(const Point& a, const Point& b, const Point& c);

// True if p lies on the closed segment [a, b].
bool onSegment(const Point& p, const Point& a, const Point& b);

// Closed axis-aligned rectangle [min.x, max.x] x [min.y, max.y].
struct Box {
  Point min;
  Point max;

  friend bool operator==(const Box& a, const Box& b) { return a.min == b.min && a.max == b.max; }
};

// The 1x1 container: [0,1] x [0,1].
Box unitSquare();

Box segmentBounds(const Point& a, const Point& b);

// Closed boxes share at least one point (touching counts).
bool boxesTouch(const Box& a, const Box& b);

// Open interiors intersect (touching does NOT count).
bool boxInteriorsIntersect(const Box& a, const Box& b);

// `inner` lies inside the closed box `outer`.
bool boxContains(const Box& outer, const Box& inner);

// Intersection of two closed segments [a,b] and [c,d].
struct SegmentIntersection {
  enum class Kind { None, Point, Overlap };
  Kind kind = Kind::None;
  Point first;   // the point (Kind::Point) or one end of the shared piece (Kind::Overlap)
  Point second;  // other end of the shared piece (Kind::Overlap only)
};

SegmentIntersection intersectSegments(const Point& a, const Point& b,
                                      const Point& c, const Point& d);

// The segments cross at a single point interior to both (no touching, no
// collinear overlap).
bool segmentsProperlyCross(const Point& a, const Point& b,
                           const Point& c, const Point& d);

}  // namespace polypack
