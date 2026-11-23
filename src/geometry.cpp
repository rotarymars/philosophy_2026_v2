#include "polypack/geometry.hpp"

namespace polypack {

int orientation(const Point& a, const Point& b, const Point& c) {
  const Rational value = (b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x);
  return signOf(value);
}

bool onSegment(const Point& p, const Point& a, const Point& b) {
  if (orientation(a, b, p) != 0) return false;
  // Collinear: p is between a and b iff (a - p) and (b - p) do not point the same way.
  return signOf(dot(a - p, b - p)) <= 0;
}

Box unitSquare() { return Box{Point(0, 0), Point(1, 1)}; }

Box segmentBounds(const Point& a, const Point& b) {
  return Box{Point(a.x < b.x ? a.x : b.x, a.y < b.y ? a.y : b.y),
             Point(a.x < b.x ? b.x : a.x, a.y < b.y ? b.y : a.y)};
}

bool boxesTouch(const Box& a, const Box& b) {
  return a.min.x <= b.max.x && b.min.x <= a.max.x && a.min.y <= b.max.y && b.min.y <= a.max.y;
}

bool boxInteriorsIntersect(const Box& a, const Box& b) {
  return a.min.x < b.max.x && b.min.x < a.max.x && a.min.y < b.max.y && b.min.y < a.max.y;
}

bool boxContains(const Box& outer, const Box& inner) {
  return outer.min.x <= inner.min.x && inner.max.x <= outer.max.x &&
         outer.min.y <= inner.min.y && inner.max.y <= outer.max.y;
}

SegmentIntersection intersectSegments(const Point& a, const Point& b,
                                      const Point& c, const Point& d) {
  SegmentIntersection result;
  const int o1 = orientation(a, b, c);
  const int o2 = orientation(a, b, d);

  if (o1 == 0 && o2 == 0) {
    // All four points are collinear: intersect the parameter ranges along a->b.
    const Point dir = b - a;
    const Rational length2 = dot(dir, dir);
    const Rational tc = dot(c - a, dir);
    const Rational td = dot(d - a, dir);
    const Rational low = tc < td ? tc : td;
    const Rational high = tc < td ? td : tc;
    const Rational from = low > 0 ? low : Rational(0);
    const Rational to = high < length2 ? high : length2;
    if (from > to) return result;
    result.first = a + dir * Rational(from / length2);
    if (from == to) {
      result.kind = SegmentIntersection::Kind::Point;
    } else {
      result.kind = SegmentIntersection::Kind::Overlap;
      result.second = a + dir * Rational(to / length2);
    }
    return result;
  }

  const int o3 = orientation(c, d, a);
  const int o4 = orientation(c, d, b);
  if (o1 * o2 > 0 || o3 * o4 > 0) return result;

  // Not parallel here (a parallel, non-collinear pair has o1 == o2 != 0).
  const Point r = b - a;
  const Point s = d - c;
  const Rational t = cross(c - a, s) / cross(r, s);
  result.kind = SegmentIntersection::Kind::Point;
  result.first = a + r * t;
  return result;
}

bool segmentsProperlyCross(const Point& a, const Point& b, const Point& c, const Point& d) {
  if (orientation(a, b, c) * orientation(a, b, d) >= 0) return false;
  return orientation(c, d, a) * orientation(c, d, b) < 0;
}

}  // namespace polypack
