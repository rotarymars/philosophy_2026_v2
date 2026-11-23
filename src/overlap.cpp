#include "polypack/overlap.hpp"

#include <algorithm>
#include <utility>
#include <vector>

namespace polypack {
namespace {

// Does some piece of p's boundary reveal that the interiors overlap?
//
// Each edge of p is cut at every point where it meets q's boundary. Every
// resulting piece then lies either
//   (a) on an edge of q: since both polygons are counter-clockwise, their
//       interiors are on the same side iff the two edges point the same way;
//   (b) strictly inside q;
//   (c) strictly outside q.
// If the interiors intersect, the boundary of the intersection region is made
// of pieces of type (a) with matching directions or of type (b) for one of
// the two polygons, so testing both orders is a complete check.
bool boundaryRevealsOverlap(const Polygon& p, const Polygon& q) {
  const std::vector<Point>& pv = p.vertices();
  const std::vector<Point>& qv = q.vertices();
  const std::size_t n = pv.size();
  const std::size_t m = qv.size();

  std::vector<std::pair<Rational, Point>> cuts;
  for (std::size_t i = 0; i < n; ++i) {
    const Point& a = pv[i];
    const Point& b = pv[(i + 1) % n];
    const Box edgeBox = segmentBounds(a, b);
    if (!boxesTouch(edgeBox, q.bounds())) continue;  // entirely outside q

    const Point dir = b - a;
    cuts.clear();
    cuts.emplace_back(Rational(0), a);
    cuts.emplace_back(dot(dir, dir), b);
    for (std::size_t j = 0; j < m; ++j) {
      const Point& c = qv[j];
      const Point& d = qv[(j + 1) % m];
      if (!boxesTouch(edgeBox, segmentBounds(c, d))) continue;
      const SegmentIntersection x = intersectSegments(a, b, c, d);
      if (x.kind == SegmentIntersection::Kind::None) continue;
      cuts.emplace_back(dot(x.first - a, dir), x.first);
      if (x.kind == SegmentIntersection::Kind::Overlap) {
        cuts.emplace_back(dot(x.second - a, dir), x.second);
      }
    }
    std::sort(cuts.begin(), cuts.end(),
              [](const auto& l, const auto& r) { return l.first < r.first; });
    cuts.erase(std::unique(cuts.begin(), cuts.end(),
                           [](const auto& l, const auto& r) { return l.first == r.first; }),
               cuts.end());

    for (std::size_t k = 0; k + 1 < cuts.size(); ++k) {
      const Point& u = cuts[k].second;
      const Point& v = cuts[k + 1].second;

      bool onBoundaryOfQ = false;
      for (std::size_t j = 0; j < m; ++j) {
        const Point& c = qv[j];
        const Point& d = qv[(j + 1) % m];
        if (onSegment(u, c, d) && onSegment(v, c, d)) {
          if (signOf(dot(dir, d - c)) > 0) return true;  // shared edge, same side
          onBoundaryOfQ = true;
          break;
        }
      }
      if (onBoundaryOfQ) continue;

      const Point mid = (u + v) * Rational(1, 2);
      if (q.locate(mid) == Location::Inside) return true;
    }
  }
  return false;
}

}  // namespace

bool interiorsOverlap(const Polygon& a, const Polygon& b) {
  if (!boxInteriorsIntersect(a.bounds(), b.bounds())) return false;

  // Fast path: two edges crossing transversally always means overlap.
  const std::vector<Point>& av = a.vertices();
  const std::vector<Point>& bv = b.vertices();
  for (std::size_t i = 0, n = av.size(); i < n; ++i) {
    const Point& p0 = av[i];
    const Point& p1 = av[(i + 1) % n];
    const Box edgeBox = segmentBounds(p0, p1);
    for (std::size_t j = 0, m = bv.size(); j < m; ++j) {
      const Point& q0 = bv[j];
      const Point& q1 = bv[(j + 1) % m];
      if (!boxesTouch(edgeBox, segmentBounds(q0, q1))) continue;
      if (segmentsProperlyCross(p0, p1, q0, q1)) return true;
    }
  }

  return boundaryRevealsOverlap(a, b) || boundaryRevealsOverlap(b, a);
}

bool isContainedIn(const Polygon& polygon, const Box& box) {
  // A box is convex, so containing every vertex means containing the polygon.
  return boxContains(box, polygon.bounds());
}

}  // namespace polypack
