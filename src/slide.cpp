#include "polypack/slide.hpp"

#include "polypack/overlap.hpp"

#include <algorithm>

namespace polypack {
namespace {

// "along" is the coordinate that decreases while sliding, "across" the other.
const Rational& along(const Point& p, Direction d) { return d == Direction::Down ? p.y : p.x; }
const Rational& across(const Point& p, Direction d) { return d == Direction::Down ? p.x : p.y; }
const Rational& alongMin(const Box& b, Direction d) { return along(b.min, d); }
const Rational& alongMax(const Box& b, Direction d) { return along(b.max, d); }
const Rational& acrossMin(const Box& b, Direction d) { return across(b.min, d); }
const Rational& acrossMax(const Box& b, Direction d) { return across(b.max, d); }

Point shift(Direction d, const Rational& s) {
  return d == Direction::Down ? Point(Rational(0), Rational(-s)) : Point(Rational(-s), Rational(0));
}

// If the line "across == across(v)" meets segment [a, b] at a single point,
// stores that point's along-coordinate in `out`.
bool alongAtAcross(const Point& v, const Point& a, const Point& b, Direction d, Rational& out) {
  const Rational& va = across(v, d);
  const Rational& aa = across(a, d);
  const Rational& ba = across(b, d);
  if (aa == ba) return false;  // parallel to the motion: its endpoints are covered by neighbours
  if (va < (aa < ba ? aa : ba) || va > (aa < ba ? ba : aa)) return false;
  out = along(a, d) + (along(b, d) - along(a, d)) * (va - aa) / (ba - aa);
  return true;
}

}  // namespace

Rational maxSlide(const Polygon& moving, Direction d, const std::vector<Polygon>& obstacles,
                  const Box& container) {
  const Box& mb = moving.bounds();
  const Rational wall = alongMin(mb, d) - alongMin(container, d);
  if (wall <= 0) return Rational(0);

  // Only obstacles in the swept band, and not entirely behind the motion, matter.
  std::vector<const Polygon*> relevant;
  for (const Polygon& o : obstacles) {
    const Box& ob = o.bounds();
    if (acrossMax(ob, d) < acrossMin(mb, d) || acrossMin(ob, d) > acrossMax(mb, d)) continue;
    if (alongMin(ob, d) >= alongMax(mb, d)) continue;
    relevant.push_back(&o);
  }
  if (relevant.empty()) return wall;

  // Every distance at which a vertex of one polygon reaches an edge of the
  // other. Between two consecutive events the overlap status cannot change.
  std::vector<Rational> events{Rational(0), wall};
  Rational hit;
  const std::vector<Point>& mv = moving.vertices();
  for (const Polygon* o : relevant) {
    const std::vector<Point>& ov = o->vertices();
    for (std::size_t j = 0; j < ov.size(); ++j) {
      const Point& a = ov[j];
      const Point& b = ov[(j + 1) % ov.size()];
      for (const Point& v : mv) {  // moving vertex falls onto an obstacle edge
        if (!alongAtAcross(v, a, b, d, hit)) continue;
        Rational s = along(v, d) - hit;
        if (s >= 0 && s <= wall) events.push_back(std::move(s));
      }
    }
    for (std::size_t j = 0; j < mv.size(); ++j) {
      const Point& a = mv[j];
      const Point& b = mv[(j + 1) % mv.size()];
      for (const Point& w : ov) {  // a moving edge falls onto an obstacle vertex
        if (!alongAtAcross(w, a, b, d, hit)) continue;
        Rational s = hit - along(w, d);
        if (s >= 0 && s <= wall) events.push_back(std::move(s));
      }
    }
  }
  std::sort(events.begin(), events.end());
  events.erase(std::unique(events.begin(), events.end()), events.end());

  // The start is valid; walk the intervals until the first one that overlaps.
  for (std::size_t k = 0; k + 1 < events.size(); ++k) {
    const Rational mid = (events[k] + events[k + 1]) / 2;
    const Polygon probe = moving.translated(shift(d, mid));
    for (const Polygon* o : relevant) {
      if (interiorsOverlap(probe, *o)) return events[k];
    }
  }
  return wall;
}

Polygon settleBottomLeft(const Polygon& start, const std::vector<Polygon>& obstacles,
                         const Box& container, int maxRounds) {
  Polygon current = start;
  for (int round = 0; round < maxRounds; ++round) {
    bool moved = false;
    for (const Direction d : {Direction::Down, Direction::Left}) {
      const Rational s = maxSlide(current, d, obstacles, container);
      if (s > 0) {
        current = current.translated(shift(d, s));
        moved = true;
      }
    }
    if (!moved) break;
  }
  return current;
}

}  // namespace polypack
