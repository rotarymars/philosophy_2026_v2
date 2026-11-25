// Minimal self-contained test runner (no external dependencies).

#include "polypack/polypack.hpp"

#include <algorithm>
#include <functional>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

using namespace polypack;

namespace {

int g_failures = 0;
int g_checks = 0;

#define CHECK(cond)                                                                    \
  do {                                                                                 \
    ++g_checks;                                                                        \
    if (!(cond)) {                                                                     \
      ++g_failures;                                                                    \
      std::cerr << "  FAIL " << __FILE__ << ':' << __LINE__ << ": " #cond << '\n';     \
    }                                                                                  \
  } while (0)

#define CHECK_THROWS(expr)                                                             \
  do {                                                                                 \
    ++g_checks;                                                                        \
    bool threw_ = false;                                                               \
    try {                                                                              \
      (void)(expr);                                                                    \
    } catch (const std::invalid_argument&) {                                           \
      threw_ = true;                                                                   \
    }                                                                                  \
    if (!threw_) {                                                                     \
      ++g_failures;                                                                    \
      std::cerr << "  FAIL " << __FILE__ << ':' << __LINE__                            \
                << ": expected throw: " #expr << '\n';                                 \
    }                                                                                  \
  } while (0)

Rational R(std::string_view s) { return parseRational(s); }
Point P(std::string_view x, std::string_view y) { return Point(R(x), R(y)); }
Polygon path(std::string_view s) { return Polygon::fromSvgPath(s); }

Polygon rect(std::string_view x0, std::string_view y0, std::string_view x1, std::string_view y1) {
  return Polygon({P(x0, y0), P(x1, y0), P(x1, y1), P(x0, y1)});
}

// Checks both argument orders, since the relation must be symmetric.
bool overlap(const Polygon& a, const Polygon& b) {
  const bool ab = interiorsOverlap(a, b);
  const bool ba = interiorsOverlap(b, a);
  CHECK(ab == ba);
  return ab;
}

// ---------------------------------------------------------------------------

void testRational() {
  CHECK(R("1/3") == Rational(1, 3));
  CHECK(R("-0.25") == Rational(-1, 4));
  CHECK(R("1.5/3") == Rational(1, 2));
  CHECK(R("1/-2") == Rational(-1, 2));
  CHECK(R("7") == 7);
  CHECK(R(".5") == Rational(1, 2));
  CHECK(R("0.1") * 3 == R("0.3"));  // exact, unlike double
  CHECK(toString(R("2/4")) == "1/2");
  CHECK_THROWS(R(""));
  CHECK_THROWS(R("abc"));
  CHECK_THROWS(R("1/0"));
  CHECK_THROWS(R("1..2"));
  CHECK_THROWS(R("-"));
}

void testPolygonConstruction() {
  const Polygon sq = path("M 0 0 L 1/2 0 L 1/2 1/2 L 0 1/2 Z");
  CHECK(sq.size() == 4);
  CHECK(sq.area() == Rational(1, 4));
  CHECK(sq.centroid() == P("1/4", "1/4"));
  CHECK(sq.bounds() == (Box{P("0", "0"), P("1/2", "1/2")}));

  // Clockwise input is normalised to counter-clockwise.
  const Polygon cw = path("M 0 0 L 0 1/2 L 1/2 1/2 L 1/2 0 Z");
  CHECK(cw.area() == Rational(1, 4));
  CHECK(orientation(cw.vertices()[0], cw.vertices()[1], cw.vertices()[2]) > 0);

  // Straight-through vertices are removed.
  CHECK(path("M 0 0 L 1/4 0 L 1/2 0 L 1/2 1/2 L 0 1/2 Z").size() == 4);

  // Relative commands, H/V.
  const Polygon rel = path("m 1/4 1/4 h 1/2 v 1/2 h -1/2 z");
  CHECK(rel.area() == Rational(1, 4));
  CHECK(rel.bounds() == (Box{P("1/4", "1/4"), P("3/4", "3/4")}));

  // Implicit lineto after M, decimals and commas.
  const Polygon tri = path("M0,0 1,0 0.0,1");
  CHECK(tri.area() == Rational(1, 2));
  CHECK(tri.centroid() == P("1/3", "1/3"));

  // An explicit return to the start point is dropped.
  CHECK(path("M 0 0 L 1 0 L 0 1 L 0 0 Z").size() == 3);

  // Exact round trip through the path string.
  const Polygon odd = path("M 0 0 L 2/7 1/9 L 1/3 5/11 L -1/13 1/5 Z");
  CHECK(path(odd.toPathString()) == odd);

  // Invalid inputs.
  CHECK_THROWS(path("M 0 0 L 1 1 L 1 0 L 0 1 Z"));            // bow-tie, self-crossing
  CHECK_THROWS(path("M 0 0 L 1/2 0 L 1 0 Z"));                // all collinear
  CHECK_THROWS(path("M 0 0 L 0 0 L 1 0 L 0 1 Z"));            // zero-length edge
  CHECK_THROWS(path("M 0 0 L 1 0 L 2 0 L 1 0 L 1 1 Z"));      // zero-width spike
  CHECK_THROWS(path("M 0 0 L 4 0 L 3 2 L 2 0 L 1 2 Z"));      // vertex touches own edge
  CHECK_THROWS(path("M 0 0 L 1 0 Z"));                        // too few vertices
  CHECK_THROWS(path("L 0 0 L 1 0 L 0 1 Z"));                  // must start with M
  CHECK_THROWS(path("M 0 0 C 1 0 1 1 0 1 Z"));                // curves unsupported
  CHECK_THROWS(path("M 0 0 L 1 0 L 0 1 Z M 2 2 L 3 2 L 2 3 Z"));  // two subpaths
  CHECK_THROWS(path("M 0 0 L 1"));                            // missing coordinate
  CHECK_THROWS(path("M 0 0 L 1 0 L 0 # Z"));                  // garbage
}

void testLocate() {
  const Polygon sq = rect("0", "0", "1/2", "1/2");
  CHECK(sq.locate(P("1/4", "1/4")) == Location::Inside);
  CHECK(sq.locate(P("1/2", "1/4")) == Location::Boundary);
  CHECK(sq.locate(P("0", "0")) == Location::Boundary);
  CHECK(sq.locate(P("1/4", "1/2")) == Location::Boundary);
  CHECK(sq.locate(P("3/4", "1/4")) == Location::Outside);
  CHECK(sq.locate(P("-1/4", "1/4")) == Location::Outside);

  // Ray passing exactly through a vertex.
  const Polygon tri = path("M 0 0 L 1 0 L 1/2 1/2 Z");
  CHECK(tri.locate(P("0", "1/2")) == Location::Outside);
  CHECK(tri.locate(P("1/2", "1/4")) == Location::Inside);
  CHECK(tri.locate(P("1/4", "0")) == Location::Boundary);

  // Non-convex: the notch of a U is outside.
  const Polygon u = path("M 0 0 L 3/4 0 L 3/4 3/4 L 1/2 3/4 L 1/2 1/4 L 1/4 1/4 L 1/4 3/4 L 0 3/4 Z");
  CHECK(u.locate(P("3/8", "1/2")) == Location::Outside);
  CHECK(u.locate(P("1/8", "1/2")) == Location::Inside);
  CHECK(u.locate(P("3/8", "1/4")) == Location::Boundary);
}

void testRotation() {
  CHECK(Rotation::fromHalfAngleTangent(1) == Rotation::quarterTurns(1));
  const Rotation r345 = Rotation::fromHalfAngleTangent(Rational(1, 2));
  CHECK(r345.cos() == Rational(3, 5));
  CHECK(r345.sin() == Rational(4, 5));
  CHECK(r345 * r345.inverse() == Rotation());
  CHECK(Rotation::quarterTurns(1) * Rotation::quarterTurns(1) == Rotation::quarterTurns(2));
  CHECK(Rotation::quarterTurns(-1) == Rotation::quarterTurns(3));
  CHECK(Rotation::quarterTurns(1).apply(P("1", "0")) == P("0", "1"));
  CHECK(Rotation::fromCosSin(Rational(5, 13), Rational(-12, 13)).sin() == Rational(-12, 13));
  CHECK_THROWS(Rotation::fromCosSin(Rational(1, 2), Rational(1, 2)));

  const Polygon tri = path("M 0 0 L 1/2 0 L 1/5 1/3 Z");
  const Polygon turned = tri.rotated(r345);
  CHECK(turned.area() == tri.area());          // exact, no drift
  CHECK(turned.centroid() == tri.centroid());  // rotated about its centroid
  Polygon back = tri;
  for (int k = 0; k < 4; ++k) back = back.rotated(Rotation::quarterTurns(1), P("1/7", "2/9"));
  CHECK(back == tri);

  const Polygon moved = tri.translated(P("1/3", "-1/4"));
  CHECK(moved.bounds().min == P("1/3", "-1/4"));
  CHECK(tri.scaled(2, P("0", "0")).area() == tri.area() * 4);
  CHECK(tri.scaled(-1, P("0", "0")).area() == tri.area());
}

void testOverlap() {
  const Polygon a = rect("0", "0", "1/2", "1/2");

  CHECK(!overlap(a, rect("1/2", "0", "1", "1/2")));        // shared edge
  CHECK(!overlap(a, rect("1/2", "1/2", "1", "1")));        // shared corner
  CHECK(!overlap(a, rect("1/2", "1/4", "1", "3/4")));      // partial shared edge
  CHECK(!overlap(a, rect("3/5", "0", "1", "1/2")));        // separated
  CHECK(overlap(a, rect("1/4", "1/4", "3/4", "3/4")));     // corner overlap
  CHECK(overlap(a, a));                                     // identical
  CHECK(overlap(rect("0", "0", "1", "1"), rect("1/4", "1/4", "1/2", "1/2")));  // containment
  CHECK(overlap(rect("0", "0", "1", "1"), rect("0", "0", "1/4", "1/4")));      // contained, shared edges

  // Plus sign: edges cross, but no vertex lies inside the other shape.
  CHECK(overlap(rect("0", "2/5", "1", "3/5"), rect("2/5", "0", "3/5", "1")));

  // Non-convex U with a plug in the notch.
  const Polygon u = path("M 0 0 L 3/4 0 L 3/4 3/4 L 1/2 3/4 L 1/2 1/4 L 1/4 1/4 L 1/4 3/4 L 0 3/4 Z");
  CHECK(!overlap(u, rect("1/4", "1/4", "1/2", "3/4")));    // exact fit, touches 3 sides
  CHECK(!overlap(u, rect("1/4", "1/4", "1/2", "1")));      // sticks out of the top
  CHECK(overlap(u, rect("1/4", "1/4", "0.6", "3/4")));     // too wide
  CHECK(overlap(u, rect("1/4", "1/5", "1/2", "3/4")));     // too low
  CHECK(!overlap(u, rect("0", "3/4", "3/4", "1")));        // lid on top

  // Vertex touching an edge.
  CHECK(!overlap(a, path("M 1/4 1/2 L 1/2 3/4 L 0 3/4 Z")));
  CHECK(overlap(a, path("M 1/4 0.49 L 1/2 3/4 L 0 3/4 Z")));

  // Exactness with a rotated shape: an edge-to-edge neighbour touches but
  // does not overlap, while shifting it by 1/100 inwards does overlap.
  const Rotation r = Rotation::fromHalfAngleTangent(Rational(1, 3));  // cos 4/5, sin 3/5
  const Polygon piece = rect("0", "0", "1/2", "1/4").rotated(r, P("0", "0"));
  const Polygon neighbour = piece.translated(r.apply(P("1/2", "0")));
  const Polygon pushed = piece.translated(r.apply(P("0.49", "0")));
  CHECK(!overlap(piece, neighbour));
  CHECK(overlap(piece, pushed));
}

// Independent reference for convex polygons: separating axis theorem.
// Interiors are disjoint iff some edge normal separates the projections
// (touching projections count as separated).
bool satOverlap(const Polygon& a, const Polygon& b) {
  for (const Polygon* poly : {&a, &b}) {
    const auto& v = poly->vertices();
    for (std::size_t i = 0; i < v.size(); ++i) {
      const Point e = v[(i + 1) % v.size()] - v[i];
      const Point normal(-e.y, e.x);
      auto project = [&](const Polygon& p, Rational& lo, Rational& hi) {
        lo = hi = dot(normal, p.vertices()[0]);
        for (const Point& q : p.vertices()) {
          const Rational d = dot(normal, q);
          if (d < lo) lo = d;
          if (d > hi) hi = d;
        }
      };
      Rational aLo, aHi, bLo, bHi;
      project(a, aLo, aHi);
      project(b, bLo, bHi);
      if (aHi <= bLo || bHi <= aLo) return false;
    }
  }
  return true;
}

void testOverlapAgainstReference() {
  // Coordinates on a coarse grid so that touching / collinear cases are common.
  std::mt19937 rng(12345);
  std::uniform_int_distribution<int> coord(0, 4);
  auto randomTriangle = [&]() {
    while (true) {
      std::vector<Point> v;
      for (int k = 0; k < 3; ++k) v.emplace_back(Rational(coord(rng), 4), Rational(coord(rng), 4));
      if (orientation(v[0], v[1], v[2]) != 0) return Polygon(v);
    }
  };
  int mismatches = 0;
  int overlapping = 0;
  for (int iter = 0; iter < 20000; ++iter) {
    const Polygon a = randomTriangle();
    const Polygon b = randomTriangle();
    const bool expected = satOverlap(a, b);
    overlapping += expected;
    if (overlap(a, b) != expected && ++mismatches <= 5) {
      std::cerr << "  mismatch: " << a.toPathString() << " vs " << b.toPathString()
                << " expected " << expected << '\n';
    }
  }
  CHECK(mismatches == 0);
  CHECK(overlapping > 1000 && overlapping < 19000);  // the sample exercises both outcomes
}

void testGenerator() {
  std::mt19937_64 rng(7);
  GeneratorConfig config;
  const std::vector<Rotation> rotations = defaultRotations();
  for (int i = 0; i < 200; ++i) {
    const Polygon p = randomPolygon(rng, config);
    CHECK(p.bounds().min == P("0", "0"));
    CHECK(static_cast<int>(p.size()) <= config.maxVertices);
    for (const Rotation& r : rotations) {
      Polygon q = p.rotated(r, P("0", "0"));
      q = q.translated(-q.bounds().min);
      CHECK(isContainedIn(q, unitSquare()));
    }
  }
  GeneratorConfig bad;
  bad.maxRadius = 0.6;
  CHECK_THROWS(validate(bad));
}

void testSlide() {
  const Box unit = unitSquare();
  const Polygon sq = rect("1/2", "3/4", "3/4", "1");

  // Empty bin: falls to the floor, then slides to the left wall.
  CHECK(maxSlide(sq, Direction::Down, {}, unit) == Rational(3, 4));
  CHECK(settleBottomLeft(sq, {}, unit) == rect("0", "0", "1/4", "1/4"));

  // Lands exactly on a shelf.
  const std::vector<Polygon> shelf = {rect("0", "0", "1", "1/4")};
  CHECK(maxSlide(sq, Direction::Down, shelf, unit) == Rational(1, 2));

  // Grazing: touching only at a corner and sliding along a shared vertical
  // line must not block the motion.
  const std::vector<Polygon> post = {rect("0", "0", "1/4", "1/2")};
  CHECK(maxSlide(rect("1/4", "1/2", "1/2", "3/4"), Direction::Down, post, unit) == Rational(1, 2));
  CHECK(maxSlide(rect("1/4", "0", "1/2", "1/4"), Direction::Left, post, unit) == 0);

  // A rotated triangle settling onto a slanted surface rests exactly in contact.
  const std::vector<Polygon> ramp = {path("M 0 0 L 1 0 L 0 1/2 Z")};
  const Polygon tri = path("M 0 0 L 1/5 0 L 1/10 1/6 Z")
                          .rotated(Rotation::fromHalfAngleTangent(Rational(1, 2)), P("0", "0"));
  const Polygon start = tri.translated(P("1/2", "3/4") - tri.bounds().min);
  const Polygon rest = settleBottomLeft(start, ramp, unit);
  CHECK(!overlap(rest, ramp[0]));
  CHECK(isContainedIn(rest, unit));
  CHECK(maxSlide(rest, Direction::Down, ramp, unit) == 0);
  CHECK(maxSlide(rest, Direction::Left, ramp, unit) == 0);
}

// Every packing produced by the environment must be exactly valid.
bool packingIsValid(const PackingEnv& env) {
  for (const Bin& bin : env.allBins()) {
    for (std::size_t i = 0; i < bin.items().size(); ++i) {
      if (!isContainedIn(bin.items()[i], bin.bounds())) return false;
      for (std::size_t j = i + 1; j < bin.items().size(); ++j) {
        if (interiorsOverlap(bin.items()[i], bin.items()[j])) return false;
      }
    }
  }
  return true;
}

// The fast mask must agree exactly with the plain definition of legality:
// the item, put at the grid spot, fits in the bin and overlaps nothing.
void checkMaskAgainstBruteForce(const PackingEnv& env) {
  for (int a = 0; a < env.actionCount(); ++a) {
    const Action act = env.decodeAction(a);
    if (act.slot == env.config().openBins) continue;  // the new-bin slot is a policy choice
    const std::vector<Polygon>& items = env.slotItems(act.slot);
    const std::vector<Bin> bins = env.allBins();
    if (static_cast<std::size_t>(act.slot) >= std::min<std::size_t>(bins.size(), env.config().openBins)) {
      CHECK(!env.isLegal(a));  // that bin is not open yet
      continue;
    }
    Bin bin;
    for (const Polygon& p : items) bin.tryPlace(p);
    CHECK(env.isLegal(a) == bin.canPlace(env.anchorPlacement(a)));
  }
}

void testEnv() {
  EnvConfig config;
  config.itemsPerEpisode = 25;
  PackingEnv env(config);
  CHECK(env.actionCount() == 3 * 8 * 16 * 16);
  for (int a : {0, 77, 4095, 6143}) CHECK(env.encodeAction(env.decodeAction(a)) == a);

  // First item: only the "new bin" slot at spot (0, 0), any rotation, is legal.
  env.reset(42);
  for (int a = 0; a < env.actionCount(); ++a) {
    const Action act = env.decodeAction(a);
    CHECK(env.isLegal(a) == (act.slot == config.openBins && act.gx == 0 && act.gy == 0));
  }
  CHECK_THROWS(env.step(0));

  // Random legal play gives exact, valid packings, and the mask stays exact.
  std::mt19937 rng(3);
  int steps = 0;
  float totalReward = 0;
  while (!env.done()) {
    if (steps % 6 == 3) checkMaskAgainstBruteForce(env);
    std::vector<int> legal;
    for (int a = 0; a < env.actionCount(); ++a) {
      if (env.isLegal(a)) legal.push_back(a);
    }
    totalReward += env.step(legal[rng() % legal.size()]).reward;
    ++steps;
  }
  CHECK(steps == config.itemsPerEpisode);
  CHECK(packingIsValid(env));
  CHECK(-totalReward == static_cast<float>(env.binsUsed()));
  CHECK(env.binsUsed() >= env.areaLowerBound());

  // Baselines: deterministic for a seed and exactly valid. The mask is also
  // checked on the denser packings they make.
  PackingEnv a(config), b(config);
  a.reset(9);
  b.reset(9);
  int k = 0;
  while (!a.done()) {
    CHECK(a.currentItem() == b.currentItem());  // same seed, same items
    if (++k % 8 == 0) checkMaskAgainstBruteForce(b);
    a.step(firstFitAction(a));
    b.step(greedyBottomLeftAction(b));
  }
  CHECK(packingIsValid(a));
  CHECK(packingIsValid(b));

  // Without settling, the item stays exactly where it was put, and an empty
  // bin offers every spot where the item fits.
  EnvConfig free = config;
  free.settle = false;
  free.maskThreads = 4;
  PackingEnv f(free);
  f.reset(11);
  int offered = 0;
  for (int act = 0; act < f.actionCount(); ++act) offered += f.isLegal(act) ? 1 : 0;
  CHECK(offered > f.rotations());
  std::mt19937 rng2(5);
  while (!f.done()) {
    std::vector<int> legal;
    for (int act = 0; act < f.actionCount(); ++act) {
      if (f.isLegal(act)) legal.push_back(act);
    }
    const int pick = legal[rng2() % legal.size()];
    CHECK(f.simulate(pick) == f.anchorPlacement(pick));
    f.step(pick);
  }
  CHECK(packingIsValid(f));

  // Observation buffers have the documented sizes and agree with the mask.
  a.reset(5);
  a.step(firstFitAction(a));
  const std::size_t px = static_cast<std::size_t>(a.rasterSize()) * a.rasterSize();
  std::vector<std::uint8_t> bins(a.slots() * px), items(a.rotations() * px),
      contact(a.slots() * a.rotations() * px), mask(a.actionCount());
  std::vector<float> slotF(a.slots() * PackingEnv::kSlotFeatures), globalF(PackingEnv::kGlobalFeatures);
  a.observe(bins.data(), items.data(), contact.data(), slotF.data(), globalF.data(), mask.data());
  CHECK(mask == a.mask());
  CHECK(slotF[0] == 1.0f && slotF[1] == 0.0f);  // slot 0 holds the open bin
  CHECK(std::count_if(bins.begin(), bins.begin() + px, [](std::uint8_t v) { return v > 0; }) > 0);
  // In the empty new-bin slot, the bottom-left corner touches two walls, the
  // middle of the bin touches nothing.
  const std::size_t newBin = static_cast<std::size_t>(a.config().openBins) * a.rotations() * px;
  const int last = a.rasterSize() - 1;
  CHECK(contact[newBin + static_cast<std::size_t>(last) * a.rasterSize()] > 0);
  CHECK(contact[newBin + static_cast<std::size_t>(a.rasterSize() / 2) * a.rasterSize() + 2] == 0 ||
        contact[newBin + static_cast<std::size_t>(a.rasterSize() / 2) * a.rasterSize() + 2] <
            contact[newBin + static_cast<std::size_t>(last) * a.rasterSize()]);
}

void testBin() {
  Bin bin;
  CHECK(bin.tryPlace(rect("0", "0", "1/2", "1/2")));
  CHECK(bin.tryPlace(rect("1/2", "0", "1", "1/2")));
  CHECK(bin.tryPlace(rect("0", "1/2", "1/2", "1")));
  CHECK(!bin.tryPlace(rect("1/4", "1/4", "3/4", "3/4")));   // overlaps
  CHECK(!bin.tryPlace(rect("1/2", "1/2", "3/2", "1")));     // outside the box
  CHECK(bin.tryPlace(rect("1/2", "1/2", "1", "1")));
  CHECK(bin.items().size() == 4);
  CHECK(bin.fillRatio() == 1);
  CHECK(!bin.canPlace(rect("0", "0", "1/10", "1/10")));
  CHECK(bin.toSvg().find("<svg") == 0);
}

void testOffline() {
  EnvConfig config;
  config.itemsPerEpisode = 30;

  // episodeItems reproduces the online item sequence exactly.
  PackingEnv env(config);
  env.reset(5);
  const std::vector<Polygon> items = episodeItems(config, 5);
  CHECK(items.size() == 30);
  int k = 0;
  while (!env.done()) {
    CHECK(env.currentItem() == items[static_cast<std::size_t>(k)]);
    ++k;
    env.step(greedyBottomLeftAction(env));
  }
  CHECK(k == 30);

  // resetWithItems feeds exactly those items, in that order.
  PackingEnv fixed(config);
  fixed.resetWithItems(items);
  CHECK(fixed.currentItem() == items[0]);
  CHECK_THROWS(fixed.resetWithItems(std::vector<Polygon>(items.begin(), items.begin() + 3)));

  for (std::uint64_t seed : {5ULL, 11ULL, 1000000007ULL}) {
    const OfflineResult r = packOfflineEpisode(config, seed);
    const std::vector<Polygon> its = episodeItems(config, seed);
    CHECK(r.bins == static_cast<int>(r.packing.size()));
    CHECK(r.bins >= r.lowerBound);
    CHECK(r.lowerBound >= 1);
    CHECK(!r.pattern.empty());
    CHECK(r.patternsTried >= 1 && r.patternsTried <= kDefaultOfflinePatterns);
    // Every item is packed exactly once (up to rotation/translation: compare
    // areas as a multiset) and the packing is valid.
    std::vector<Rational> want, got;
    for (const Polygon& p : its) want.push_back(p.area());
    for (const Bin& b : r.packing) {
      const std::vector<Polygon>& in = b.items();
      CHECK(!in.empty());
      for (std::size_t i = 0; i < in.size(); ++i) {
        got.push_back(in[i].area());
        CHECK(isContainedIn(in[i], b.bounds()));
        for (std::size_t j = i + 1; j < in.size(); ++j) CHECK(!interiorsOverlap(in[i], in[j]));
      }
    }
    std::sort(want.begin(), want.end());
    std::sort(got.begin(), got.end());
    CHECK(want == got);
    // Offline sees everything and never closes bins: never worse than the
    // bounded-space greedy player on the same episode.
    PackingEnv online(config);
    online.reset(seed);
    while (!online.done()) online.step(greedyBottomLeftAction(online));
    CHECK(r.bins <= online.binsUsed());
  }

  // The threaded batch version gives the same answers as one at a time.
  const std::vector<OfflineResult> batch = packOfflineEpisodes(config, 20, 6, 6, 4);
  CHECK(batch.size() == 6);
  for (int i = 0; i < 6; ++i) {
    const OfflineResult one = packOfflineEpisode(config, 20 + static_cast<std::uint64_t>(i), 6);
    CHECK(batch[static_cast<std::size_t>(i)].bins == one.bins);
    CHECK(batch[static_cast<std::size_t>(i)].pattern == one.pattern);
    CHECK(batch[static_cast<std::size_t>(i)].lowerBound == one.lowerBound);
  }
}

}  // namespace

int main() {
  const std::vector<std::pair<const char*, std::function<void()>>> tests = {
      {"rational", testRational},
      {"polygon construction", testPolygonConstruction},
      {"locate", testLocate},
      {"rotation", testRotation},
      {"overlap", testOverlap},
      {"overlap vs SAT reference (random triangles)", testOverlapAgainstReference},
      {"bin", testBin},
      {"generator", testGenerator},
      {"slide", testSlide},
      {"environment", testEnv},
      {"offline packer", testOffline},
  };
  for (const auto& [name, fn] : tests) {
    std::cout << "[ RUN  ] " << name << '\n';
    try {
      fn();
    } catch (const std::exception& e) {
      ++g_failures;
      std::cerr << "  FAIL unexpected exception: " << e.what() << '\n';
    }
  }
  std::cout << g_checks << " checks, " << g_failures << " failures\n";
  return g_failures == 0 ? 0 : 1;
}
