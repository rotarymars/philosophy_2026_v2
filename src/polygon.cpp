#include "polypack/polygon.hpp"

#include <algorithm>
#include <cctype>
#include <stdexcept>
#include <string>
#include <utility>

namespace polypack {
namespace {

std::string describe(const Point& p) {
  return "(" + toString(p.x) + ", " + toString(p.y) + ")";
}

Rational signedDoubleArea(const std::vector<Point>& v) {
  Rational sum = 0;
  for (std::size_t i = 0, n = v.size(); i < n; ++i) sum += cross(v[i], v[(i + 1) % n]);
  return sum;
}

// ---- SVG-style path parsing ------------------------------------------------

struct Token {
  bool isCommand;
  char command;
  std::string_view text;
};

std::vector<Token> tokenize(std::string_view s) {
  std::vector<Token> tokens;
  std::size_t i = 0;
  while (i < s.size()) {
    const char c = s[i];
    if (std::isspace(static_cast<unsigned char>(c)) || c == ',') {
      ++i;
      continue;
    }
    if (std::isalpha(static_cast<unsigned char>(c))) {
      tokens.push_back({true, c, s.substr(i, 1)});
      ++i;
      continue;
    }
    // Number: [+-] then digits, '.', and '/' (a '/' may be followed by a sign).
    const std::size_t start = i;
    if (c == '+' || c == '-') ++i;
    while (i < s.size()) {
      const char d = s[i];
      if (std::isdigit(static_cast<unsigned char>(d)) || d == '.') {
        ++i;
      } else if (d == '/') {
        ++i;
        if (i < s.size() && (s[i] == '+' || s[i] == '-')) ++i;
      } else {
        break;
      }
    }
    if (i == start) {
      throw std::invalid_argument("svg path: unexpected character '" + std::string(1, c) + "'");
    }
    tokens.push_back({false, 0, s.substr(start, i - start)});
  }
  return tokens;
}

}  // namespace

// ---- Construction ------------------------------------------------------------

Polygon::Polygon(std::vector<Point> vertices) : vertices_(std::move(vertices)) {
  removeRedundantVertices();
  if (vertices_.size() < 3) {
    throw std::invalid_argument("polygon needs at least 3 non-collinear vertices");
  }
  ensureSimple();
  if (signOf(signedDoubleArea(vertices_)) < 0) std::reverse(vertices_.begin(), vertices_.end());
  computeCache();
}

Polygon::Polygon(Trusted, std::vector<Point> vertices) : vertices_(std::move(vertices)) {
  computeCache();
}

void Polygon::removeRedundantVertices() {
  for (std::size_t i = 0, n = vertices_.size(); i < n && n >= 2; ++i) {
    if (vertices_[i] == vertices_[(i + 1) % n]) {
      throw std::invalid_argument("polygon has a zero-length edge at " + describe(vertices_[i]));
    }
  }
  bool changed = true;
  while (changed && vertices_.size() >= 3) {
    changed = false;
    std::size_t i = 0;
    while (i < vertices_.size() && vertices_.size() >= 3) {
      const std::size_t n = vertices_.size();
      const Point& prev = vertices_[(i + n - 1) % n];
      const Point& cur = vertices_[i];
      const Point& next = vertices_[(i + 1) % n];
      if (orientation(prev, cur, next) == 0) {
        if (signOf(dot(cur - prev, next - cur)) > 0) {
          // Straight-through vertex: it adds nothing to the shape.
          vertices_.erase(vertices_.begin() + static_cast<std::ptrdiff_t>(i));
          changed = true;
          continue;
        }
        throw std::invalid_argument("polygon is degenerate (folds back on itself) at " +
                                    describe(cur));
      }
      ++i;
    }
  }
}

void Polygon::ensureSimple() const {
  const std::size_t n = vertices_.size();
  for (std::size_t i = 0; i < n; ++i) {
    const Point& a = vertices_[i];
    const Point& b = vertices_[(i + 1) % n];
    for (std::size_t j = i + 1; j < n; ++j) {
      const bool adjacent = (j == i + 1) || (i == 0 && j == n - 1);
      if (adjacent) continue;  // share exactly one vertex (spikes were rejected earlier)
      const Point& c = vertices_[j];
      const Point& d = vertices_[(j + 1) % n];
      if (!segmentBoxesTouch(a, b, c, d)) continue;
      if (intersectSegments(a, b, c, d).kind != SegmentIntersection::Kind::None) {
        throw std::invalid_argument("polygon is not simple: edge " + describe(a) + "-" +
                                    describe(b) + " meets edge " + describe(c) + "-" +
                                    describe(d));
      }
    }
  }
}

void Polygon::computeCache() {
  area_ = signedDoubleArea(vertices_) / 2;
  bounds_ = Box{vertices_[0], vertices_[0]};
  for (const Point& p : vertices_) {
    if (p.x < bounds_.min.x) bounds_.min.x = p.x;
    if (p.y < bounds_.min.y) bounds_.min.y = p.y;
    if (p.x > bounds_.max.x) bounds_.max.x = p.x;
    if (p.y > bounds_.max.y) bounds_.max.y = p.y;
  }
}

Polygon Polygon::fromSvgPath(std::string_view path) {
  const std::vector<Token> tokens = tokenize(path);
  std::vector<Point> points;
  Point current(0, 0);
  char command = 0;
  bool closed = false;
  std::size_t i = 0;

  auto number = [&]() -> Rational {
    if (i >= tokens.size() || tokens[i].isCommand) {
      throw std::invalid_argument(std::string("svg path: expected a number after '") + command +
                                  "'");
    }
    return parseRational(tokens[i++].text);
  };

  while (i < tokens.size()) {
    if (closed) {
      throw std::invalid_argument("svg path: content after Z (only one closed polygon is supported)");
    }
    if (tokens[i].isCommand) {
      command = tokens[i].command;
      ++i;
      const bool isMove = command == 'M' || command == 'm';
      if (points.empty() && !isMove) throw std::invalid_argument("svg path: must start with M");
      if (!points.empty() && isMove) {
        throw std::invalid_argument("svg path: multiple subpaths are not supported");
      }
      if (command == 'Z' || command == 'z') {
        closed = true;
        continue;
      }
    } else if (command == 0) {
      throw std::invalid_argument("svg path: must start with M");
    }

    switch (command) {
      case 'M':
      case 'L': {
        Rational x = number();
        Rational y = number();
        current = Point(std::move(x), std::move(y));
        break;
      }
      case 'm':
      case 'l': {
        Rational dx = number();
        Rational dy = number();
        current = current + Point(std::move(dx), std::move(dy));
        break;
      }
      case 'H': current.x = number(); break;
      case 'h': current.x += number(); break;
      case 'V': current.y = number(); break;
      case 'v': current.y += number(); break;
      default:
        throw std::invalid_argument(std::string("svg path: unsupported command '") + command +
                                    "' (only M, L, H, V, Z and lower-case forms; no curves)");
    }
    points.push_back(current);
    // SVG rule: extra coordinate pairs after a moveto are linetos.
    if (command == 'M') command = 'L';
    if (command == 'm') command = 'l';
  }

  if (points.size() >= 2 && points.back() == points.front()) points.pop_back();
  return Polygon(std::move(points));
}

// ---- Queries -------------------------------------------------------------------

Point Polygon::centroid() const {
  Rational cx = 0;
  Rational cy = 0;
  for (std::size_t i = 0, n = vertices_.size(); i < n; ++i) {
    const Point& a = vertices_[i];
    const Point& b = vertices_[(i + 1) % n];
    const Rational w = cross(a, b);
    cx += (a.x + b.x) * w;
    cy += (a.y + b.y) * w;
  }
  const Rational denominator = 6 * area_;
  return Point(Rational(cx / denominator), Rational(cy / denominator));
}

Location Polygon::locate(const Point& p) const {
  bool inside = false;
  for (std::size_t i = 0, n = vertices_.size(); i < n; ++i) {
    const Point& a = vertices_[i];
    const Point& b = vertices_[(i + 1) % n];
    if (onSegment(p, a, b)) return Location::Boundary;
    // Ray towards +x, half-open rule on y so vertices are counted once.
    if (a.y <= p.y && b.y > p.y) {
      if (orientation(a, b, p) > 0) inside = !inside;
    } else if (b.y <= p.y && a.y > p.y) {
      if (orientation(a, b, p) < 0) inside = !inside;
    }
  }
  return inside ? Location::Inside : Location::Outside;
}

// ---- Transforms ----------------------------------------------------------------

Polygon Polygon::translated(const Point& offset) const {
  std::vector<Point> v;
  v.reserve(vertices_.size());
  for (const Point& p : vertices_) v.push_back(p + offset);
  return Polygon(Trusted{}, std::move(v));
}

Polygon Polygon::rotated(const Rotation& rotation, const Point& pivot) const {
  std::vector<Point> v;
  v.reserve(vertices_.size());
  for (const Point& p : vertices_) v.push_back(rotation.apply(p, pivot));
  return Polygon(Trusted{}, std::move(v));
}

Polygon Polygon::rotated(const Rotation& rotation) const { return rotated(rotation, centroid()); }

Polygon Polygon::scaled(const Rational& factor, const Point& pivot) const {
  if (factor == 0) throw std::invalid_argument("Polygon::scaled: factor must be non-zero");
  // A negative factor is a 180-degree rotation plus scaling, so orientation is kept.
  std::vector<Point> v;
  v.reserve(vertices_.size());
  for (const Point& p : vertices_) v.push_back(pivot + (p - pivot) * factor);
  return Polygon(Trusted{}, std::move(v));
}

std::string Polygon::toPathString() const {
  std::string out;
  for (std::size_t i = 0; i < vertices_.size(); ++i) {
    out += (i == 0 ? "M " : " L ");
    out += toString(vertices_[i].x);
    out += ' ';
    out += toString(vertices_[i].y);
  }
  out += " Z";
  return out;
}

}  // namespace polypack
