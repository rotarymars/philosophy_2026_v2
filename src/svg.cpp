#include "polypack/svg.hpp"

#include <algorithm>
#include <sstream>

namespace polypack {

std::string renderSvg(const std::vector<Polygon>& polygons, const Box& frame, int sizePx) {
  const double minX = toDouble(frame.min.x);
  const double maxY = toDouble(frame.max.y);
  const double width = toDouble(frame.max.x) - minX;
  const double height = maxY - toDouble(frame.min.y);
  const double scale = sizePx / std::max(width, height);
  const double w = width * scale;
  const double h = height * scale;

  std::ostringstream out;
  out << "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"" << w << "\" height=\"" << h
      << "\" viewBox=\"-2 -2 " << w + 4 << ' ' << h + 4 << "\">\n";
  out << "  <rect x=\"0\" y=\"0\" width=\"" << w << "\" height=\"" << h
      << "\" fill=\"#fafafa\" stroke=\"#222\" stroke-width=\"1.5\"/>\n";
  for (std::size_t k = 0; k < polygons.size(); ++k) {
    const int hue = static_cast<int>((k * 137) % 360);  // golden-angle colour spread
    out << "  <polygon fill=\"hsl(" << hue << ",65%,62%)\" fill-opacity=\"0.85\" "
        << "stroke=\"#333\" stroke-width=\"1\" points=\"";
    for (const Point& p : polygons[k].vertices()) {
      out << (toDouble(p.x) - minX) * scale << ',' << (maxY - toDouble(p.y)) * scale << ' ';
    }
    out << "\"/>\n";
  }
  out << "</svg>\n";
  return out.str();
}

}  // namespace polypack
