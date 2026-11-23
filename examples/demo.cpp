// Places a few hand-drawn shapes into the 1x1 bin and writes an SVG picture.
//
//   ./polypack_demo bin.svg

#include "polypack/polypack.hpp"

#include <fstream>
#include <iostream>

using namespace polypack;

int main(int argc, char** argv) {
  Bin bin;

  const Polygon shapes[] = {
      Polygon::fromSvgPath("M 0 0 L 1/2 0 L 1/2 1/3 L 0 1/3 Z"),
      Polygon::fromSvgPath("M 1/2 0 L 1 0 L 3/4 1/2 Z"),
      // L-shape, then rotated by the exact 3-4-5 rotation about its centroid.
      Polygon::fromSvgPath("M 0 0 h 1/4 v 1/8 h -1/8 v 1/4 h -1/8 z")
          .rotated(Rotation::fromHalfAngleTangent(Rational(1, 2)))
          .translated(Point(Rational(1, 5), Rational(1, 2))),
      Polygon::fromSvgPath("M 1/2 1/2 L 1 1/2 L 1 1 L 1/2 1 L 3/4 3/4 Z"),
      // This one overlaps the triangle above and will be rejected.
      Polygon::fromSvgPath("M 0.7 0.1 l 0.2 0 l 0 0.2 l -0.2 0 z"),
  };

  for (const Polygon& shape : shapes) {
    const bool placed = bin.tryPlace(shape);
    std::cout << (placed ? "placed   " : "rejected ") << shape.toPathString() << '\n';
  }
  std::cout << "fill ratio = " << toString(bin.fillRatio()) << " (~" << toDouble(bin.fillRatio())
            << ")\n";

  if (argc > 1) {
    std::ofstream(argv[1]) << bin.toSvg();
    std::cout << "wrote " << argv[1] << '\n';
  }
  return 0;
}
