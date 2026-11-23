# polypack

Exact 2D polygon geometry for an online bin-packing playground (1x1 container).

* Coordinates are exact rationals (GMP `mpq_class`), so there's no floating-point error.
* Polygons are written as SVG-like paths: `M 0 0 L 1/2 0 L 1/4 1/3 Z` (`M L H V Z`, plus lowercase for relative moves).
* Rotations are exact. They're given by `t = tan(θ/2)` (any rational `t`), or as quarter turns.
* `interiorsOverlap(a, b)` gives an exact answer, and touching edges or corners is allowed.

```sh
sudo apt install libgmp-dev   # dependency
cmake -S . -B build -G Ninja
cmake --build build
ctest --test-dir build --output-on-failure
./build/polypack_demo bin.svg
```
