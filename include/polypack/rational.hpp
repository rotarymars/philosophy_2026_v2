#pragma once

// Exact rational arithmetic used for every coordinate in polypack.
//
// No floating point is involved in geometry: every coordinate, length and
// rotation is an exact fraction, so predicates such as "do these two shapes
// overlap?" or "do they merely touch?" are answered exactly.
//
// NOTE: gmpxx uses expression templates. Never store an arithmetic result in
// an `auto` variable; always spell out `Rational`.

#include <gmpxx.h>

#include <string>
#include <string_view>

namespace polypack {

using Rational = mpq_class;

// Parses "3", "-7", "1/3", "-2/5", "0.125", ".5", "1.5/4", "1/-2".
// Decimals are converted exactly (0.1 == 1/10). Throws std::invalid_argument.
Rational parseRational(std::string_view text);

// Exact text form: "p" or "p/q" (round-trips through parseRational).
std::string toString(const Rational& value);

// Lossy conversion, intended only for rendering / feeding neural networks.
double toDouble(const Rational& value);

// -1, 0 or +1.
int signOf(const Rational& value);

}  // namespace polypack
