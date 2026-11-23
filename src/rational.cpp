#include "polypack/rational.hpp"

#include <stdexcept>
#include <string>

namespace polypack {
namespace {

[[noreturn]] void fail(std::string_view whole) {
  throw std::invalid_argument("invalid rational number: \"" + std::string(whole) + "\"");
}

// [+-] digits [. digits], parsed exactly.
Rational parseDecimal(std::string_view s, std::string_view whole) {
  std::size_t i = 0;
  bool negative = false;
  if (i < s.size() && (s[i] == '+' || s[i] == '-')) {
    negative = s[i] == '-';
    ++i;
  }
  std::string digits;
  std::size_t fractionDigits = 0;
  bool seenDot = false;
  for (; i < s.size(); ++i) {
    const char c = s[i];
    if (c >= '0' && c <= '9') {
      digits.push_back(c);
      if (seenDot) ++fractionDigits;
    } else if (c == '.' && !seenDot) {
      seenDot = true;
    } else {
      fail(whole);
    }
  }
  if (digits.empty()) fail(whole);

  mpz_class numerator(digits, 10);
  mpz_class denominator;
  mpz_ui_pow_ui(denominator.get_mpz_t(), 10, fractionDigits);
  Rational value(numerator, denominator);
  value.canonicalize();
  if (negative) value = -value;
  return value;
}

}  // namespace

Rational parseRational(std::string_view text) {
  const std::size_t slash = text.find('/');
  if (slash == std::string_view::npos) return parseDecimal(text, text);

  const Rational numerator = parseDecimal(text.substr(0, slash), text);
  const Rational denominator = parseDecimal(text.substr(slash + 1), text);
  if (denominator == 0) {
    throw std::invalid_argument("division by zero in rational number: \"" + std::string(text) + "\"");
  }
  return Rational(numerator / denominator);
}

std::string toString(const Rational& value) { return value.get_str(); }

double toDouble(const Rational& value) { return value.get_d(); }

int signOf(const Rational& value) {
  const int s = sgn(value);
  return (s > 0) - (s < 0);
}

}  // namespace polypack
