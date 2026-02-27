#ifndef WO_COULOMB_TEST_TOLERANCE_H
#define WO_COULOMB_TEST_TOLERANCE_H

#include <cmath>
#include <gtest/gtest.h>

namespace wo_test {

struct Tolerance {
    double abs;
    double rel;
};

constexpr Tolerance kSmallTolerance{1e-6, 5e-3};
constexpr Tolerance kMediumTolerance{1e-6, 2e-3};
constexpr Tolerance kGoldenTolerance{1e-8, 5e-5};

inline void expectNearRelAbs(double value, double reference, const Tolerance& tol)
{
    const double diff = std::abs(value - reference);
    const double limit = tol.abs + tol.rel * std::abs(reference);
    EXPECT_LE(diff, limit) << "value=" << value << " reference=" << reference
                           << " abs_diff=" << diff << " limit=" << limit;
}

}  // namespace wo_test

#endif  // WO_COULOMB_TEST_TOLERANCE_H
