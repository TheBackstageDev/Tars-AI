#ifndef TARS_MATH_DERIVATES_HPP
#define TARS_MATH_DERIVATES_HPP

namespace TMATH
{
    double derive(double (*f)(double), double x0, double h = 1e-5)
    {
        return (f(x0 + h) - f(x0 - h)) / (2 * h);
    }
} // namespace TMATH

#endif // TARS_MATH_DERIVATES_HPP