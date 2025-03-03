#ifndef TARS_MATH_MATRIX_COMPONENT_HPP
#define TARS_MATH_MATRIX_COMPONENT_HPP

#include "tarsmath/linear_algebra/vector_component.hpp"

#include <array>
#include <vector>
#include <assert.h>

namespace TMATH
{
    template<typename T>
    class Matrix_t
    {
    public:
        Matrix_t(size_t rows, size_t cols)
            : rows_(rows), cols_(cols), elements_(rows, std::vector<T>(cols)) {}

        Matrix_t(const std::vector<std::vector<T>>& elements)
            : elements_(elements), rows_(elements.size()), cols_(elements[0].size()) {}

        Matrix_t(const std::vector<T>& elements)
            : rows_(elements.size()), cols_(1), elements_(rows_, std::vector<T>(1))
        {
            for (size_t i = 0; i < rows_; ++i)
            {
                elements_[i][0] = elements[i];
            }
        }

        Matrix_t(const T& x, size_t rows, size_t cols)
            : rows_(rows), cols_(cols), elements_(rows, std::vector<T>(cols, x)) {}

        inline T& at(size_t row, size_t col)
        {
            assert((row < rows_ && col < cols_) && "Matrix Index out of bounds");
            return elements_[row][col];
        }
    
        inline const T& at(size_t row, size_t col) const
        {
            assert((row < rows_ && col < cols_) && "Matrix Index out of bounds");
            return elements_[row][col];
        }

        inline std::vector<T>& rowAt(size_t row)
        {
            assert((row < rows_ ) && "Matrix Row Index out of bounds");
            return elements_[row];
        }

        inline const std::vector<T>& rowAt(size_t row) const
        {
            assert((row < rows_ ) && "Matrix Row Index out of bounds");
            return elements_[row];
        }

        inline size_t rows() const {
            return rows_;
        }
        inline size_t cols() const {
            return cols_;
        }

        constexpr bool operator==(const Matrix_t<T>& other) const {
            if (this->rows() != other.rows() || this->cols() != other.cols()) {
                return false;
            }

            for (size_t i = 0; i < rows_; ++i) {
                for (size_t j = 0; j < cols_; ++j) {
                    if (elements_[i][j] != other.elements_[i][j]) {
                        return false;
                    }
                }
            }
            return true;
        }
        constexpr bool operator!=(const Matrix_t<T>& other) const {
            return !(*this == other);
        }
        
        Matrix_t<T> operator+(const Matrix_t<T>& other) const
        {
            assert((rows_ == other.rows_ && cols_ == other.cols_) && "Matrix dimensions must agree for addition");
    
            Matrix_t<T> result(rows_, cols_);
            for (size_t i = 0; i < rows_; ++i)
            {
                for (size_t j = 0; j < cols_; ++j)
                {
                    result.at(i, j) = elements_[i][j] + other.at(i, j);
                }
            }
            return result;
        }
        Matrix_t<T> operator-(const Matrix_t<T>& other) const
        {
            assert((rows_ == other.rows_ && cols_ == other.cols_) && "Matrix dimensions must agree for subtraction");
    
            Matrix_t<T> result(rows_, cols_);
            for (size_t i = 0; i < rows_; ++i)
            {
                for (size_t j = 0; j < cols_; ++j)
                {
                    result.at(i, j) = elements_[i][j] - other.at(i, j);
                }
            }
            return result;
        }
        Matrix_t<T> operator*(const Matrix_t<T>& other) const
        {
            assert(cols_ == other.rows() && "Matrix multiplication requires the number of columns in the first matrix to be equal to the number of rows in the second matrix");
    
            Matrix_t<T> result(rows_, other.cols());
            for (size_t i = 0; i < rows_; ++i)
            {
                for (size_t j = 0; j < other.cols(); ++j)
                {
                    result.at(i, j) = T{};
                    for (size_t k = 0; k < cols_; ++k)
                    {
                        result.at(i, j) += elements_[i][k] * other.at(k, j);
                    }
                }
            }
            return result;
        }    
        Matrix_t<T> operator/(const Matrix_t<T>& other) const
        {
            assert((rows_ == other.rows_ && cols_ == other.cols_) && "Matrix dimensions must agree for division");
    
            Matrix_t<T> result(rows_, cols_);
            for (size_t i = 0; i < rows_; ++i)
            {
                for (size_t j = 0; j < cols_; ++j)
                {
                    result.at(i, j) = elements_[i][j] / other.at(i, j);
                }
            }
            return result;
        }
        Matrix_t<T> operator*(const T& scalar) const
        {
            Matrix_t<T> result(rows_, cols_);
            for (size_t i = 0; i < rows_; ++i)
            {
                for (size_t j = 0; j < cols_; ++j)
                {
                    result.at(i, j) = elements_[i][j] * scalar;
                }
            }
            return result;
        } 
        Matrix_t<T> operator/(const T& scalar) const
        {
            Matrix_t<T> result(rows_, cols_);
            for (size_t i = 0; i < rows_; ++i)
            {
                for (size_t j = 0; j < cols_; ++j)
                {
                    result.at(i, j) = elements_[i][j] / scalar;
                }
            }
            return result;
        }  
        Matrix_t<T> transpose() const
        {
            Matrix_t<T> result(cols_, rows_);
            for (size_t i = 0; i < rows_; ++i)
            {
                for (size_t j = 0; j < cols_; ++j)
                {
                    result.at(j, i) = elements_[i][j];
                }
            }
            return result;
        }
  
        Matrix_t<T>& operator+=(const Matrix_t<T>& other)
        {
            assert((rows_ == other.rows_ && cols_ == other.cols_) && "Matrix dimensions must agree for addition");

            for (size_t i = 0; i < rows_; ++i)
            {
                for (size_t j = 0; j < cols_; ++j)
                {
                    elements_[i][j] += other.at(i, j);
                }
            }
            return *this;
        }
        Matrix_t<T>& operator-=(const Matrix_t<T>& other)
        {
            assert((rows_ == other.rows_ && cols_ == other.cols_) && "Matrix dimensions must agree for subtraction");

            for (size_t i = 0; i < rows_; ++i)
            {
                for (size_t j = 0; j < cols_; ++j)
                {
                    elements_[i][j] -= other.at(i, j);
                }
            }
            return *this;
        }

        Matrix_t<T> elementWiseMultiplication(const Matrix_t<T>& otherMatrix) const
        {
            assert((rows_ == otherMatrix.rows_ && cols_ == otherMatrix.cols_) && "Matrix dimensions must agree for element-wise multiplication");
            Matrix_t<T> result(rows_, cols_);
            for (size_t row = 0; row < rows_; ++row)
            {
                for (size_t col = 0; col < cols_; ++col)
                {
                    result.at(row, col) = elements_[row][col] * otherMatrix.at(row, col);
                }
            }
        
            return result;
        }

    private:
        size_t rows_, cols_;
        std::vector<std::vector<T>> elements_;
    };
    struct Matrix2x2
    {
        std::array<std::array<double, 2>, 2> elements;

        Matrix2x2()
        : elements{{{0, 0}, {0, 0}}} {}
        Matrix2x2(double x)
        : elements{{{x, 0}, {0, x}}} {}
        Matrix2x2(std::array<std::array<double, 2>, 2> elements)
        : elements(elements) {}
        Matrix2x2(double a, double b, double c, double d)
        : elements{{{a, b}, {c, d}}} {}

        constexpr bool operator==(const Matrix2x2& other) const {
            for (size_t i = 0; i < 2; ++i) {
                for (size_t j = 0; j < 2; ++j) {
                    if (elements[i][j] != other.elements[i][j]) {
                        return false;
                    }
                }
            }
            return true;
        }
        constexpr bool operator!=(const Matrix2x2& other) const {
            return !(*this == other);
        }

        Matrix2x2 operator+(const Matrix2x2& other) const {
            Matrix2x2 result{};
            for (size_t i = 0; i < 2; ++i) {
                for (size_t j = 0; j < 2; ++j) {
                    result.elements[i][j] = elements[i][j] + other.elements[i][j];
                }
            }
            return result;
        }
        Matrix2x2 operator-(const Matrix2x2& other) const {
            Matrix2x2 result{};
            for (size_t i = 0; i < 2; ++i) {
                for (size_t j = 0; j < 2; ++j) {
                    result.elements[i][j] = elements[i][j] - other.elements[i][j];
                }
            }
            return result;
        }
        Matrix2x2 operator*(const Matrix2x2& other) const {
            Matrix2x2 result{};
            for (size_t i = 0; i < 2; ++i) {
                for (size_t j = 0; j < 2; ++j) {
                    result.elements[i][j] = 0;
                    for (size_t k = 0; k < 2; ++k) {
                        result.elements[i][j] += elements[i][k] * other.elements[k][j];
                    }
                }
            }
            return result;
        }
        Matrix2x2 operator/(const Matrix2x2& other) const {
            Matrix2x2 result{};
            for (size_t i = 0; i < 2; ++i) {
                for (size_t j = 0; j < 2; ++j) {
                    result.elements[i][j] = elements[i][j] / other.elements[i][j];
                }
            }
            return result;
        }
        Matrix2x2 operator*(const double& scalar) const {
            Matrix2x2 result{};
            for (size_t i = 0; i < 2; ++i) {
                for (size_t j = 0; j < 2; ++j) {
                    result.elements[i][j] = elements[i][j] * scalar;
                }
            }
            return result;
        }
        Matrix2x2 operator/(const double& scalar) const {
            Matrix2x2 result{};
            for (size_t i = 0; i < 2; ++i) {
                for (size_t j = 0; j < 2; ++j) {
                    result.elements[i][j] = elements[i][j] / scalar;
                }
            }
            return result;
        }
        Matrix2x2 transpose() const {
            Matrix2x2 result{};
            for (size_t i = 0; i < 2; ++i) {
                for (size_t j = 0; j < 2; ++j) {
                    result.elements[j][i] = elements[i][j];
                }
            }
            return result;
        }
        constexpr double determinant() const {
            return elements[0][0] * elements[1][1] - elements[0][1] * elements[1][0];
        }
        Matrix2x2 inverse() const {
            double det = this->determinant();

            if (det == 0) {
                return Matrix2x2();
            }

            double invDet = 1 / det;
            return Matrix2x2(elements[1][1] * invDet, -elements[0][1] * invDet, -elements[1][0] * invDet, elements[0][0] * invDet);
        }

        Matrix2x2 operator*(const Vector2& v) const {
            return Matrix2x2(elements[0][0] * v.x, elements[0][1] * v.y, elements[1][0] * v.x, elements[1][1] * v.y);
        }
        Matrix2x2 operator/(const Vector2& v) const {
            return Matrix2x2(elements[0][0] / v.x, elements[0][1] / v.y, elements[1][0] / v.x, elements[1][1] / v.y);
        }

        static Matrix2x2 identity() {
            return Matrix2x2(1);
        }
    };
    struct Matrix3x3
    {
        std::array<std::array<double, 3>, 3> elements;

        Matrix3x3()
        : elements{{{0, 0, 0}, {0, 0, 0}, {0, 0, 0}}} {}
        Matrix3x3(double x)
        : elements{{{x, 0, 0}, {0, x, 0}, {0, 0, x}}} {}
        Matrix3x3(std::array<std::array<double, 3>, 3> elements)
        : elements(elements) {}
        Matrix3x3(double a, double b, double c, double d, double e, double f, double g, double h, double i)
        : elements{{{a, b, c}, {d, e, f}, {g, h, i}}} {}
        Matrix3x3(const Matrix2x2& m)
        : elements{{{m.elements[0][0], m.elements[0][1], 0},
                    {m.elements[1][0], m.elements[1][1], 0},
                    {0, 0, 1}}} {}

        constexpr bool operator==(const Matrix3x3& other) const {
            for (size_t i = 0; i < 3; ++i) {
                for (size_t j = 0; j < 3; ++j) {
                    if (elements[i][j] != other.elements[i][j]) {
                        return false;
                    }
                }
            }
            return true;
        }
        constexpr bool operator!=(const Matrix3x3& other) const {
            return !(*this == other);
        }

        Matrix3x3 operator+(const Matrix3x3& other) const {
            Matrix3x3 result{};
            for (size_t i = 0; i < 3; ++i) {
                for (size_t j = 0; j < 3; ++j) {
                    result.elements[i][j] = elements[i][j] + other.elements[i][j];
                }
            }
            return result;
        }
        Matrix3x3 operator-(const Matrix3x3& other) const {
            Matrix3x3 result{};
            for (size_t i = 0; i < 3; ++i) {
                for (size_t j = 0; j < 3; ++j) {
                    result.elements[i][j] = elements[i][j] - other.elements[i][j];
                }
            }
            return result;
        }
        Matrix3x3 operator*(const Matrix3x3& other) const {
            Matrix3x3 result{};
            for (size_t i = 0; i < 3; ++i) {
                for (size_t j = 0; j < 3; ++j) {
                    result.elements[i][j] = 0;
                    for (size_t k = 0; k < 3; ++k) {
                        result.elements[i][j] += elements[i][k] * other.elements[k][j];
                    }
                }
            }
            return result;
        }
        Matrix3x3 operator/(const Matrix3x3& other) const {
            Matrix3x3 result{};
            for (size_t i = 0; i < 3; ++i) {
                for (size_t j = 0; j < 3; ++j) {
                    result.elements[i][j] = elements[i][j] / other.elements[i][j];
                }
            }
            return result;
        }
        Matrix3x3 operator*(const double& scalar) const {
            Matrix3x3 result{};
            for (size_t i = 0; i < 3; ++i) {
                for (size_t j = 0; j < 3; ++j) {
                    result.elements[i][j] = elements[i][j] * scalar;
                }
            }
            return result;
        }
        Matrix3x3 operator/(const double& scalar) const {
            Matrix3x3 result{};
            for (size_t i = 0; i < 3; ++i) {
                for (size_t j = 0; j < 3; ++j) {
                    result.elements[i][j] = elements[i][j] / scalar;
                }
            }
            return result;
        }

        Matrix3x3 operator*(const Vector3& v) const {
            return Matrix3x3(elements[0][0] * v.x, elements[0][1] * v.y, elements[0][2] * v.z,
                            elements[1][0] * v.x, elements[1][1] * v.y, elements[1][2] * v.z,
                            elements[2][0] * v.x, elements[2][1] * v.y, elements[2][2] * v.z);
        }
        Matrix3x3 operator/(const Vector3& v) const {
            return Matrix3x3(elements[0][0] / v.x, elements[0][1] / v.y, elements[0][2] / v.z,
                            elements[1][0] / v.x, elements[1][1] / v.y, elements[1][2] / v.z,
                            elements[2][0] / v.x, elements[2][1] / v.y, elements[2][2] / v.z);
        }

        Matrix3x3 transpose() const {
            Matrix3x3 result{};
            for (size_t i = 0; i < 3; ++i) {
                for (size_t j = 0; j < 3; ++j) {
                    result.elements[j][i] = elements[i][j];
                }
            }
            return result;
        }
        constexpr double determinant() const {
            return elements[0][0] * (elements[1][1] * elements[2][2] - elements[1][2] * elements[2][1]) -
                elements[0][1] * (elements[1][0] * elements[2][2] - elements[1][2] * elements[2][0]) +
                elements[0][2] * (elements[1][0] * elements[2][1] - elements[1][1] * elements[2][0]);
        }
        Matrix3x3 inverse() const {
            double det = this->determinant();

            if (det == 0) {
                return Matrix3x3();
            }

            double invDet = 1 / det;
            return Matrix3x3((elements[1][1] * elements[2][2] - elements[1][2] * elements[2][1]) * invDet,
                            (elements[0][2] * elements[2][1] - elements[0][1] * elements[2][2]) * invDet,
                            (elements[0][1] * elements[1][2] - elements[0][2] * elements[1][1]) * invDet,
                            (elements[1][2] * elements[2][0] - elements[1][0] * elements[2][2]) * invDet,
                            (elements[0][0] * elements[2][2] - elements[0][2] * elements[2][0]) * invDet,
                            (elements[0][2] * elements[1][0] - elements[0][0] * elements[1][2]) * invDet,
                            (elements[1][0] * elements[2][1] - elements[1][1] * elements[2][0]) * invDet,
                            (elements[0][1] * elements[2][0] - elements[0][0] * elements[2][1]) * invDet,
                            (elements[0][0] * elements[1][1] - elements[0][1] * elements[1][0]) * invDet);
        }

        static Matrix3x3 identity() {
            return Matrix3x3(1);
        }
        Matrix2x2 toMatrix2x2() const {
            return Matrix2x2(elements[0][0], elements[0][1], elements[1][0], elements[1][1]);
        }
    };
    struct Matrix4x4
    {
        std::array<std::array<double, 4>, 4> elements;

        Matrix4x4()
        : elements{{{0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}}} {}
        Matrix4x4(double x)
        : elements{{{x, 0, 0, 0}, {0, x, 0, 0}, {0, 0, x, 0}, {0, 0, 0, x}}} {}
        Matrix4x4(std::array<std::array<double, 4>, 4> elements)
        : elements(elements) {}
        Matrix4x4(double a, double b, double c, double d, double e, double f, double g, double h, double i, double j, double k, double l, double m, double n, double o, double p)
        : elements{{{a, b, c, d}, {e, f, g, h}, {i, j, k, l}, {m, n, o, p}}} {}
        Matrix4x4(const Matrix3x3& m)
        : elements{{{m.elements[0][0], m.elements[0][1], m.elements[0][2], 0},
                    {m.elements[1][0], m.elements[1][1], m.elements[1][2], 0},
                    {m.elements[2][0], m.elements[2][1], m.elements[2][2], 0},
                    {0, 0, 0, 1}}} {}

        constexpr bool operator==(const Matrix4x4& other) const {
            for (size_t i = 0; i < 4; ++i) {
                for (size_t j = 0; j < 4; ++j) {
                    if (elements[i][j] != other.elements[i][j]) {
                        return false;
                    }
                }
            }
            return true;
        }
        constexpr bool operator!=(const Matrix4x4& other) const {
            return !(*this == other);
        }
        Matrix4x4 operator+(const Matrix4x4& other) const {
            Matrix4x4 result{};
            for (size_t i = 0; i < 4; ++i) {
                for (size_t j = 0; j < 4; ++j) {
                    result.elements[i][j] = elements[i][j] + other.elements[i][j];
                }
            }
            return result;
        }
        Matrix4x4 operator-(const Matrix4x4& other) const {
            Matrix4x4 result{};
            for (size_t i = 0; i < 4; ++i) {
                for (size_t j = 0; j < 4; ++j) {
                    result.elements[i][j] = elements[i][j] - other.elements[i][j];
                }
            }
            return result;
        }
        Matrix4x4 operator*(const Matrix4x4& other) const {
            Matrix4x4 result{};
            for (size_t i = 0; i < 4; ++i) {
                for (size_t j = 0; j < 4; ++j) {
                    result.elements[i][j] = 0;
                    for (size_t k = 0; k < 4; ++k) {
                        result.elements[i][j] += elements[i][k] * other.elements[k][j];
                    }
                }
            }
            return result;
        }
        Matrix4x4 operator/(const Matrix4x4& other) const {
            Matrix4x4 result{};
            for (size_t i = 0; i < 4; ++i) {
                for (size_t j = 0; j < 4; ++j) {
                    result.elements[i][j] = elements[i][j] / other.elements[i][j];
                }
            }
            return result;
        }
        Matrix4x4 operator*(const double& scalar) const {
            Matrix4x4 result{};
            for (size_t i = 0; i < 4; ++i) {
                for (size_t j = 0; j < 4; ++j) {
                    result.elements[i][j] = elements[i][j] * scalar;
                }
            }
            return result;
        }
        Matrix4x4 operator/(const double& scalar) const {
            Matrix4x4 result{};
            for (size_t i = 0; i < 4; ++i) {
                for (size_t j = 0; j < 4; ++j) {
                    result.elements[i][j] = elements[i][j] / scalar;
                }
            }
            return result;
        }

        Matrix4x4 operator*(const Vector4& v) const {
            return Matrix4x4(elements[0][0] * v.x, elements[0][1] * v.y, elements[0][2] * v.z, elements[0][3] * v.w,
                            elements[1][0] * v.x, elements[1][1] * v.y, elements[1][2] * v.z, elements[1][3] * v.w,
                            elements[2][0] * v.x, elements[2][1] * v.y, elements[2][2] * v.z, elements[2][3] * v.w,
                            elements[3][0] * v.x, elements[3][1] * v.y, elements[3][2] * v.z, elements[3][3] * v.w);
        }
        Matrix4x4 operator/(const Vector4& v) const {
            return Matrix4x4(elements[0][0] / v.x, elements[0][1] / v.y, elements[0][2] / v.z, elements[0][3] / v.w,
                            elements[1][0] / v.x, elements[1][1] / v.y, elements[1][2] / v.z, elements[1][3] / v.w,
                            elements[2][0] / v.x, elements[2][1] / v.y, elements[2][2] / v.z, elements[2][3] / v.w,
                            elements[3][0] / v.x, elements[3][1] / v.y, elements[3][2] / v.z, elements[3][3] / v.w);
        }

        Matrix4x4 transpose() const {
            Matrix4x4 result{};
            for (size_t i = 0; i < 4; ++i) {
                for (size_t j = 0; j < 4; ++j) {
                    result.elements[j][i] = elements[i][j];
                }
            }
            return result;
        }
        constexpr double determinant() const {
            return elements[0][0] * (elements[1][1] * (elements[2][2] * elements[3][3] - elements[2][3] * elements[3][2]) -
                                    elements[1][2] * (elements[2][1] * elements[3][3] - elements[2][3] * elements[3][1]) +
                                    elements[1][3] * (elements[2][1] * elements[3][2] - elements[2][2] * elements[3][1])) -
                elements[0][1] * (elements[1][0] * (elements[2][2] * elements[3][3] - elements[2][3] * elements[3][2]) -
                                    elements[1][2] * (elements[2][0] * elements[3][3] - elements[2][3] * elements[3][0]) +
                                    elements[1][3] * (elements[2][0] * elements[3][2] - elements[2][2] * elements[3][0])) +
                elements[0][2] * (elements[1][0] * (elements[2][1] * elements[3][3] - elements[2][3] * elements[3][1]) -
                                    elements[1][1] * (elements[2][0] * elements[3][3] - elements[2][3] * elements[3][0]) +
                                    elements[1][3] * (elements[2][0] * elements[3][1] - elements[2][1] * elements[3][0])) -
                elements[0][3] * (elements[1][0] * (elements[2][1] * elements[3][2] - elements[2][2] * elements[3][1]) -
                                    elements[1][1] * (elements[2][0] * elements[3][2] - elements[2][2] * elements[3][0]) +
                                    elements[1][2] * (elements[2][0] * elements[3][1] - elements[2][1] * elements[3][0]));
        }
        Matrix4x4 inverse() const {
            double det = determinant();

            if (det == 0) {
                return Matrix4x4();
            }

            /* TO DO */
        }

        static Matrix4x4 identity() {
            return Matrix4x4(1);
        }
        Matrix3x3 toMatrix3x3() const {
            return Matrix3x3(elements[0][0], elements[0][1], elements[0][2],
                            elements[1][0], elements[1][1], elements[1][2],
                            elements[2][0], elements[2][1], elements[2][2]);
        }
    };
        
} // namespace TarsMath

#endif // TARS_MATH_MATRIX_COMPONENT_HPP