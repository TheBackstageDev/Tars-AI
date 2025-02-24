#ifndef TARS_MATH_VECTOR_COMPONENT
#define TARS_MATH_VECTOR_COMPONENT

#include <cmath>

struct Vector2
{
    union
    {
        struct
        {
            float x, y;
        };
        struct
        {
            float r, g;
        };
    };

    Vector2() : x(0), y(0) {}
    Vector2(float x, float y) : x(x), y(y) {}

    constexpr bool operator==(const Vector2& v) const
    {
        return !(*this == v);
    }

    constexpr bool operator!=(const Vector2& v) const
    {
        return x != v.x || y != v.y;
    }

    Vector2 operator+(const Vector2& v) const
    {
        return Vector2(x + v.x, y + v.y);
    }

    Vector2 operator-(const Vector2& v) const
    {
        return Vector2(x - v.x, y - v.y);
    }

    Vector2 operator*(float s) const
    {
        return Vector2(x * s, y * s);
    }

    Vector2 operator/(float s) const
    {
        return Vector2(x / s, y / s);
    }

    Vector2& operator+=(const Vector2& v)
    {
        x += v.x;
        y += v.y;
        return *this;
    }

    Vector2& operator-=(const Vector2& v)
    {
        x -= v.x;
        y -= v.y;
        return *this;
    }

    Vector2& operator*=(float s)
    {
        x *= s;
        y *= s;
        return *this;
    }

    Vector2& operator/=(float s)
    {
        x /= s;
        y /= s;
        return *this;
    }

    Vector2 normalized() const
    {
        return *this / length();
    }

    float length() const
    {
        return std::sqrt(x * x + y * y);
    }

    constexpr float dot(const Vector2& v) const
    {
        return x * v.x + y * v.y;
    }
};
struct Vector3
{
    union 
    {
        struct {
            float x, y, z;
        };
        struct {
            float r, g, b;
        };
    };

    Vector3() : x(0), y(0), z(0) {}
    Vector3(float x, float y, float z) : x(x), y(y), z(z) {}

    constexpr bool operator==(const Vector3& v) const
    {
        return !(*this == v);
    }

    constexpr bool operator!=(const Vector3& v) const
    {
        return x != v.x || y != v.y || z != v.z;
    }

    Vector3 operator+(const Vector3& v) const
    {
        return Vector3(x + v.x, y + v.y, z + v.z);
    }

    Vector3 operator-(const Vector3& v) const
    {
        return Vector3(x - v.x, y - v.y, z - v.z);
    }

    Vector3 operator*(float s) const
    {
        return Vector3(x * s, y * s, z * s);
    }

    Vector3 operator/(float s) const
    {
        return Vector3(x / s, y / s, z / s);
    }

    Vector3& operator+=(const Vector3& v)
    {
        x += v.x;
        y += v.y;
        z += v.z;
        return *this;
    }

    Vector3& operator-=(const Vector3& v)
    {
        x -= v.x;
        y -= v.y;
        z -= v.z;
        return *this;
    }

    Vector3& operator*=(float s)
    {
        x *= s;
        y *= s;
        z *= s;
        return *this;
    }

    Vector3& operator/=(float s)
    {
        x /= s;
        y /= s;
        z /= s;
        return *this;
    }

    Vector3 normalized() const
    {
        return *this / length();
    }

    float length() const
    {
        return std::sqrt(x * x + y * y + z * z);
    }

    constexpr float dot(const Vector3& v) const
    {
        return x * v.x + y * v.y + z * v.z;
    }

    Vector2 xy() const
    {
        return Vector2(x, y);
    }

    Vector2 rg() const
    {
        return Vector2(r, g);
    }
};
struct Vector4
{
    union
    {
        struct
        {
            float x, y, z, w;
        };
        struct
        {
            float r, g, b, a;
        };
    };

    Vector4() : x(0), y(0), z(0), w(0) {}
    Vector4(float x, float y, float z, float w) : x(x), y(y), z(z), w(w) {}

    constexpr bool operator==(const Vector4& v) const
    {
        return !(*this == v);
    }

    constexpr bool operator!=(const Vector4& v) const
    {
        return x != v.x || y != v.y || z != v.z || w != v.w;
    }

    Vector4 operator+(const Vector4& v) const
    {
        return Vector4(x + v.x, y + v.y, z + v.z, w + v.w);
    }

    Vector4 operator-(const Vector4& v) const
    {
        return Vector4(x - v.x, y - v.y, z - v.z, w - v.w);
    }

    Vector4 operator*(float s) const
    {
        return Vector4(x * s, y * s, z * s, w * s);
    }

    Vector4 operator/(float s) const
    {
        return Vector4(x / s, y / s, z / s, w / s);
    }

    Vector4& operator+=(const Vector4& v)
    {
        x += v.x;
        y += v.y;
        z += v.z;
        w += v.w;
        return *this;
    }

    Vector4& operator-=(const Vector4& v)
    {
        x -= v.x;
        y -= v.y;
        z -= v.z;
        w -= v.w;
        return *this;
    }

    Vector4& operator*=(float s)
    {
        x *= s;
        y *= s;
        z *= s;
        w *= s;
        return *this;
    }

    Vector4& operator/=(float s)
    {
        x /= s;
        y /= s;
        z /= s;
        w /= s;
        return *this;
    }

    Vector4 normalized() const
    {
        return *this / length();
    }

    float length() const
    {
        return std::sqrt(x * x + y * y + z * z + w * w);
    }

    constexpr float dot(const Vector4& v) const
    {
        return x * v.x + y * v.y + z * v.z + w * v.w;
    }

    Vector3 xyz() const
    {
        return Vector3(x, y, z);
    }

    Vector2 xy() const
    {
        return Vector2(x, y);
    }

    Vector3 rgb() const
    {
        return Vector3(r, g, b);
    }

    Vector2 rg() const
    {
        return Vector2(r, g);
    }
};

#endif // TARS_MATH_VECTOR_COMPONENT
