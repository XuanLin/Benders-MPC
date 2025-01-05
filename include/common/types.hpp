#pragma once

#include <Eigen/Dense>

namespace optimization {

// Dynamic size constant from Eigen
constexpr int Dynamic = Eigen::Dynamic;

// Generic matrix/vector types with template parameters
template <typename Scalar, int Rows, int Cols = Rows>
using Matrix = Eigen::Matrix<Scalar, Rows, Cols>;

template <typename Scalar, int Size>
using Vector = Matrix<Scalar, Size, 1>;

// Common double-precision matrix types
template <int Rows, int Cols = Rows>
using MatrixXd = Matrix<double, Rows, Cols>;

using Matrix2d = MatrixXd<2>;
using Matrix3d = MatrixXd<3>;
using Matrix4d = MatrixXd<4>;
using Matrix34d = MatrixXd<3, 4>;
using Matrix44d = MatrixXd<4, 4>;

// Common double-precision vector types
template <int Size>
using VectorXd = Vector<double, Size>;

using Vector2d = VectorXd<2>;
using Vector3d = VectorXd<3>;
using Vector4d = VectorXd<4>;

// Dynamic-sized types
using MatrixDyn = MatrixXd<Dynamic, Dynamic>;
using VectorDyn = VectorXd<Dynamic>;

// Integer types if needed
template <int Size>
using VectorXi = Vector<int, Size>;

using Vector2i = VectorXi<2>;
using Vector3i = VectorXi<3>;
using Vector4i = VectorXi<4>;

} // namespace optimization

