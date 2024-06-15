#include <Eigen/Dense>

/// Eigen dynamic
constexpr int Dynamic = ::Eigen::Dynamic;

/// Fixed-size matrix (general)
template <typename S, int N, int M = N>
using Matrix = ::Eigen::Matrix<S, N, M>;

/// Fixed-size vector (general)
template <typename S, int N>
using Vector = Matrix<S, N, 1>;

// Fixed-size matrix (double)
template <int N, int M = N>
using Matrixd = Matrix<double, N, M>;
using Matrix1d = Matrixd<1>;
using Matrix2d = Matrixd<2>;
using Matrix3d = Matrixd<3>;
using Matrix4d = Matrixd<4>;
using Matrix5d = Matrixd<5>;
using Matrix6d = Matrixd<6>;
using Matrix33d = Matrixd<3, 3>;
using Matrix34d = Matrixd<3, 4>;
using Matrix44d = Matrixd<4, 4>;
using Matrix66d = Matrixd<6, 6>;
/// Dynamic-size matrix (double)
using MatrixXd = Matrixd<Dynamic>;

// Fixed-size vector (double)
template <int N>
using Vectord = Vector<double, N>;
using Vector1d = Vectord<1>;
using Vector2d = Vectord<2>;
using Vector3d = Vectord<3>;
using Vector4d = Vectord<4>;
using Vector5d = Vectord<5>;
using Vector6d = Vectord<6>;
using Vector7d = Vectord<7>;
/// Dynamic-size vector (double)
using VectorXd = Vectord<Dynamic>;

/// Fixed-size vector (integer)
template <int N>
using Vectori = Vector<int, N>;
using Vector1i = Vectori<1>;
using Vector2i = Vectori<2>;
using Vector3i = Vectori<3>;
using Vector4i = Vectori<4>;
using Vector5i = Vectori<5>;
using Vector6i = Vectori<6>;
/// Dynamic-size vector (integer)
using VectorXi = Vectori<Dynamic>;