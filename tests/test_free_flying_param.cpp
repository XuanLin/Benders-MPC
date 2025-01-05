#include <iostream>
#include <Eigen/Core>
#include "optimization/util/solver_params.hpp"
#include "problems/free_flying/free_flying_param.hpp"
#include "common/types.hpp"

using namespace std;
using namespace optimization;

int main() {
    // Test for 3 obstacles
    std::cout << "=== 3 Obstacles ===\n";
    FlyingRobotParams params3(0.0, 0.0, 3);
    std::cout << "Q:\n" << params3.Q << "\n\n";
    std::cout << "R:\n" << params3.R << "\n\n";
    std::cout << "Qn:\n" << params3.Qn << "\n\n";
    std::cout << "E:\n" << params3.E << "\n\n";
    std::cout << "F:\n" << params3.F << "\n\n";
    std::cout << "G:\n" << params3.G << "\n\n";
    std::cout << "H1:\n" << params3.H1 << "\n\n";
    std::cout << "H2:\n" << params3.H2 << "\n\n";
    std::cout << "H3:\n" << params3.H3 << "\n\n";

    // Test for 6 obstacles
    std::cout << "=== 6 Obstacles ===\n";
    FlyingRobotParams params6(0.0, 0.0, 6);
    std::cout << "Q:\n" << params6.Q << "\n\n";
    std::cout << "R:\n" << params6.R << "\n\n";
    std::cout << "Qn:\n" << params6.Qn << "\n\n";
    std::cout << "E:\n" << params6.E << "\n\n";
    std::cout << "F:\n" << params6.F << "\n\n";
    std::cout << "G:\n" << params6.G << "\n\n";
    std::cout << "H1:\n" << params6.H1 << "\n\n";
    std::cout << "H2:\n" << params6.H2 << "\n\n";
    std::cout << "H3:\n" << params6.H3 << "\n\n";

    // Test for 9 obstacles
    std::cout << "=== 9 Obstacles ===\n";
    FlyingRobotParams params9(0.0, 0.0, 9);
    std::cout << "Q:\n" << params9.Q << "\n\n";
    std::cout << "R:\n" << params9.R << "\n\n";
    std::cout << "Qn:\n" << params9.Qn << "\n\n";
    std::cout << "E:\n" << params9.E << "\n\n";
    std::cout << "F:\n" << params9.F << "\n\n";
    std::cout << "G:\n" << params9.G << "\n\n";
    std::cout << "H1:\n" << params9.H1 << "\n\n";
    std::cout << "H2:\n" << params9.H2 << "\n\n";
    std::cout << "H3:\n" << params9.H3 << "\n\n";

    return 0;
}

